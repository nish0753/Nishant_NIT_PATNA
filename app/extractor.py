import os
import requests
import json
import google.generativeai as genai
from dotenv import load_dotenv
import re
from PIL import Image, ImageEnhance
import io
import asyncio
import time
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from pdf2image import convert_from_bytes

load_dotenv()

# Configure Gemini
GENAI_API_KEY = os.getenv("GOOGLE_API_KEY")

if GENAI_API_KEY:
    genai.configure(api_key=GENAI_API_KEY)

def clean_json(text: str) -> str:
    # Remove markdown code blocks
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        text = match.group(1)
    else:
        # Try to find the first { and last }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            text = text[start:end+1]

    # Fix missing commas between objects in a list: } { -> }, {
    text = re.sub(r'}\s*{', '}, {', text)
    
    # Attempt to fix trailing commas
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    
    return text

def preprocess_image(image_bytes: bytes) -> bytes:
    """
    Enhance image quality for better OCR/Extraction.
    - Grayscale
    - Contrast Enhancement
    - Sharpness Enhancement
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Enhance Contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)  # Increase contrast by 50%
        
        # Enhance Sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)  # Increase sharpness
        
        # Save back to bytes
        output_buffer = io.BytesIO()
        image.save(output_buffer, format='JPEG', quality=95)
        return output_buffer.getvalue()
    except Exception as e:
        print(f"Warning: Image preprocessing failed: {e}")
        return image_bytes  # Return original if failed

def download_image(url: str) -> bytes:
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        content_type = response.headers.get('Content-Type', '')
        if not content_type.startswith('image/') and content_type != 'application/pdf':
            raise ValueError(f"Unsupported content type: {content_type}. Only images and PDFs are supported.")
            
        return response.content, content_type
    except Exception as e:
        print(f"Error downloading image: {e}")
        raise

async def extract_bill_data(image_source, mime_type: str = None) -> dict:
    if not GENAI_API_KEY:
        raise Exception("GOOGLE_API_KEY not found in environment variables.")

    # 1. Get Raw Data
    if isinstance(image_source, str):
        loop = asyncio.get_event_loop()
        file_data, mime_type = await loop.run_in_executor(None, download_image, image_source)
    else:
        file_data = image_source
        if not mime_type:
            raise ValueError("Mime type must be provided for file uploads.")

    # 2. Prepare Content Parts (List of images) and OCR Context
    content_parts = []
    ocr_context = ""
    
    import io
    import gc
    from pdf2image import convert_from_bytes
    try:
        import pytesseract
    except ImportError:
        pytesseract = None
        print("WARNING: pytesseract not found. OCR step will be skipped.")

    if mime_type == 'application/pdf':
        print("DEBUG: PDF detected. Converting to images for batch processing.")
        try:
            loop = asyncio.get_event_loop()
            # Convert all pages to images
            # Reduced DPI to 200 to prevent OOM on Render (Free Tier 512MB RAM)
            images = await loop.run_in_executor(None, lambda: convert_from_bytes(file_data, dpi=200))
            print(f"DEBUG: PDF converted. Total pages: {len(images)}")

            for i, image in enumerate(images):
                # OCR Step (Hybrid Pipeline)
                if pytesseract:
                    try:
                        # Debug Tesseract Path
                        # print(f"DEBUG: Tesseract cmd: {pytesseract.pytesseract.tesseract_cmd}")
                        
                        # Run OCR in executor to avoid blocking
                        text = await loop.run_in_executor(None, pytesseract.image_to_string, image)
                        ocr_context += f"\n--- Page {i+1} Raw OCR Text ---\n{text}\n"
                    except Exception as e:
                        print(f"OCR Warning on page {i+1}: {e}")
                        # If tesseract is not found, try setting default path for Linux/Docker
                        if "not installed" in str(e) or "not in your PATH" in str(e):
                             print("DEBUG: Attempting to set default Tesseract path for Docker...")
                             pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
                             try:
                                 text = await loop.run_in_executor(None, pytesseract.image_to_string, image)
                                 ocr_context += f"\n--- Page {i+1} Raw OCR Text ---\n{text}\n"
                                 print("DEBUG: OCR succeeded with explicit path.")
                             except Exception as e2:
                                 print(f"OCR Retry Failed: {e2}")

                with io.BytesIO() as output:
                    image.save(output, format="JPEG", quality=95)
                    img_bytes = output.getvalue()
                    content_parts.append({'mime_type': 'image/jpeg', 'data': img_bytes})
                # Close image to free memory
                image.close()
            
            # Explicit GC
            gc.collect()

        except Exception as e:
            print(f"ERROR: PDF Conversion Failed: {e}")
            raise
    
    elif mime_type.startswith('image/'):
        print("Applying image preprocessing...")
        loop = asyncio.get_event_loop()
        processed_data = await loop.run_in_executor(None, preprocess_image, file_data)
        
        # OCR for single image
        if pytesseract:
            try:
                import PIL.Image
                img_obj = PIL.Image.open(io.BytesIO(processed_data))
                text = await loop.run_in_executor(None, pytesseract.image_to_string, img_obj)
                ocr_context += f"\n--- Raw OCR Text ---\n{text}\n"
            except Exception as e:
                print(f"OCR Warning: {e}")

        content_parts.append({'mime_type': 'image/jpeg', 'data': processed_data})
    
    else:
        # Fallback for other types (shouldn't happen based on download_image)
        content_parts.append({'mime_type': mime_type, 'data': file_data})

    # 3. Call Gemini (One Call for All Pages + OCR Context)
    print(f"DEBUG: Sending {len(content_parts)} page(s) to Gemini in a single request...")
    if ocr_context:
        print("DEBUG: Including OCR context in prompt.")
    
    return await call_gemini_api_async(content_parts, ocr_context)


async def call_gemini_api_async(content_parts, ocr_context=""):
    """Helper function to call Gemini with a list of content parts and optional OCR context."""
    # Updated to gemini-2.0-flash (Stable)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    base_prompt = """
    You are an expert data extraction agent. Your task is to extract line item details from the provided bill/invoice images.
    The input consists of one or more pages of a SINGLE document.

    GOAL: Extract all purchasable items from ALL pages such that the sum of their 'item_amount' equals the Final Bill Total.

    CRITICAL INSTRUCTIONS:
    1. **Distinguish Amounts vs. Identifiers**: Do NOT confuse IDs/Dates with monetary amounts.
    2. **No Subtotals**: Do NOT extract 'Sub-total', 'Total', 'Tax', or 'Discount' lines as items.
    3. **Charges are Items**: DO extract distinct charges like 'Pharmacy Charge', 'Shipping'.
    4. **Exact Values**: Extract 'item_amount' exactly as shown.
    5. **Page Mapping**: The input images correspond to pages 1, 2, 3... in order. Assign items to the correct 'page_no'.
    6. **Use OCR Context**: Raw OCR text is provided below to help you decipher blurry text or ambiguous numbers. Use it to validate your visual understanding.

    Output strictly in JSON format matching this structure:
    {
      "pagewise_line_items": [
        {
          "page_no": "1",
          "page_type": "Bill Detail",
          "bill_items": [
            { "item_name": "...", "item_amount": 0.0, "item_rate": 0.0, "item_quantity": 0.0 }
          ]
        }
      ],
      "total_item_count": 0
    }
    """
    
    if ocr_context:
        prompt = base_prompt + f"\n\n=== RAW OCR TEXT CONTEXT ===\n{ocr_context}"
    else:
        prompt = base_prompt

    generation_config = genai.GenerationConfig(
        max_output_tokens=8192,
        response_mime_type="application/json",
        response_schema={
            "type": "object",
            "properties": {
                "pagewise_line_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "page_no": {"type": "string"},
                            "page_type": {"type": "string"},
                            "bill_items": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "item_name": {"type": "string"},
                                        "item_amount": {"type": "number"},
                                        "item_rate": {"type": "number"},
                                        "item_quantity": {"type": "number"}
                                    },
                                    "required": ["item_name", "item_amount", "item_rate", "item_quantity"]
                                }
                            }
                        },
                        "required": ["page_no", "page_type", "bill_items"]
                    }
                },
                "total_item_count": {"type": "integer"}
            },
            "required": ["pagewise_line_items", "total_item_count"]
        }
    )

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    max_retries = 10
    for attempt in range(max_retries):
        try:
            # Construct the full prompt content: [img1, img2, ..., text_prompt]
            full_content = content_parts + [prompt]
            
            response = await model.generate_content_async(
                full_content,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            try:
                text = response.text.strip()
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
                return {"pagewise_line_items": [], "total_item_count": 0}

            text = clean_json(text)
                
            try:
                data_dict = json.loads(text)
                
                # Recalculate total_item_count
                total_count = 0
                for page in data_dict.get("pagewise_line_items", []):
                    for item in page.get("bill_items", []):
                        total_count += 1
                data_dict['total_item_count'] = total_count
                
                return data_dict

            except json.JSONDecodeError as e:
                print(f"JSON Parse Error: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
                raise

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "Quota exceeded" in error_str or "ResourceExhausted" in error_str:
                if attempt < max_retries - 1:
                    wait_time = 5 * (2 ** attempt)
                    print(f"Quota exceeded (Attempt {attempt+1}/{max_retries}). Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    continue
            
            print(f"Gemini API Error: {e}")
            if attempt < max_retries - 1:
                 await asyncio.sleep(2)
                 continue
            raise ValueError(f"Gemini API failed: {e}")
