import os
import requests
import json
import io
import gc
import shutil
import glob
import logging
import asyncio
import time
import re

from google import genai
from google.genai import types
from dotenv import load_dotenv
from PIL import Image, ImageEnhance
from pdf2image import convert_from_bytes

try:
    import pytesseract as _pytesseract
except ImportError:
    _pytesseract = None

try:
    import json_repair
except ImportError:
    json_repair = None

load_dotenv()

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Configuration ---
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2"))
PDF_DPI = int(os.getenv("PDF_DPI", "200"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "10"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Configure Gemini
GENAI_API_KEY = os.getenv("GOOGLE_API_KEY")

# Create client once (reuse across calls)
_client = None

def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not GENAI_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY not found in environment variables.")
        _client = genai.Client(api_key=GENAI_API_KEY)
    return _client

# --- Tesseract Setup ---
def _setup_tesseract():
    """Configure tesseract path once at startup."""
    if _pytesseract is None:
        logger.warning("pytesseract not installed. OCR step will be skipped.")
        return None

    tesseract_path = shutil.which("tesseract")
    if tesseract_path:
        logger.info("Tesseract found at: %s", tesseract_path)
        _pytesseract.pytesseract.tesseract_cmd = tesseract_path
        return _pytesseract

    common_paths = ["/usr/bin/tesseract", "/usr/local/bin/tesseract", "/bin/tesseract"]
    for p in common_paths:
        if os.path.exists(p):
            logger.info("Tesseract found at: %s", p)
            _pytesseract.pytesseract.tesseract_cmd = p
            return _pytesseract

    logger.warning("Tesseract binary NOT found. OCR step will be skipped.")
    return None

pytesseract = _setup_tesseract()

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
        image = image.convert('L')
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)
        output_buffer = io.BytesIO()
        image.save(output_buffer, format='JPEG', quality=95)
        return output_buffer.getvalue()
    except Exception as e:
        logger.warning("Image preprocessing failed: %s", e)
        return image_bytes

def download_image(url: str) -> bytes:
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        content_type = response.headers.get('Content-Type', '')
        if not content_type.startswith('image/') and content_type != 'application/pdf':
            raise ValueError(f"Unsupported content type: {content_type}")
        return response.content, content_type
    except Exception as e:
        logger.error("Error downloading image: %s", e)
        raise

async def extract_bill_data(image_source, mime_type: str = None) -> dict:
    if not GENAI_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not found in environment variables.")

    # 1. Get Raw Data
    if isinstance(image_source, str):
        loop = asyncio.get_running_loop()
        file_data, mime_type = await loop.run_in_executor(None, download_image, image_source)
    else:
        file_data = image_source
        if not mime_type:
            raise ValueError("Mime type must be provided for file uploads.")

    # 2. Magic Byte Detection (Fix for application/octet-stream)
    if file_data and len(file_data) > 4:
        if file_data[:4] == b'%PDF':
            logger.debug("Detected PDF magic bytes. Overriding mime_type '%s' to 'application/pdf'", mime_type)
            mime_type = 'application/pdf'

    # 3. Prepare Content Parts (List of images) and OCR Context
    content_parts = []
    ocr_context = ""

    if mime_type == 'application/pdf':
        logger.info("PDF detected. Converting to images for batch processing.")
        try:
            loop = asyncio.get_running_loop()
            images = await loop.run_in_executor(None, lambda: convert_from_bytes(file_data, dpi=PDF_DPI))
            logger.info("PDF converted. Total pages: %d", len(images))

            # Smart Batching: Process pages in chunks to avoid Output Token Limit (8192)
            
            all_pagewise_items = []
            total_item_count = 0
            
            # Prepare tasks for parallel execution
            batch_tasks = []
            
            for i in range(0, len(images), BATCH_SIZE):
                batch_images = images[i : i + BATCH_SIZE]
                
                batch_content_parts = []
                batch_ocr_context = ""
                
                for j, image in enumerate(batch_images):
                    page_num = i + j + 1
                    
                    if pytesseract:
                        try:
                            text = await loop.run_in_executor(None, pytesseract.image_to_string, image)
                            batch_ocr_context += f"\n--- Page {page_num} Raw OCR Text ---\n{text}\n"
                        except Exception as e:
                            logger.warning("OCR Warning on page %d: %s", page_num, e)

                    with io.BytesIO() as output:
                        image.save(output, format="JPEG", quality=95)
                        img_bytes = output.getvalue()
                        batch_content_parts.append(
                            types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg')
                        )
                    image.close()

                # Call Gemini for this batch
                logger.info("Sending batch of %d page(s) to Gemini...", len(batch_content_parts))
                batch_result = await call_gemini_api_async(batch_content_parts, batch_ocr_context)
                
                # Aggregate results
                if batch_result:
                    if "pagewise_line_items" in batch_result:
                        all_pagewise_items.extend(batch_result["pagewise_line_items"])
                    if "total_item_count" in batch_result:
                        total_item_count += batch_result["total_item_count"]
                
                # Small delay to be nice to the API
                await asyncio.sleep(1)

            # Explicit GC
            gc.collect()
            
            return {
                "pagewise_line_items": all_pagewise_items,
                "total_item_count": total_item_count
            }

        except Exception as e:
            logger.error("PDF Processing Failed: %s", e)
            raise
    
    elif mime_type.startswith('image/'):
        logger.info("Applying image preprocessing...")
        loop = asyncio.get_running_loop()
        processed_data = await loop.run_in_executor(None, preprocess_image, file_data)
        
        # OCR for single image
        if pytesseract:
            try:
                import PIL.Image
                img_obj = PIL.Image.open(io.BytesIO(processed_data))
                text = await loop.run_in_executor(None, pytesseract.image_to_string, img_obj)
                ocr_context += f"\n--- Raw OCR Text ---\n{text}\n"
            except Exception as e:
                logger.warning("OCR Warning: %s", e)

        content_parts.append(
            types.Part.from_bytes(data=processed_data, mime_type='image/jpeg')
        )
        
        # Single image call
        return await call_gemini_api_async(content_parts, ocr_context)
    
    else:
        # Fallback
        content_parts.append(
            types.Part.from_bytes(data=file_data, mime_type=mime_type)
        )
        return await call_gemini_api_async(content_parts, ocr_context)


async def call_gemini_api_async(content_parts, ocr_context=""):
    """Call Gemini with a list of content parts and optional OCR context.
    Returns result_dict with extracted data."""
    client = _get_client()
    
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

    generation_config = types.GenerateContentConfig(
        max_output_tokens=8192,
        response_mime_type="application/json",
        response_json_schema={
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
        },
        safety_settings=[
            types.SafetySetting(category='HARM_CATEGORY_HARASSMENT', threshold='OFF'),
            types.SafetySetting(category='HARM_CATEGORY_HATE_SPEECH', threshold='OFF'),
            types.SafetySetting(category='HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold='OFF'),
            types.SafetySetting(category='HARM_CATEGORY_DANGEROUS_CONTENT', threshold='OFF'),
        ],
    )

    max_retries = MAX_RETRIES
    for attempt in range(max_retries):
        try:
            # Construct the full prompt content: [img1, img2, ..., text_prompt]
            full_content = content_parts + [prompt]
            
            response = await client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=full_content,
                config=generation_config,
            )
            
            try:
                text = response.text.strip()
            except Exception:
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
                logger.warning("JSON Parse Error: %s", e)
                # Try robust repair
                if json_repair:
                    try:
                        logger.debug("Attempting to repair JSON with json_repair...")
                        data_dict = json_repair.loads(text)
                    
                        # Validate structure after repair
                        if isinstance(data_dict, dict) and "pagewise_line_items" in data_dict:
                            total_count = 0
                            for page in data_dict.get("pagewise_line_items", []):
                                for item in page.get("bill_items", []):
                                    total_count += 1
                            data_dict['total_item_count'] = total_count
                            logger.info("JSON repair successful.")
                            return data_dict
                    except Exception as repair_e:
                        logger.warning("JSON Repair Failed: %s", repair_e)

                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
                raise

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "Quota exceeded" in error_str or "ResourceExhausted" in error_str:
                if attempt < max_retries - 1:
                    wait_time = min(5 * (2 ** attempt), 60)  # Cap at 60s
                    logger.warning("Quota exceeded (Attempt %d/%d). Retrying in %ds...", attempt + 1, max_retries, wait_time)
                    await asyncio.sleep(wait_time)
                    continue
            
            logger.error("Gemini API Error: %s", e)
            if attempt < max_retries - 1:
                 await asyncio.sleep(2)
                 continue
            raise ValueError(f"Gemini API failed after {max_retries} attempts: {e}")
