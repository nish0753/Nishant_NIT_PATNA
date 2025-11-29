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

    if isinstance(image_source, str):
        # It's a URL - download synchronously (could be made async but fast enough usually)
        # For true async, we'd use aiohttp, but keeping it simple for now as requests is fast for single files
        loop = asyncio.get_event_loop()
        file_data, mime_type = await loop.run_in_executor(None, download_image, image_source)
    else:
        # It's raw bytes
        file_data = image_source
        if not mime_type:
            raise ValueError("Mime type must be provided for file uploads.")

    # Apply preprocessing if it's an image
    if mime_type and mime_type.startswith('image/'):
        print("Applying image preprocessing...")
        loop = asyncio.get_event_loop()
        file_data = await loop.run_in_executor(None, preprocess_image, file_data)
        # Mime type remains image/jpeg effectively after processing
        mime_type = 'image/jpeg' 

    # Switching to gemini-1.5-flash as requested
    # --- PAGE-BY-PAGE PROCESSING LOGIC ---
    import io
    import gc
    import hashlib

    # If it's a PDF, we convert to images using pdf2image
    if mime_type == 'application/pdf':
        print("DEBUG: PDF detected. Converting to images using pdf2image.")
        try:
            # Run pdf2image in executor as it can be CPU intensive
            loop = asyncio.get_event_loop()
            # 300 DPI is good for OCR
            images = await loop.run_in_executor(None, lambda: convert_from_bytes(file_data, dpi=300))
            num_pages = len(images)
            
            print(f"DEBUG: PDF converted. Total pages: {num_pages}")

            all_pagewise_items = []
            total_item_count = 0
            seen_page_hashes = set()
            
            # Semaphore to limit concurrency (Reduced to 1 for maximum stability on free tier)
            sem = asyncio.Semaphore(1)

            async def process_page(i, image):
                async with sem:
                    # Add jitter
                    await asyncio.sleep(1) 
                    
                    print(f"DEBUG: Processing Page {i+1}/{num_pages}...")
                    page_data = None
                    try:
                        # Convert PIL image to bytes
                        with io.BytesIO() as output:
                            image.save(output, format="JPEG", quality=95)
                            page_data = output.getvalue()

                        # Calculate hash to detect duplicates
                        page_hash = hashlib.md5(page_data).hexdigest()
                        
                        if page_hash in seen_page_hashes:
                            print(f"DEBUG: Page {i+1} is a duplicate. Skipping.")
                            return None
                        seen_page_hashes.add(page_hash)

                        # Call Gemini for this single page (as image)
                        print(f"DEBUG: Calling Gemini for Page {i+1}...")
                        # Note: We send 'image/jpeg' because we converted it
                        page_result = await call_gemini_api_async(page_data, 'image/jpeg')
                        print(f"DEBUG: Gemini returned for Page {i+1}.")
                        
                        if page_result:
                            # Tag with page number
                            if "pagewise_line_items" in page_result:
                                for p_item in page_result["pagewise_line_items"]:
                                    p_item["page_no"] = str(i + 1)
                            return page_result
                        return None

                    except Exception as e:
                        print(f"ERROR processing page {i+1}: {e}")
                        return None
                    finally:
                        if page_data:
                            del page_data
                        # Explicitly close the image to free memory
                        if image:
                            image.close()

            # Create tasks for all pages
            tasks = [process_page(i, img) for i, img in enumerate(images)]
            results = await asyncio.gather(*tasks)

            # Aggregate results
            for res in results:
                if res:
                    if "pagewise_line_items" in res:
                        all_pagewise_items.extend(res["pagewise_line_items"])
                    if "total_item_count" in res:
                        total_item_count += res["total_item_count"]
            
            # Explicit GC after batch
            gc.collect()

            print("DEBUG: PDF processing complete. Returning aggregated results.")
            # Sort by page number to maintain order
            all_pagewise_items.sort(key=lambda x: int(x.get("page_no", "0")))

            return {
                "pagewise_line_items": all_pagewise_items,
                "total_item_count": total_item_count
            }

        except Exception as e:
            print(f"ERROR: PDF Processing Failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to single-shot if PDF splitting fails
            pass

    # --- SINGLE SHOT LOGIC (Images or Fallback) ---
    return await call_gemini_api_async(file_data, mime_type)


async def call_gemini_api_async(file_data, mime_type):
    """Helper function to call Gemini for a single file/page async."""
    # Updated to gemini-2.0-flash-exp
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    prompt = """
    You are an expert data extraction agent. Your task is to extract line item details from the provided bill/invoice (image or PDF).

    GOAL: Extract all purchasable items such that the sum of their 'item_amount' equals the Final Bill Total.

    CRITICAL INSTRUCTIONS (Read Carefully):
    1. **Distinguish Amounts vs. Identifiers**: 
       - Do NOT confuse "Invoice Number", "Date", "Time", or "ID codes" with monetary amounts. 
       - Look for currency symbols ($, ₹, etc.) or columns labeled "Amount", "Total", "Price" to confirm values.
    2. **No Subtotals**: Do NOT extract 'Sub-total', 'Total', 'Tax', or 'Discount' lines as items. Only extract the individual products/services that make up the bill.
    3. **Charges are Items**: DO extract distinct charges like 'Pharmacy Charge', 'Consultation Fee', 'Shipping' if they are line items adding to the total.
    4. **Exact Values**: Extract 'item_amount' exactly as shown (post-discount if applicable to the line).

    FIELDS TO EXTRACT per item:
    - item_name: Name/Description (String). Preserve newlines if present.
    - item_amount: Net Amount (Float). The final value of this line item contributing to the bill total.
    - item_rate: Unit Price (Float).
    - item_quantity: Quantity (Float). Default to 1.0 if not specified but implied.

    PAGE CLASSIFICATION:
    - "Bill Detail": Contains detailed line items.
    - "Final Bill": Summary page with totals (if separate).
    - "Pharmacy": Specific to pharmacy bills.
    If unsure, default to "Bill Detail".

    Also calculate:
    - total_item_count: Total number of line items extracted across all pages.

    If the bill has multiple pages, treat this file as one document. If page numbers are visible, use them, otherwise default to "1".

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

    generation_config = genai.GenerationConfig(
        max_output_tokens=8192,
        response_mime_type="application/json"
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
            # Use async generate_content
            response = await model.generate_content_async(
                [
                    {'mime_type': mime_type, 'data': file_data},
                    prompt
                ],
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            # Clean up response text to ensure it's valid JSON
            try:
                text = response.text.strip()
            except Exception as e:
                print(f"Error reading response text: {e}")
                # Try to get text from candidates manually
                try:
                    text = response.candidates[0].content.parts[0].text
                except:
                    if attempt < max_retries - 1:
                        print("Retrying due to empty/blocked response...")
                        await asyncio.sleep(2)
                        continue
                    print("ERROR: Could not extract text from Gemini response. Skipping this page.")
                    return {"pagewise_line_items": [], "total_item_count": 0}

            # print(f"DEBUG: Raw Gemini Response (Attempt {attempt+1}):\n{text[:100]}...") 
            
            text = clean_json(text)
                
            try:
                data_dict = json.loads(text)
                
                # Recalculate total_item_count to be sure
                total_count = 0
                for page in data_dict.get("pagewise_line_items", []):
                    for item in page.get("bill_items", []):
                        total_count += 1
                
                data_dict['total_item_count'] = total_count
                
                return data_dict # Success!
                
            except json.JSONDecodeError as e:
                print(f"JSON Parse Error on attempt {attempt+1}: {e}")
                # Try fallback repair
                text_fixed = re.sub(r'"\s*\n\s*"', '",\n"', text)
                try:
                    data_dict = json.loads(text_fixed)
                    # Recalculate total_item_count
                    total_count = 0
                    for page in data_dict.get("pagewise_line_items", []):
                        for item in page.get("bill_items", []):
                            total_count += 1
                    data_dict['total_item_count'] = total_count
                    return data_dict
                except:
                    if attempt < max_retries - 1:
                        print(f"Retrying due to bad JSON...")
                        await asyncio.sleep(2)
                        continue
                    raise ValueError(f"Failed to parse JSON after {max_retries} attempts: {e}")

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "Quota exceeded" in error_str or "ResourceExhausted" in error_str:
                if attempt < max_retries - 1:
                    # Exponential backoff: 5, 10, 20, 40, 80...
                    wait_time = 5 * (2 ** attempt)
                    print(f"Quota exceeded (Attempt {attempt+1}/{max_retries}). Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    continue
            
            print(f"Gemini API Error: {e}")
            if attempt < max_retries - 1:
                 await asyncio.sleep(2)
                 continue
            raise ValueError(f"Gemini API failed: {e}")
