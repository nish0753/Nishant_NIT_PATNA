import os
import requests
import json
import google.generativeai as genai
from dotenv import load_dotenv
import re
from PIL import Image, ImageEnhance
import io

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

def extract_bill_data(image_source, mime_type: str = None) -> dict:
    if not GENAI_API_KEY:
        raise Exception("GOOGLE_API_KEY not found in environment variables.")

    if isinstance(image_source, str):
        # It's a URL
        file_data, mime_type = download_image(image_source)
    else:
        # It's raw bytes
        file_data = image_source
        if not mime_type:
            raise ValueError("Mime type must be provided for file uploads.")

    # Apply preprocessing if it's an image
    if mime_type and mime_type.startswith('image/'):
        print("Applying image preprocessing...")
        file_data = preprocess_image(file_data)
        # Mime type remains image/jpeg effectively after processing
        mime_type = 'image/jpeg' 

    # Switching to gemini-flash-latest to avoid rate limits on experimental models
    # --- PAGE-BY-PAGE PROCESSING LOGIC ---
    import io
    from pypdf import PdfReader, PdfWriter

    # If it's a PDF, we split it page by page
    if mime_type == 'application/pdf':
        print("DEBUG: PDF detected. Starting page-by-page processing.")
        try:
            # Get total pages first
            with io.BytesIO(file_data) as f:
                pdf_reader = PdfReader(f)
                num_pages = len(pdf_reader.pages)
            
            print(f"DEBUG: PDF loaded. Total pages: {num_pages}")

            import gc
            import hashlib

            all_pagewise_items = []
            total_item_count = 0
            seen_page_hashes = set()

            # Process one page at a time, strictly isolating memory
            for i in range(num_pages):
                print(f"DEBUG: Processing Page {i+1}/{num_pages}...")
                
                page_data = None
                try:
                    # Re-open PDF just to extract this one page
                    with io.BytesIO(file_data) as f:
                        reader = PdfReader(f)
                        writer = PdfWriter()
                        writer.add_page(reader.pages[i])
                        
                        with io.BytesIO() as page_out:
                            writer.write(page_out)
                            page_data = page_out.getvalue()
                        
                        del reader
                        del writer
                        gc.collect()

                    # Calculate hash to detect duplicates
                    page_hash = hashlib.md5(page_data).hexdigest()
                    if page_hash in seen_page_hashes:
                        print(f"DEBUG: Page {i+1} is a duplicate. Skipping.")
                        continue
                    seen_page_hashes.add(page_hash)

                    # Call Gemini for this single page
                    print(f"DEBUG: Calling Gemini for Page {i+1}...")
                    page_result = call_gemini_api(page_data, 'application/pdf')
                    print(f"DEBUG: Gemini returned for Page {i+1}.")
                    
                    # Aggregate results
                    if page_result and "pagewise_line_items" in page_result:
                        for p_item in page_result["pagewise_line_items"]:
                            p_item["page_no"] = str(i + 1)
                            all_pagewise_items.append(p_item)
                    
                    if page_result and "total_item_count" in page_result:
                        total_item_count += page_result["total_item_count"]

                except Exception as e:
                    print(f"ERROR processing page {i+1}: {e}")
                    # Continue to next page instead of crashing
                    continue
                finally:
                    # Ensure memory is freed
                    if page_data:
                        del page_data
                    gc.collect()

            print("DEBUG: PDF processing complete. Returning aggregated results.")
            return {
                "pagewise_line_items": all_pagewise_items,
                "total_item_count": total_item_count
            }

        except Exception as e:
            print(f"ERROR: PDF Processing Failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to single-shot if PDF splitting fails (e.g. encrypted)
            pass

    # --- SINGLE SHOT LOGIC (Images or Fallback) ---
    return call_gemini_api(file_data, mime_type)


def call_gemini_api(file_data, mime_type):
    """Helper function to call Gemini for a single file/page."""
    model = genai.GenerativeModel('gemini-flash-latest')
    
    prompt = """
    You are an expert data extraction agent. Your task is to extract line item details from the provided bill/invoice (image or PDF).
    
    GOAL: Extract all purchasable items such that the sum of their 'item_amount' equals the Final Bill Total.
    
    STEPS:
    1. Identify the 'Final Total' amount on the bill.
    2. Extract all individual line items (products, services, charges).
    3. Ensure you do NOT extract 'Sub-total' lines or 'Total' lines as items, to avoid double counting.
    4. Ensure you DO extract all distinct charges (like 'Pharmacy Charge', 'Consultation Fee', etc.) if they are part of the final total and not just a sum of other listed items.
    5. Verify: Sum(item_amount) should be very close to Final Total.

    FIELDS TO EXTRACT per item:
    - item_name: Name or description (Exactly as mentioned in the bill, preserve newlines if present).
    - item_amount: Net Amount of the item post discounts as mentioned in the bill (float).
    - item_rate: Unit price/rate exactly as mentioned in the bill (float).
    - item_quantity: Quantity exactly as mentioned in the bill (float).

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

    import time
    
    from google.generativeai.types import HarmCategory, HarmBlockThreshold

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
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
                print(f"DEBUG: Response candidates: {response.candidates}")
                
                # Try to get text from candidates manually
                try:
                    text = response.candidates[0].content.parts[0].text
                except:
                    if attempt < max_retries - 1:
                        print("Retrying due to empty/blocked response...")
                        continue
                    print("ERROR: Could not extract text from Gemini response. Skipping this page.")
                    return {"pagewise_line_items": [], "total_item_count": 0}

            print(f"DEBUG: Raw Gemini Response (Attempt {attempt+1}):\n{text[:500]}...") # Log first 500 chars
            
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
                        time.sleep(1)
                        continue
                    raise ValueError(f"Failed to parse JSON after {max_retries} attempts: {e}")

        except Exception as e:
            if "429" in str(e) or "Quota exceeded" in str(e) or "ResourceExhausted" in str(e):
                if attempt < max_retries - 1:
                    print(f"Quota exceeded. Retrying in {2 ** (attempt + 1)} seconds...")
                    time.sleep(2 ** (attempt + 1))
                    continue
            print(f"Gemini API Error: {e}")
            if attempt < max_retries - 1:
                 continue
            raise ValueError(f"Gemini API failed: {e}")
