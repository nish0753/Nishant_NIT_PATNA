import os
import requests
import json
import google.generativeai as genai
from typing import List, Optional, Union, Literal
from pydantic import BaseModel
from dotenv import load_dotenv
import re
from PIL import Image, ImageEnhance
import io

load_dotenv()

# Configure Gemini
GENAI_API_KEY = os.getenv("GOOGLE_API_KEY")

if GENAI_API_KEY:
    genai.configure(api_key=GENAI_API_KEY)

class TokenUsage(BaseModel):
    total_tokens: int
    input_tokens: int
    output_tokens: int

class BillItem(BaseModel):
    item_name: str
    item_amount: float
    item_rate: Optional[float] = 0.0
    item_quantity: Optional[float] = 1.0

class PageWiseItems(BaseModel):
    page_no: str
    page_type: Literal["Bill Detail", "Final Bill", "Pharmacy"]
    bill_items: List[BillItem]

class ExtractionData(BaseModel):
    pagewise_line_items: List[PageWiseItems]
    total_item_count: int

class ExtractionResponse(BaseModel):
    is_success: bool
    token_usage: Optional[TokenUsage] = None
    data: Optional[ExtractionData] = None
    error: Optional[str] = None

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

def extract_bill_data(image_source: Union[str, bytes], mime_type: str = None) -> ExtractionResponse:
    if not GENAI_API_KEY:
        return ExtractionResponse(is_success=False, error="GOOGLE_API_KEY not found in environment variables.")

    try:
        if isinstance(image_source, str):
            # It's a URL
            file_data, mime_type = download_image(image_source)
        else:
            # It's raw bytes
            file_data = image_source
            if not mime_type:
                return ExtractionResponse(is_success=False, error="Mime type must be provided for file uploads.")

        # Apply preprocessing if it's an image
        if mime_type and mime_type.startswith('image/'):
            print("Applying image preprocessing...")
            file_data = preprocess_image(file_data)
            # Mime type remains image/jpeg effectively after processing
            mime_type = 'image/jpeg' 

        model = genai.GenerativeModel('gemini-2.0-flash')
        
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
        - reconciled_amount: Sum of all 'item_amount' values.

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
          "total_item_count": 0,
          "reconciled_amount": 0.0
        }
        """

        response = model.generate_content([
            {'mime_type': mime_type, 'data': file_data},
            prompt
        ])
        
        # Extract token usage
        usage = response.usage_metadata
        token_usage = TokenUsage(
            total_tokens=usage.total_token_count,
            input_tokens=usage.prompt_token_count,
            output_tokens=usage.candidates_token_count
        )

        # Clean up response text to ensure it's valid JSON
        text = response.text.strip()
        print(f"DEBUG: Raw Gemini Response:\n{text}")
        
        text = clean_json(text)
            
        try:
            data_dict = json.loads(text)
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {e}. Attempting fallback...")
            # Try to insert commas between fields if that's the issue
            text_fixed = re.sub(r'"\s*\n\s*"', '",\n"', text)
            try:
                data_dict = json.loads(text_fixed)
            except:
                return ExtractionResponse(is_success=False, error=f"Failed to parse JSON: {e}")
        
        # Recalculate total_item_count and reconciled_amount to be sure
        total_count = 0
        total_amount = 0.0
        for page in data_dict.get("pagewise_line_items", []):
            for item in page.get("bill_items", []):
                total_count += 1
                total_amount += item.get("item_amount", 0.0)
        
        data_dict['total_item_count'] = total_count

        return ExtractionResponse(
            is_success=True,
            token_usage=token_usage,
            data=ExtractionData(**data_dict)
        )

    except Exception as e:
        return ExtractionResponse(is_success=False, error=str(e))
