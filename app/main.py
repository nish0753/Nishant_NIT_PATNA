from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from .extractor import extract_bill_data
from .schemas import FinalResponse
import os
import logging
import requests
import asyncio

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

from starlette.requests import Request

@app.exception_handler(Exception)
async def universal_exception_handler(request: Request, exc: Exception):
    logger.error("Error while extracting: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "is_success": False,
            "token_usage": {
                "total_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0
            },
            "data": {
                "pagewise_line_items": [],
                "total_item_count": 0
            },
            "error": "Internal server error"
        }
    )

class BillRequest(BaseModel):
    document: str

def safe_float(value):
    """Converts anything to float safely."""
    try:
        if value is None:
            return 0.0
        return float(str(value).replace(",", "").strip())
    except:
        return 0.0

def build_response(result):
    """Fully safe wrapper for hackathon API."""
    
    if not isinstance(result, dict):
        result = {}

    pages = result.get("pagewise_line_items", [])
    if not isinstance(pages, list):
        pages = []

    page_entries = []

    for page in pages:
        if not isinstance(page, dict):
            continue

        items = page.get("bill_items", [])
        if not isinstance(items, list):
            items = []

        safe_items = []
        for item in items:
            if not isinstance(item, dict):
                continue

            safe_items.append({
                "item_name": item.get("item_name", "") or "",
                "item_amount": safe_float(item.get("item_amount")),
                "item_rate": safe_float(item.get("item_rate")),
                "item_quantity": safe_float(item.get("item_quantity")),
            })

        page_entries.append({
            "page_no": str(page.get("page_no", "1")),
            "page_type": page.get("page_type", "Bill Detail"),
            "bill_items": safe_items
        })

    total_items = sum(len(p["bill_items"]) for p in page_entries)

    return {
        "is_success": True,
        "token_usage": {
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0
        },
        "data": {
            "pagewise_line_items": page_entries,
            "total_item_count": total_items
        }
    }

def download_file_sync(url: str):
    """Synchronous download function to be run in executor."""
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    return response.content, response.headers.get('Content-Type', '')

@app.post("/extract-bill-data", response_model=FinalResponse)
async def extract_bill(request: BillRequest):
    try:
        # Step 1: Download the document
        logger.info("Downloding Start...")
        logger.info(f"Received request for document: {request.document}")
        
        # Run synchronous download in a separate thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        pdf_bytes, content_type = await loop.run_in_executor(None, download_file_sync, request.document)
        
        logger.info(f"Download complete. Size: {len(pdf_bytes)} bytes. Type: {content_type}")

        # Step 2: Process it
        logger.info("Processing Start...")
        # Pass bytes directly to extractor
        result = await extract_bill_data(pdf_bytes, mime_type=content_type)

        # Step 3: Wrap output into required format
        logger.info("Processing complete. Building response.")
        
        # Check for empty items
        if not result.get("pagewise_line_items") and result.get("total_item_count", 0) == 0:
             logger.warning("No items detected")

        response = build_response(result)
        item_count = response["data"]["total_item_count"]
        logger.info(f"SUCCESS: Extracted {item_count} items from document: {request.document}")
        return response

    except Exception as e:
        logger.error(f"FAILURE: Failed to extract from document: {request.document}. Error: {e}")
        logger.error("Error while extracting: %s", e)
        # Return failure response as per diagram requirements
        return {
            "is_success": False,
            "token_usage": {
                "total_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0
            },
            "data": {
                "pagewise_line_items": [],
                "total_item_count": 0
            }
        }


@app.post("/extract-from-file", response_model=FinalResponse)
async def extract_from_file(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file upload: {file.filename}")
        contents = await file.read()
        result = await extract_bill_data(contents, mime_type=file.content_type)
        response = build_response(result)
        item_count = response["data"]["total_item_count"]
        logger.info(f"SUCCESS: Extracted {item_count} items from file: {file.filename}")
        return response
    except Exception as e:
        logger.error(f"FAILURE: Failed to extract from file: {file.filename}. Error: {e}")
        logger.error(f"Error in extract_from_file: {e}")
        return {
            "is_success": False,
            "token_usage": {
                "total_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0
            },
            "data": {
                "pagewise_line_items": [],
                "total_item_count": 0
            }
        }


@app.get("/")
async def read_index():
    return FileResponse(os.path.join(STATIC_DIR, 'index.html'))
