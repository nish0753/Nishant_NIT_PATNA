from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from .extractor import extract_bill_data
from .schemas import FinalResponse
import os

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

from fastapi.responses import JSONResponse
from starlette.requests import Request

@app.exception_handler(Exception)
async def universal_exception_handler(request: Request, exc: Exception):
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


@app.post("/extract-bill-data", response_model=FinalResponse)
async def extract_bill(request: BillRequest):
    result = extract_bill_data(request.document)
    return build_response(result)


@app.post("/extract-from-file", response_model=FinalResponse)
async def extract_from_file(file: UploadFile = File(...)):
    contents = await file.read()
    result = extract_bill_data(contents, mime_type=file.content_type)
    return build_response(result)


@app.get("/")
async def read_index():
    return FileResponse(os.path.join(STATIC_DIR, 'index.html'))
