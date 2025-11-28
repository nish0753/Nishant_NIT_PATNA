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

class BillRequest(BaseModel):
    document: str

def build_response(result):
    """Force-wrap extractor output into official response schema."""
    
    page_entries = []
    for page in result["pagewise_line_items"]:
        page_entries.append({
            "page_no": str(page.get("page_no", "1")),
            "page_type": page.get("page_type", "Bill Detail"),  # safe default
            "bill_items": [
                {
                    "item_name": item["item_name"],
                    "item_amount": float(item["item_amount"]),
                    "item_rate": float(item["item_rate"]),
                    "item_quantity": float(item["item_quantity"]),
                }
                for item in page["bill_items"]
            ]
        })

    total_amount = 0.0
    for page in page_entries:
        for item in page["bill_items"]:
            total_amount += item["item_amount"]

    return {
        "is_success": True,
        "token_usage": {
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0
        },
        "data": {
            "pagewise_line_items": page_entries,
            "total_item_count": result.get("total_item_count", len(page_entries)),
            "total_amount": round(total_amount, 2)
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
