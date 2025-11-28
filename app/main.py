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

@app.post("/extract-bill-data", response_model=FinalResponse)
async def extract_bill(request: BillRequest):
    result = extract_bill_data(request.document)

    return {
        "is_success": True,
        "token_usage": {
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0
        },
        "data": {
            "pagewise_line_items": result["pagewise_line_items"],
            "total_item_count": result["total_item_count"]
        }
    }

@app.post("/extract-from-file", response_model=FinalResponse)
async def extract_from_file(file: UploadFile = File(...)):
    contents = await file.read()
    result = extract_bill_data(contents, mime_type=file.content_type)

    return {
        "is_success": True,
        "token_usage": {
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0
        },
        "data": {
            "pagewise_line_items": result["pagewise_line_items"],
            "total_item_count": result["total_item_count"]
        }
    }

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(STATIC_DIR, 'index.html'))
