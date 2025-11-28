from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from .extractor import extract_bill_data, ExtractionResponse
import os

app = FastAPI()

# Get the directory where main.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class BillRequest(BaseModel):
    document: str

@app.post("/extract-bill-data", response_model=ExtractionResponse)
async def extract_bill(request: BillRequest):
    response = extract_bill_data(request.document)
    if not response.is_success:
        return response
    return response

@app.post("/extract-from-file", response_model=ExtractionResponse)
async def extract_from_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        response = extract_bill_data(contents, mime_type=file.content_type)
        return response
    except Exception as e:
        return ExtractionResponse(is_success=False, error=str(e))

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(STATIC_DIR, 'index.html'))
