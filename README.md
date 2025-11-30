# Automated Bill Extraction System 🧾

> **A High-Precision, Hybrid AI Pipeline for extracting structured data from complex medical bills.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-green)
![Gemini](https://img.shields.io/badge/AI-Gemini%202.0%20Flash-orange)
![Docker](https://img.shields.io/badge/Deployment-Docker-blue)

## 🚀 Overview

This project solves the challenge of extracting line-item details from unstructured and complex PDF bills. Unlike traditional OCR or regex parsers that break on layout changes, this system uses a **Hybrid Multimodal Pipeline** combining:
1.  **Visual Understanding**: Google Gemini 2.0 Flash (Multimodal LLM).
2.  **Textual Precision**: Tesseract OCR (for raw text validation).

It is designed for **Robustness**, **Accuracy**, and **Scalability**.

## ✨ Key Features

*   **🧠 Hybrid "Golden Standard" Pipeline**: Fuses Visual Layout analysis (Gemini) with Raw Text extraction (Tesseract) to eliminate hallucinations.
*   **🛡️ Self-Healing JSON**: Integrated `json_repair` layer automatically fixes malformed API responses, ensuring 0% parse errors.
*   **⚡ Smart Batching**: Intelligently splits large multi-page PDFs into safe batches (Sequential Processing) to respect API rate limits and token windows.
*   **🔌 Magic Byte Detection**: Automatically detects PDF files even if the server returns incorrect MIME types (e.g., `application/octet-stream`).
*   **🐳 Dockerized**: Fully containerized with all system dependencies (`poppler`, `tesseract`) for seamless cloud deployment.

## 🛠️ Tech Stack

*   **Core Engine**: Google Gemini 2.0 Flash
*   **OCR Engine**: Tesseract OCR
*   **Backend Framework**: FastAPI (Async)
*   **Image Processing**: `pdf2image`, `Pillow`
*   **Deployment**: Docker, Render.com

## 🚀 Live Demo

**Base URL**: `https://nishant-nit-patna.onrender.com`

### API Endpoint
`POST /extract-bill-data`

**Curl Example:**
```bash
curl -X POST "https://nishant-nit-patna.onrender.com/extract-bill-data" \
     -H "Content-Type: application/json" \
     -d '{"document": "https://hackrx.blob.core.windows.net/assets/datathon-IIT/sample_1.png"}'
```

## 📦 Installation & Local Run

1.  **Clone the Repo:**
    ```bash
    git clone https://github.com/nish0753/Nishant_NIT_PATNA.git
    cd Nishant_NIT_PATNA
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You also need `tesseract-ocr` and `poppler-utils` installed on your system)*

3.  **Set Environment Variables:**
    Create a `.env` file:
    ```bash
    GOOGLE_API_KEY=your_gemini_api_key_here
    ```

4.  **Run the Server:**
    ```bash
    python -m uvicorn app.main:app --reload
    ```

## 📂 Project Structure

```
.
├── app/
│   ├── main.py          # FastAPI Entrypoint & Routes
│   ├── extractor.py     # Core Hybrid Pipeline Logic
│   └── schemas.py       # Pydantic Models
├── tests/               # Test Scripts
├── Dockerfile           # Production Docker Setup
├── keep_alive.py        # Utility to keep Render awake
├── requirements.txt     # Python Dependencies
└── README.md            # Documentation
```

## 👨‍💻 Author

**Nishant**
*   **College**: NIT Patna
