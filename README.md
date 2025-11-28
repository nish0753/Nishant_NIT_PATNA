# Bill Extraction API

This project provides an API to extract line item details from bill/invoice images using a Multimodal LLM (Google Gemini 1.5 Flash).

## Setup

1.  **Clone the repository** (or navigate to the directory).
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set up Environment Variables**:
    Create a `.env` file in the root directory and add your Google API Key:
    ```
    GOOGLE_API_KEY=your_api_key_here
    ```

## Usage

1.  **Start the server**:
    ```bash
    uvicorn app.main:app --reload --port 8001
    ```
2.  **API Endpoint**:
    -   **URL**: `http://127.0.0.1:8001/extract-bill-data`
    -   **Method**: `POST`
    -   **Body**:
        ```json
        {
          "document": "https://example.com/bill_image.png"
        }
        ```

## Approach

-   **FastAPI**: Used for creating the REST API.
-   **Google Gemini 1.5 Flash**: Used as the Multimodal LLM to process the image and extract structured JSON data.
-   **Pydantic**: Used for data validation and schema definition.
-   **Reconciliation**: The API calculates the total item count and reconciled amount (sum of line items) to ensure accuracy.

## Edge Case Handling

-   **Missing API Key**: The API fails gracefully with a clear error message.
-   **Invalid URLs**: The API catches download errors.
-   **Non-Image Content**: The API checks the `Content-Type` header and rejects non-image files (like PDFs) to prevent processing errors.
-   **LLM Hallucinations**: The prompt is engineered to be strict, and Pydantic validation ensures the output structure is correct.
