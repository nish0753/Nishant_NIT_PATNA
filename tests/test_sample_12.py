import requests
import json
import os

# Ensure the file exists
file_path = "training_samples/TRAINING_SAMPLES/train_sample_12.pdf"
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit(1)

url = "http://127.0.0.1:8000/extract-from-file"
files = {'file': ('train_sample_12.pdf', open(file_path, 'rb'), 'application/pdf')}

try:
    print(f"Sending {file_path}...")
    response = requests.post(url, files=files)
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")
