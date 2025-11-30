import requests
import json
import os

url = "http://127.0.0.1:8000/extract-from-file"
file_path = "test_multipage.pdf"

if not os.path.exists(file_path):
    print(f"Error: {file_path} not found.")
    exit(1)

try:
    with open(file_path, 'rb') as f:
        files = {'file': ('test_multipage.pdf', f, 'application/pdf')}
        print(f"Sending {file_path} to {url}...")
        response = requests.post(url, files=files)
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(json.dumps(data, indent=2))
        
        # Verify items
        total_items = data.get('data', {}).get('total_item_count', 0)
        print(f"\nTotal Items Extracted: {total_items}")
    else:
        print(f"Error Response: {response.text}")

except Exception as e:
    print(f"Error: {e}")
