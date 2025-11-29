import requests
import os
import json
import time

API_URL = "http://127.0.0.1:8000/extract-from-file"
SAMPLES_DIR = "training_samples/TRAINING_SAMPLES"
RESULTS_DIR = "results"

def run_batch():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    files = sorted([f for f in os.listdir(SAMPLES_DIR) if f.endswith('.pdf')])
    print(f"Found {len(files)} samples.")

    for filename in files:
        filepath = os.path.join(SAMPLES_DIR, filename)
        print(f"Processing {filename}...", end=" ", flush=True)
        
        start_time = time.time()
        try:
            with open(filepath, 'rb') as f:
                files = {'file': (filename, f, 'application/pdf')}
                response = requests.post(API_URL, files=files)
            
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                if data.get("is_success"):
                    print(f"SUCCESS ({duration:.2f}s)")
                else:
                    print(f"FAILED (API Error) ({duration:.2f}s): {data.get('error')}")
            else:
                print(f"FAILED (HTTP {response.status_code}) ({duration:.2f}s)")
                data = {"error": f"HTTP {response.status_code}", "content": response.text}

            # Save result
            with open(os.path.join(RESULTS_DIR, f"{filename}.json"), 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            print(f"ERROR: {e}")

if __name__ == "__main__":
    run_batch()
