import google.generativeai as genai
import os
from dotenv import load_dotenv
import time

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def test_model(model_name):
    print(f"Testing {model_name}...")
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Hello, are you available?")
        print(f"SUCCESS: {model_name} responded: {response.text}")
        return True
    except Exception as e:
        print(f"FAILURE: {model_name} error: {e}")
        return False

print("--- Checking Model Availability ---")
test_model("gemini-1.5-flash")
time.sleep(1)
test_model("gemini-2.0-flash-exp")
