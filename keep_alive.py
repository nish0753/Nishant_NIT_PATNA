import requests
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

URL = "https://nishant-nit-patna.onrender.com/"
INTERVAL = 600  # 10 minutes (Render sleeps after 15 mins)

def ping_server():
    try:
        response = requests.get(URL)
        if response.status_code == 200:
            logging.info(f"Ping Successful! Server is awake. Status: {response.status_code}")
        else:
            logging.warning(f"Ping Failed. Status: {response.status_code}")
    except Exception as e:
        logging.error(f"Error pinging server: {e}")

if __name__ == "__main__":
    logging.info(f"Starting Keep-Alive script for: {URL}")
    logging.info(f"Pinging every {INTERVAL/60} minutes...")
    
    while True:
        ping_server()
        time.sleep(INTERVAL)
