
import base64
import requests
import os

# Get path to a sample image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(BASE_DIR, "dataset", "janardhan", "0.jpg")

if not os.path.exists(IMAGE_PATH):
    print(f"Error: {IMAGE_PATH} not found.")
    exit(1)

with open(IMAGE_PATH, "rb") as f:
    img_data = f.read()

# Base64 encode
b64_img = base64.b64encode(img_data).decode('utf-8')

# Prepare the data mapping exactly how it's sent from Android/requests
url = "http://127.0.0.1:5000/api/facedetect"
data = {"face": b64_img}

print(f"Sending request to {url}...")
try:
    # Use requests to send a proper POST form
    response = requests.post(url, data=data)
    print(f"Status Code: {response.status_code}")
    print("Response JSON:", response.text)
except Exception as e:
    print("Error:", e)
