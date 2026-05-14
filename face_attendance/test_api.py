import urllib.request
import urllib.parse
import base64
import json

with open("dataset/janardhan/0.jpg", "rb") as f:
    img_data = f.read()

b64_img = base64.b64encode(img_data).decode('utf-8')

url = "http://127.0.0.1:5000/api/facedetect"
data = urllib.parse.urlencode({"face": b64_img}).encode('utf-8')
req = urllib.request.Request(url, data=data)

try:
    with urllib.request.urlopen(req) as response:
        result = response.read()
        print("Response:", result.decode('utf-8'))
except Exception as e:
    print("Error:", e)
