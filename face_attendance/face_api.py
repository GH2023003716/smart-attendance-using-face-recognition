from flask import Flask, jsonify, request
import os
import pickle
import face_recognition
import numpy as np
import time
from io import BytesIO
import base64
import csv
from datetime import datetime

# Path to save attendance
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ATTENDANCE_FILE = os.path.join(BASE_DIR, 'attendance.csv')
KNN_MODEL_PATH = os.path.join(BASE_DIR, "trained_knn_model.clf")

# Global variables for model and cache
knn_clf = None
last_marked = {}

def load_model():
    global knn_clf
    if os.path.exists(KNN_MODEL_PATH):
        with open(KNN_MODEL_PATH, 'rb') as f:
            knn_clf = pickle.load(f)
        print("Model loaded successfully.")
    else:
        print(f"Error: Model file not found at {KNN_MODEL_PATH}")

# Initialize the Flask application
app = Flask(__name__)

@app.route('/api/facedetect', methods=['POST'])
def upload_image():
    # check if the post request has the files part
    requestData = request.form
    if 'face' not in requestData:
        resp = jsonify({"message": "Error", "data": "No face found in request"})
        resp.status_code = 400
        return resp

    try:
        # Get the list of the uploaded files
        file = BytesIO(base64.decodebytes(requestData["face"].encode()))
        image = face_recognition.load_image_file(file)
        
        # Detection
        # Note: No downscaling for better accuracy in cropped mobile images
        X_face_locations = face_recognition.face_locations(image)

        # Fallback: if no faces found, try upsampling
        if not X_face_locations:
             X_face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=1)

        results = []
        if len(X_face_locations) != 0 and knn_clf is not None:
            # Find encodings
            faces_encodings = face_recognition.face_encodings(image, known_face_locations=X_face_locations)

            # Use the KNN model
            closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
            are_matches = [closest_distances[0][i][0] <= 0.4 for i in range(len(X_face_locations))]
            predictions = knn_clf.predict(faces_encodings)

            for name, is_match, dist in zip(predictions, are_matches, closest_distances[0]):
                res_name = name if is_match else "unknown"
                results.append({"name": res_name, "accuracy": dist[0]})
                print(f"Detected: {res_name} (Accuracy: {dist[0]:.4f})")
        
        if not results:
            print("No face detected in request.")
            # Return 200 instead of 400 to prevent client-side infinite retry loops
            resp = jsonify({"message": "success", "data": [], "info": "No face detected"})
            resp.status_code = 200
            return resp
        
        # --- ATTENDANCE LOGGING ---
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        for res in results:
            name = res.get("name")
            if name and name != "unknown":
                # Session-based marking: mark only once per session
                if name not in last_marked:
                    file_exists = os.path.isfile(ATTENDANCE_FILE)
                    with open(ATTENDANCE_FILE, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow(['Name', 'Date', 'Time'])
                        writer.writerow([name, date_str, time_str])
                    
                    print(f"Attendance marked for: {name} (Instant CSV Update)")
                    last_marked[name] = date_str
                else:
                    print(f"Skipped marking: {name} already marked in this session.")
            elif name == "unknown":
                print("Skipped marking: Person not recognized (unknown).")

        resp = jsonify({"message": "success", "data": results})
        resp.status_code = 200
        return resp

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        resp = jsonify({"message": "Error", "data": str(e)})
        resp.status_code = 500
        return resp

if __name__ == '__main__':
    load_model()
    print("\n" + "="*40)
    print("NEW ATTENDANCE SESSION STARTED")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Stop session: Press Ctrl+C")
    print("="*40 + "\n")
    app.run(port=5000, host="0.0.0.0", debug=True, threaded=False)
