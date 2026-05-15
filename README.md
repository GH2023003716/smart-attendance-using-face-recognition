# Smart Attendance Using Face Recognition

A smart attendance management system that uses **Face Recognition** to automatically identify users and mark attendance in real time.

This project combines:

* **Python + Flask backend** for face recognition and attendance processing
* **OpenCV** for image processing and face detection
* **Android frontend** for capturing and sending face data
* **CSV-based attendance storage** for maintaining attendance records


# Features

* Real-time face detection
* Face recognition using trained model
* Automatic attendance marking
* Android app integration
* Flask API communication
* Attendance logging using CSV
* Face dataset generation and training support


# Technologies Used

## Backend

* Python
* Flask
* OpenCV
* NumPy
* scikit-learn

## Frontend

* Android Studio
* Java
* Gradle


# Project Structure

smart-attendance-using-face-recognition/
│
├── FaceRecognition/        # Android frontend application
├── face_attendance/        # Python Flask backend
├── Screenshots/            # Project screenshots
└── README.md



# Backend Files

face_attendance/
│
├── dataset/                # Face dataset images
├── model/                  # Face detection models
├── attendance.csv          # Attendance records
├── face_generate.py        # Dataset generation
├── face_train.py           # Model training
├── face_api.py             # Flask API server
├── trained_knn_model.clf   # Trained recognition model
└── requirements.txt


# How It Works

1. Capture face images using the dataset generator.
2. Train the recognition model using captured images.
3. Start the Flask API server.
4. Android application sends camera frames to backend.
5. Backend detects and recognizes faces.
6. Attendance is automatically marked in `attendance.csv`.


# Setup Instructions

## Backend Setup

### 1. Install dependencies

pip install -r requirements.txt

### 2. Generate Dataset

python face_generate.py

### 3. Train the Model

python face_train.py

### 4. Run Flask API

python face_api.py



# Android Frontend Setup

1. Open `FaceRecognition` folder in Android Studio.
2. Sync Gradle files.
3. Connect Android device.
4. Run the application.

---

# Screenshots


## Model Training

<img width="1920" height="1080" alt="Screenshot (9)" src="https://github.com/user-attachments/assets/2ed8b42d-eb26-41ae-bcb7-d563215ce0a6" />
<img width="1920" height="1080" alt="Screenshot (8) - Copy" src="https://github.com/user-attachments/assets/a31b0fc8-50e1-47f1-9ffe-74180190474c" />



## Attendance CSV Output

<img width="1920" height="1080" alt="Screenshot (12)" src="https://github.com/user-attachments/assets/e59ca6f7-8b62-4a25-ad0b-6b8d093247e4" />



# Future Improvements

* Database integration
* Cloud deployment
* Better recognition accuracy
* Multiple user management
* Admin dashboard
* Live attendance analytics




# Author

**Janardhan CH**

GitHub: [https://github.com/GH2023003716](https://github.com/GH2023003716)
