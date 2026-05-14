
# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())

number = 0
frame_count = 0
detector = dlib.get_frontal_face_detector()
print("enter the person name")
name = input()
folder_name = "dataset/" + name

if os.path.exists(folder_name):
    print("Folder exist")
else:
    os.makedirs(folder_name)

# if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Camera index 0 failed, trying index 1...")
        camera = cv2.VideoCapture(1)
    if not camera.isOpened():
        print("ERROR: Could not open any camera. Please check your webcam connection.")
        exit(1)
    # Let camera warm up before first read
    print("Camera opened. Warming up...")
    for _ in range(10):
        camera.read()
    print("Ready! Look at the camera.")
else:
    camera = cv2.VideoCapture(args["video"])


def to_dlib_image(frame):
    """
    Convert a BGR OpenCV frame to a uint8 C-contiguous RGB image for dlib.
    Handles 3-channel (BGR) and 4-channel (BGRA) frames.
    Returns None if conversion fails.
    """
    if frame is None:
        return None
    # Handle 4-channel BGRA (some webcams on Windows return this)
    if len(frame.shape) == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    # Now convert BGR -> RGB (dlib needs RGB, not BGR)
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    elif len(frame.shape) == 2:
        # Already grayscale, convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    else:
        return None
    # Ensure uint8 and C-contiguous memory (required by dlib)
    rgb = np.ascontiguousarray(rgb.astype(np.uint8))
    return rgb


while True:

    if frame_count % 5 == 0:
        print("keyframe")

        (grabbed, image) = camera.read()

        # End of video file
        if args.get("video") and not grabbed:
            break

        # Skip invalid frames
        if not grabbed or image is None:
            print("Warning: Failed to grab frame, skipping...")
            frame_count += 1
            cv2.waitKey(1)
            continue

        # Resize for display and processing
        image = imutils.resize(image, width=500)
        image = np.ascontiguousarray(image)

        # Convert to dlib-compatible RGB image
        dlib_img = to_dlib_image(image)
        if dlib_img is None:
            print(f"Warning: Could not convert frame (shape={image.shape}, dtype={image.dtype}), skipping...")
            frame_count += 1
            cv2.waitKey(1)
            continue

        # Detect faces
        rects = detector(dlib_img, 1)

        # Also get grayscale for display rectangle (BGR)
        display_img = image.copy()

        # Loop over face detections
        for (i, rect) in enumerate(rects):
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cro = image[y: y + h, x: x + w]
            if cro.size == 0:
                continue
            out_image = cv2.resize(cro, (108, 108))
            fram = os.path.join(folder_name + "/", str(number) + "." + "jpg")
            number += 1
            cv2.imwrite(fram, out_image)
            cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show saved count on screen
        cv2.putText(display_img, f"Saved: {number}/52  Press Q to quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Face Capture - Look at camera!", display_img)
        frame_count += 1

    else:
        frame_count += 1
        print("redundant frame")
        (grabbed, image) = camera.read()
        if grabbed and image is not None:
            image = imutils.resize(image, width=500)
            image = np.ascontiguousarray(image)
            cv2.putText(image, f"Saved: {number}/52  Press Q to quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Face Capture - Look at camera!", image)

    if number > 51:
        print(f"\nDone! Captured 52 face images to '{folder_name}/'")
        print("Now run: python face_train.py")
        break

    # waitKey is REQUIRED for cv2.imshow to update the screen and to catch keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print(f"\nStopped. Captured {number} images so far.")
        break

# Clean up
camera.release()
cv2.destroyAllWindows()