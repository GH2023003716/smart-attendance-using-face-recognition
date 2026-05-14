import cv2
import numpy as np
import dlib
import sys

sys.stdout = open('cam_debug.txt', 'w', encoding='utf-8')
sys.stderr = sys.stdout

print("Opening camera...")
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Camera 0 failed, trying 1...")
    camera = cv2.VideoCapture(1)

# Warm up
for _ in range(15):
    camera.read()

grabbed, frame = camera.read()
camera.release()

print(f"\n=== RAW FRAME INFO ===")
print(f"grabbed     : {grabbed}")
print(f"frame       : {type(frame)}")
if frame is not None:
    print(f"shape       : {frame.shape}")
    print(f"dtype       : {frame.dtype}")
    print(f"C-contiguous: {frame.flags['C_CONTIGUOUS']}")
    print(f"min/max     : {frame.min()} / {frame.max()}")
    print(f"ndim        : {frame.ndim}")

    # Try every possible conversion and test with dlib
    detector = dlib.get_frontal_face_detector()
    variants = {}

    # 1. Raw frame
    variants["raw_bgr"] = frame

    # 2. BGR -> Gray
    if len(frame.shape) == 3:
        variants["gray"] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 3. BGR -> RGB
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        variants["rgb"] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 4. BGRA -> BGR -> RGB (if 4 channel)
    if len(frame.shape) == 3 and frame.shape[2] == 4:
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        variants["bgra_to_rgb"] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # 5. Force uint8 + contiguous
    variants["forced_rgb"] = np.ascontiguousarray(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if len(frame.shape) == 3 else frame
    ).astype(np.uint8)

    print(f"\n=== DLIB COMPATIBILITY TEST ===")
    for name, img in variants.items():
        contig = np.ascontiguousarray(img.astype(np.uint8))
        try:
            result = detector(contig, 0)
            print(f"  [{name}]  shape={contig.shape} dtype={contig.dtype} -> ✅ WORKS! (detected {len(result)} face(s))")
        except Exception as e:
            print(f"  [{name}]  shape={contig.shape} dtype={contig.dtype} -> ❌ {e}")
else:
    print("frame is None — camera returned nothing!")
