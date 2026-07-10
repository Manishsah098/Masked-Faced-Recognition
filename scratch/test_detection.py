import cv2
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mfr

def test_camera_detection():
    print("Initializing detector...")
    model_path = mfr.get_model_path("yunet.onnx")
    detector = mfr.FaceDetector(model_path, score_threshold=0.5)
    
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera 0.")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Error: Could not open camera 1.")
            return
            
    print("Webcam opened. Capturing 20 frames...")
    
    for i in range(20):
        ret, frame = cap.read()
        if not ret:
            print(f"Frame {i}: Failed to read frame.")
            time.sleep(0.1)
            continue
            
        h, w = frame.shape[:2]
        print(f"Frame {i}: Shape: {w}x{h}")
        
        start = time.time()
        faces = detector.detect(frame)
        duration = time.time() - start
        
        print(f"  Detected faces: {len(faces)} (Time: {duration*1000:.1f}ms)")
        if len(faces) > 0:
            for idx, face in enumerate(faces):
                print(f"    Face {idx}: Box: {face['box']}, Conf: {face['confidence']:.3f}")
        time.sleep(0.1)
        
    cap.release()

if __name__ == "__main__":
    test_camera_detection()
