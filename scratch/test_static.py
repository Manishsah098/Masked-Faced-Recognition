import cv2
import sys
import os
import urllib.request

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mfr

def test_static_detection():
    print("=== Static Face Detection Test ===")
    
    # 1. Ensure models
    mfr.ensure_models()
    
    # 2. Download a sample face image (a clear unmasked face)
    sample_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
    image_path = "lena_test.jpg"
    
    if not os.path.exists(image_path):
        print(f"Downloading test image from {sample_url}...")
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(sample_url, image_path)
        print("Image downloaded.")
        
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load test image.")
        return
        
    h, w = img.shape[:2]
    print(f"Image loaded: {w}x{h} pixels.")
    
    # 3. Test Detector
    det_path = mfr.get_model_path("yunet.onnx")
    print(f"Loading detector from {det_path}...")
    detector = mfr.FaceDetector(det_path, score_threshold=0.5)
    
    faces = detector.detect(img)
    print(f"Detected faces: {len(faces)}")
    
    if len(faces) > 0:
        for idx, face in enumerate(faces):
            print(f"  Face {idx}: Box: {face['box']}, Confidence: {face['confidence']:.4f}")
            
            # Test Mask Detector
            mask_path = mfr.get_model_path("mask_detector.onnx")
            print(f"  Loading mask detector from {mask_path}...")
            mask_detector = mfr.MaskDetector(mask_path)
            
            label, conf = mask_detector.predict(img, face['box'])
            print(f"  Classified as: {label} (Confidence: {conf:.4f})")
            
            # Test Recognizer
            rec_path = mfr.get_model_path("sface.onnx")
            print(f"  Loading SFace recognizer from {rec_path}...")
            recognizer = mfr.FaceRecognizer(rec_path)
            
            aligned = recognizer.align_crop(img, face['raw'])
            print(f"  Aligned crop shape: {aligned.shape}")
            
            feat = recognizer.extract_feature(aligned)
            print(f"  Extracted feature shape: {feat.shape}, norm: {float(np.linalg.norm(feat)):.4f}" if 'np' in globals() or 'numpy' in sys.modules else "  Extracted feature")
    else:
        print("Test failed: No faces detected on standard Lena image!")

if __name__ == "__main__":
    import numpy as np
    test_static_detection()
