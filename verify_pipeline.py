import sys
import os
import cv2

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import mfr

def test_pipeline():
    print("=== MFR Verification Script ===")
    
    print("\n1. Ensuring models are downloaded...")
    def download_progress(model_name, downloaded, total, status):
        if total > 0:
            pct = (downloaded / total) * 100
            print(f"   [{model_name}] {status}: {downloaded}/{total} bytes ({pct:.1f}%)", end='\r')
        else:
            print(f"   [{model_name}] {status}...", end='\r')
            
    try:
        mfr.ensure_models(download_progress)
        print("\n   All models ensured!")
    except Exception as e:
        print(f"\n   Error downloading models: {e}")
        return False

    print("\n2. Initializing Detector...")
    try:
        detector_path = mfr.get_model_path("yunet.onnx")
        detector = mfr.FaceDetector(detector_path)
        print(f"   Detector initialized successfully (model: {detector_path})")
    except Exception as e:
        print(f"   Failed to initialize detector: {e}")
        return False

    print("\n3. Initializing Recognizer...")
    try:
        recognizer_path = mfr.get_model_path("sface.onnx")
        recognizer = mfr.FaceRecognizer(recognizer_path)
        print(f"   Recognizer initialized successfully (model: {recognizer_path})")
    except Exception as e:
        print(f"   Failed to initialize recognizer: {e}")
        return False

    print("\n4. Initializing Mask Detector...")
    try:
        mask_detector_path = mfr.get_model_path("mask_detector.onnx")
        mask_detector = mfr.MaskDetector(mask_detector_path)
        print(f"   Mask Detector initialized successfully (model: {mask_detector_path})")
    except Exception as e:
        print(f"   Failed to initialize mask detector: {e}")
        return False

    print("\n5. Initializing Database...")
    try:
        db = mfr.Database("test_db.json")
        print(f"   Database initialized successfully (path: {db.db_path})")
        # Cleanup test DB if it exists
        if os.path.exists("test_db.json"):
            os.remove("test_db.json")
    except Exception as e:
        print(f"   Failed to initialize database: {e}")
        return False

    print("\n=== Verification Successful! ===")
    print("All components have loaded and initialized correctly. The pipeline is functional.")
    return True

if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)
