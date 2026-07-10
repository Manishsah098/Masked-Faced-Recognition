import os
import urllib.request
import sys

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

MODEL_URLS = {
    "yunet.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "sface.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    "mask_detector.onnx": "https://github.com/ZJW-92/ML_face_mask_detection/raw/main/mask_detector.onnx"
}

def get_model_path(model_name):
    """Returns the absolute path to a model file."""
    return os.path.join(MODELS_DIR, model_name)

def download_file(url, dest_path, progress_callback=None):
    """Downloads a file from a URL to a destination path, with progress updates."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    # Configure custom opener to handle user-agent requirements
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
    urllib.request.install_opener(opener)
    
    try:
        with urllib.request.urlopen(url) as response:
            total_size = int(response.info().get('Content-Length', 0))
            block_size = 1024 * 32  # 32 KB blocks
            downloaded = 0
            
            with open(dest_path, 'wb') as out_file:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    out_file.write(buffer)
                    if progress_callback:
                        progress_callback(downloaded, total_size)
    except Exception as e:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise e

def ensure_models(progress_callback=None):
    """Ensures all required models are present in the models directory."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    for model_name, url in MODEL_URLS.items():
        dest = get_model_path(model_name)
        # Check if file exists and is of non-trivial size
        if not os.path.exists(dest) or os.path.getsize(dest) < 1000:
            if progress_callback:
                progress_callback(model_name, 0, 100, "Downloading")
            
            def cb(downloaded, total):
                if progress_callback:
                    progress_callback(model_name, downloaded, total, "Downloading")
            
            try:
                download_file(url, dest, cb)
            except Exception as e:
                print(f"Error downloading {model_name}: {e}", file=sys.stderr)
                raise e
            
            if progress_callback:
                progress_callback(model_name, 100, 100, "Done")
        else:
            if progress_callback:
                progress_callback(model_name, 100, 100, "Already exists")
