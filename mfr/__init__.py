import os
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["OPENCV_FFMPEG_LOG_LEVEL"] = "-8"

import cv2
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

from .utils import ensure_models, get_model_path
from .detector import FaceDetector
from .recognizer import FaceRecognizer
from .mask_detector import MaskDetector
from .database import Database
from .detection_log import DetectionLog
from .agents import BiometricOrchestrator

__all__ = [
    'ensure_models',
    'get_model_path',
    'FaceDetector',
    'FaceRecognizer',
    'MaskDetector',
    'Database',
    'DetectionLog',
    'BiometricOrchestrator'
]

