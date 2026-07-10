import cv2
import numpy as np

class FaceDetector:
    """Wraps OpenCV's FaceDetectorYN for fast, accurate DNN-based face detection."""
    def __init__(self, model_path, score_threshold=0.3, nms_threshold=0.35, top_k=5000):
        self.model_path = model_path
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.detector = None
        self.target_width = 320  # Standardized width for optimal face scale and CPU speed
        self.input_size = (320, 240)  # Default size

    def _init_detector(self, size):
        """Initializes or re-initializes the detector with the correct input size."""
        self.input_size = size
        self.detector = cv2.FaceDetectorYN.create(
            self.model_path,
            "",
            self.input_size,
            self.score_threshold,
            self.nms_threshold,
            self.top_k
        )

    def detect(self, frame):
        """
        Detects faces in the given BGR frame.
        Resizes the frame internally to target_width for optimal speed and scale,
        then scales the detected bounding boxes and landmarks back to the original size.
        """
        h, w = frame.shape[:2]
        if w <= 0 or h <= 0:
            return []
            
        # Calculate resized dimensions
        scale_x = w / float(self.target_width)
        target_height = int(round(h * (float(self.target_width) / w)))
        scale_y = h / float(target_height)
        
        resized_frame = cv2.resize(frame, (self.target_width, target_height))
        
        # Initialize detector if it's the first run or if frame size has changed
        if self.detector is None or self.input_size != (self.target_width, target_height):
            self._init_detector((self.target_width, target_height))

        retval, faces = self.detector.detect(resized_frame)
        
        results = []
        if faces is not None:
            for face in faces:
                # Scale bounding box back to original size
                x = int(round(face[0] * scale_x))
                y = int(round(face[1] * scale_y))
                width = int(round(face[2] * scale_x))
                height = int(round(face[3] * scale_y))
                
                # Scale landmarks back to original size
                landmarks = []
                for i in range(5):
                    lx = int(round(face[4 + 2 * i] * scale_x))
                    ly = int(round(face[5 + 2 * i] * scale_y))
                    landmarks.append([lx, ly])
                
                # Scale raw face row back to original size for SFace alignCrop compatibility
                face_raw_scaled = face.copy()
                face_raw_scaled[0] = face[0] * scale_x
                face_raw_scaled[1] = face[1] * scale_y
                face_raw_scaled[2] = face[2] * scale_x
                face_raw_scaled[3] = face[3] * scale_y
                for i in range(5):
                    face_raw_scaled[4 + 2 * i] = face[4 + 2 * i] * scale_x
                    face_raw_scaled[5 + 2 * i] = face[5 + 2 * i] * scale_y
                
                # Confidence score
                confidence = float(face[14])
                
                results.append({
                    'box': [x, y, width, height],
                    'landmarks': landmarks,
                    'confidence': confidence,
                    'raw': face_raw_scaled
                })
        
        return results
