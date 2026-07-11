import cv2
import numpy as np

class FaceDetector:
    """Wraps OpenCV's FaceDetectorYN for fast, accurate DNN-based face detection."""
    def __init__(self, model_path, score_threshold=0.6, nms_threshold=0.35, top_k=5000):
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

    def validate_geometry(self, box, landmarks):
        """
        Validates the spatial geometry of the detected face features
        to filter out false positives (e.g., hands/fists looking like faces).
        Relaxed vertical checks are used to ensure compatibility with face masks.
        """
        x, y, w, h = box
        if w <= 0 or h <= 0:
            return False
            
        left_eye, right_eye, nose, left_mouth, right_mouth = landmarks
        
        # 1. Eye horizontal alignment check
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        eye_dist = np.sqrt(dx**2 + dy**2)
        if eye_dist == 0:
            return False
            
        # Angle of tilt in degrees
        angle = np.abs(np.degrees(np.arctan2(dy, dx)))
        if angle > 45 and angle < 135:  # Sideways orientations are usually false positives
            return False
            
        # 2. Eye distance ratio relative to bounding box width
        # Real faces have eyes spaced roughly 22% to 65% of the bounding box width.
        eye_width_ratio = eye_dist / w
        if eye_width_ratio < 0.22 or eye_width_ratio > 0.65:
            return False
            
        # 3. Relaxed vertical check (Eyes above Nose, Nose/Mouth below Eyes)
        # Masks occlude the nose and mouth, causing the landmark detector to estimate their
        # coordinates with high variance. We only enforce that nose and mouth are below eyes.
        avg_eye_y = (left_eye[1] + right_eye[1]) / 2.0
        avg_mouth_y = (left_mouth[1] + right_mouth[1]) / 2.0
        
        if nose[1] <= avg_eye_y - 0.1 * h:  # Nose must be below eyes (with 10% height tolerance)
            return False
        if avg_mouth_y <= avg_eye_y:  # Mouth must be below eyes
            return False
            
        # 4. Nose centering
        # Nose should be horizontally located between the two eyes (with a relaxed margin)
        min_eye_x = min(left_eye[0], right_eye[0])
        max_eye_x = max(left_eye[0], right_eye[0])
        margin = 0.35 * w
        if nose[0] < min_eye_x - margin or nose[0] > max_eye_x + margin:
            return False
            
        return True

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
                
                box = [x, y, width, height]
                
                # Scale landmarks back to original size
                landmarks = []
                for i in range(5):
                    lx = int(round(face[4 + 2 * i] * scale_x))
                    ly = int(round(face[5 + 2 * i] * scale_y))
                    landmarks.append([lx, ly])
                
                # Filter out false positives via geometric validation
                if not self.validate_geometry(box, landmarks):
                    continue
                
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
                    'box': box,
                    'landmarks': landmarks,
                    'confidence': confidence,
                    'raw': face_raw_scaled
                })
        
        return results
