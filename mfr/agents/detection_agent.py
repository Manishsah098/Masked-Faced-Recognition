import numpy as np
from ..detector import FaceDetector

class FaceDetectionAgent:
    """
    Perception Tier — Agent #1: Face Detection Agent
    Wraps YuNet DNN face detector to locate faces, extract 5 key landmarks,
    estimate bounding box geometry, and compute head orientation.
    """

    def __init__(self, model_path, score_threshold=0.6):
        self.detector = FaceDetector(model_path, score_threshold=score_threshold)

    def process(self, frame):
        """
        Executes face detection on the input frame.
        Returns a list of structured face perception payloads.
        """
        raw_faces = self.detector.detect(frame)
        payloads = []

        for face in raw_faces:
            box = face['box']
            landmarks = face['landmarks']
            confidence = face['confidence']

            # Estimate face orientation / tilt angle
            left_eye, right_eye = landmarks[0], landmarks[1]
            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            tilt_angle = float(np.abs(np.degrees(np.arctan2(dy, dx)))) if dx != 0 else 0.0

            payloads.append({
                'box': box,
                'landmarks': landmarks,
                'detection_confidence': confidence,
                'tilt_angle': tilt_angle,
                'raw': face['raw']
            })

        return payloads
