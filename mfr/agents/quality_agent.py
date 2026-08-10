import cv2
import numpy as np

class FaceQualityAgent:
    """
    Perception Tier — Agent #2: Face Quality Agent
    Evaluates image sharpness (Laplacian variance), contrast, face resolution,
    and tilt angle to produce a normalized 0-100% Quality Score.
    """

    def __init__(self, min_quality_threshold=40.0):
        self.min_quality_threshold = min_quality_threshold

    def process(self, frame, face_payload):
        box = face_payload['box']
        tilt = face_payload.get('tilt_angle', 0.0)
        fh, fw = frame.shape[:2]
        x, y, w, h = box

        x, y = max(0, x), max(0, y)
        w = min(w, fw - x)
        h = min(h, fh - y)

        if w <= 10 or h <= 10:
            return {
                'quality_score': 0.0,
                'blur_score': 0.0,
                'resolution_score': 0.0,
                'pose_score': 0.0,
                'is_acceptable': False
            }

        crop = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # 1. Blur score using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(100.0, (laplacian_var / 250.0) * 100.0)

        # 2. Face Resolution score (ideal face width >= 80px)
        resolution_score = min(100.0, (w / 120.0) * 100.0)

        # 3. Pose / Tilt score (penalty for tilt angle > 15 deg)
        pose_score = max(0.0, 100.0 - (tilt * 2.5))

        # 4. Contrast & Brightness check
        contrast_std = gray.std()
        contrast_score = min(100.0, (contrast_std / 50.0) * 100.0)

        # Weighted final Quality Score
        quality_score = float(np.round(
            0.40 * blur_score +
            0.30 * resolution_score +
            0.15 * pose_score +
            0.15 * contrast_score,
            1
        ))

        is_acceptable = quality_score >= self.min_quality_threshold

        return {
            'quality_score': quality_score,
            'blur_score': float(np.round(blur_score, 1)),
            'resolution_score': float(np.round(resolution_score, 1)),
            'pose_score': float(np.round(pose_score, 1)),
            'is_acceptable': is_acceptable
        }
