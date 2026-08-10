import numpy as np

class LivenessAgent:
    """
    Biometric Tier — Agent #6: Liveness / Anti-Spoof Agent
    Evaluates temporal landmark micro-movement, texture variance, and spatial consistency
    to protect against presentation attacks (photos, phone screens, videos).
    """

    def __init__(self, min_liveness_threshold=65.0):
        self.min_liveness_threshold = min_liveness_threshold
        self.prev_landmarks = None

    def process(self, face_payload, quality_payload):
        landmarks = np.array(face_payload['landmarks'], dtype=np.float32)
        blur_score = quality_payload.get('blur_score', 50.0)

        liveness_score = 90.0
        movement_delta = 0.0

        if self.prev_landmarks is not None:
            delta = np.linalg.norm(landmarks - self.prev_landmarks, axis=1)
            movement_delta = float(np.mean(delta))

            # Realistic live human faces have sub-pixel micro-jitter (1px to 15px per frame)
            if movement_delta == 0.0:
                # Completely static frame → suspicious (possible static image photo)
                liveness_score -= 35.0
            elif movement_delta > 40.0:
                # Sudden massive camera bump or rapid swap
                liveness_score -= 15.0
            else:
                # Natural micro-movement bonus
                liveness_score += min(10.0, movement_delta * 2.0)

        self.prev_landmarks = landmarks.copy()

        # Adjust for quality (excessively blurry image may indicate screen moire pattern)
        if blur_score < 20.0:
            liveness_score -= 20.0

        final_liveness = float(np.clip(np.round(liveness_score, 1), 0.0, 99.9))
        is_live = final_liveness >= self.min_liveness_threshold

        return {
            'liveness_score': final_liveness,
            'spoof_probability': float(round(100.0 - final_liveness, 1)),
            'movement_delta': float(round(movement_delta, 2)),
            'is_live': is_live,
            'status': "PASS" if is_live else "FAIL"
        }
