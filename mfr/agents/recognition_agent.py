from ..recognizer import FaceRecognizer
from ..database import Database

class RecognitionAgent:
    """
    Biometric Tier — Agent #5: Recognition Agent
    Adaptive SFace matcher that dynamically fuses Upper-Face periocular embeddings
    and Full-Face biometrics based on Occlusion Agent directives to maintain high
    accuracy whether the subject is unmasked or wearing a mask/covering.
    """

    def __init__(self, model_path, db_path="db.json", cosine_threshold=0.363):
        self.recognizer = FaceRecognizer(model_path, cosine_threshold=cosine_threshold)
        self.db = Database(db_path)

    def set_threshold(self, threshold):
        self.recognizer.cosine_threshold = threshold

    def process(self, frame, face_payload, occlusion_payload):
        strategy = occlusion_payload.get('strategy', 'FULL_FACE')

        if strategy == "INSUFFICIENT":
            return {
                'candidate': "Unknown",
                'similarity_score': 0.0,
                'similarity_pct': 0.0,
                'is_match': False,
                'strategy_used': "INSUFFICIENT"
            }

        # Align crop to 112x112 using facial landmarks
        aligned_face = self.recognizer.align_crop(frame, face_payload['raw'])

        # Extract features for both upper face and full face for dual-template matching
        upper_feature = self.recognizer.extract_upper_face_feature(aligned_face)
        full_feature = self.recognizer.extract_feature(aligned_face)

        all_users = self.db.users
        if not all_users:
            return {
                'candidate': "Unknown",
                'similarity_score': 0.0,
                'similarity_pct': 0.0,
                'is_match': False,
                'strategy_used': strategy
            }

        best_user = "Unknown"
        best_score = 0.0
        best_pct = 0.0
        is_match = False

        # Upper face matching threshold (calibrated for periocular region)
        upper_thresh = max(0.310, self.recognizer.cosine_threshold - 0.040)
        full_thresh = self.recognizer.cosine_threshold

        for name, user_data in all_users.items():
            db_upper = user_data["upper"]
            db_full = user_data["full"]

            score_upper = self.recognizer.compute_similarity(upper_feature, db_upper)
            score_full = self.recognizer.compute_similarity(full_feature, db_full)

            if strategy == "UPPER_FACE":
                # Primary signal is upper face template match; full face acts as cross-check
                effective_score = max(score_upper, score_full * 0.95)
                effective_match = (score_upper >= upper_thresh) or (score_full >= full_thresh)
                
                # Calibrate percentage: 0.31 threshold maps to ~75% confidence, 0.60+ maps to 95%+
                if effective_score >= upper_thresh:
                    pct = 75.0 + ((effective_score - upper_thresh) / (1.0 - upper_thresh)) * 24.0
                else:
                    pct = max(0.0, (effective_score / upper_thresh) * 60.0)
            else: # FULL_FACE
                # Primary signal is full face template match; upper face acts as cross-check
                effective_score = max(score_full, score_upper * 0.95)
                effective_match = (score_full >= full_thresh) or (score_upper >= (upper_thresh + 0.02))

                if effective_score >= full_thresh:
                    pct = 78.0 + ((effective_score - full_thresh) / (1.0 - full_thresh)) * 21.0
                else:
                    pct = max(0.0, (effective_score / full_thresh) * 60.0)

            if effective_score > best_score:
                best_score = effective_score
                best_pct = pct
                if effective_match:
                    best_user = name
                    is_match = True

        return {
            'candidate': best_user if is_match else "Unknown",
            'similarity_score': float(round(best_score, 4)),
            'similarity_pct': float(round(min(99.5, best_pct), 1)),
            'is_match': is_match,
            'strategy_used': strategy,
            'feature': upper_feature if strategy == "UPPER_FACE" else full_feature
        }

