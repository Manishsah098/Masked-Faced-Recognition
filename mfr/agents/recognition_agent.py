from ..recognizer import FaceRecognizer
from ..database import Database

class RecognitionAgent:
    """
    Biometric Tier — Agent #5: Recognition Agent
    Adaptive SFace matcher that switches between Full-Face and Upper-Face virtual masking
    embeddings based on Occlusion Agent directives.
    """

    def __init__(self, model_path, db_path="db.json", cosine_threshold=0.363):
        self.recognizer = FaceRecognizer(model_path, cosine_threshold=cosine_threshold)
        self.db = Database(db_path)

    def set_threshold(self, threshold):
        self.recognizer.cosine_threshold = threshold

    def process(self, frame, face_payload, occlusion_payload):
        strategy = occlusion_payload['strategy']

        if strategy == "INSUFFICIENT":
            return {
                'candidate': "Unknown",
                'similarity_score': 0.0,
                'is_match': False,
                'strategy_used': "INSUFFICIENT"
            }

        # Align crop
        aligned_face = self.recognizer.align_crop(frame, face_payload['raw'])

        # Extract features according to strategy
        if strategy == "UPPER_FACE":
            feature = self.recognizer.extract_upper_face_feature(aligned_face)
        else: # FULL_FACE
            feature = self.recognizer.extract_feature(aligned_face)

        # Retrieve registered templates from database
        all_users = self.db.users
        if not all_users:
            return {
                'candidate': "Unknown",
                'similarity_score': 0.0,
                'is_match': False,
                'strategy_used': strategy
            }

        best_user = "Unknown"
        best_score = 0.0

        for name, user_data in all_users.items():
            if strategy == "UPPER_FACE":
                db_feature = user_data["upper"]
            else:
                db_feature = user_data["full"]

            score = self.recognizer.compute_similarity(feature, db_feature)
            if score > best_score:
                best_score = score
                best_user = name

        is_match = best_score >= self.recognizer.cosine_threshold

        return {
            'candidate': best_user if is_match else "Unknown",
            'similarity_score': float(round(best_score, 4)),
            'similarity_pct': float(round(best_score * 100.0, 1)),
            'is_match': is_match,
            'strategy_used': strategy,
            'feature': feature
        }
