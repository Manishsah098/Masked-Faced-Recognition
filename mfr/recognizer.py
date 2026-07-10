import cv2
import numpy as np

class FaceRecognizer:
    """Wraps OpenCV's FaceRecognizerSF for face alignment and feature embedding extraction."""
    def __init__(self, model_path, cosine_threshold=0.363):
        self.model_path = model_path
        self.cosine_threshold = cosine_threshold
        # SFace model expects FaceRecognizerSF.create
        self.recognizer = cv2.FaceRecognizerSF.create(self.model_path, "")

    def align_crop(self, frame, face_raw):
        """Aligns and crops the face region based on facial landmarks to 112x112 pixels."""
        return self.recognizer.alignCrop(frame, face_raw)

    def extract_feature(self, aligned_face):
        """Extracts 128-dimensional feature embedding from a 112x112 aligned face."""
        # recognizer.feature returns a float32 numpy array of shape (1, 128)
        feature = self.recognizer.feature(aligned_face)
        return feature.flatten()

    def get_upper_face_image(self, aligned_face):
        """
        Applies a virtual mask to the aligned face by overwriting the mouth and nose region.
        In standard 112x112 face alignment:
        - Eyes are around y = 51
        - Nose tip is around y = 71
        - Mouth corners are around y = 92
        We black out from y = 65 downwards to cover the nose and mouth.
        """
        masked_face = aligned_face.copy()
        # Set the lower region (y >= 65) to neutral gray (127, 127, 127)
        masked_face[65:, :] = 127
        return masked_face

    def extract_upper_face_feature(self, aligned_face):
        """Extracts feature embedding from the upper face region only (lower face masked out)."""
        upper_face = self.get_upper_face_image(aligned_face)
        return self.extract_feature(upper_face)

    def compute_similarity(self, feat1, feat2):
        """Computes the Cosine Similarity between two 128D embeddings."""
        # Ensure they are numpy arrays
        f1 = np.array(feat1).flatten()
        f2 = np.array(feat2).flatten()
        
        norm1 = np.linalg.norm(f1)
        norm2 = np.linalg.norm(f2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return float(np.dot(f1, f2) / (norm1 * norm2))

    def match(self, feat1, feat2):
        """
        Compares two embeddings.
        Returns (is_match, similarity_score).
        """
        score = self.compute_similarity(feat1, feat2)
        is_match = score >= self.cosine_threshold
        return is_match, score
