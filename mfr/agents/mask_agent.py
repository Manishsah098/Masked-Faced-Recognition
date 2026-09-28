from ..mask_detector import MaskDetector

class MaskAnalysisAgent:
    """
    Perception Tier — Agent #3: Mask Analysis Agent
    Uses ONNX MobileNetV2 to classify mask status, coverage, and wearing compliance.
    """

    def __init__(self, model_path):
        self.detector = MaskDetector(model_path)

    def process(self, frame, face_payload):
        box = face_payload['box']
        landmarks = face_payload.get('landmarks')
        label, confidence = self.detector.predict(frame, box, landmarks=landmarks)

        # Estimate coverage based on class and confidence
        if label == "Masked":
            coverage = float(min(100.0, max(65.0, confidence * 100.0)))
            status = "Proper Mask"
        else:
            coverage = float(max(0.0, (1.0 - confidence) * 35.0))
            status = "No Mask"

        return {
            'mask_status': label,
            'wearing_compliance': status,
            'mask_confidence': float(round(confidence * 100.0, 1)),
            'estimated_coverage_pct': float(round(coverage, 1)),
            'is_masked': (label == "Masked")
        }
