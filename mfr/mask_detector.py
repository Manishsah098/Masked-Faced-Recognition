import cv2
import numpy as np

class MaskDetector:
    """Classifies if a cropped face is wearing a mask or not using an ONNX model."""
    def __init__(self, model_path):
        self.model_path = model_path
        # Load the ONNX model using OpenCV DNN
        self.net = cv2.dnn.readNetFromONNX(self.model_path)
        
        # Set preferable backend to CPU for general compatibility
        try:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        except Exception:
            pass

    def predict(self, frame, box):
        """
        Predicts if the face inside the bounding box is masked or unmasked.
        box: [x, y, w, h]
        Returns (label, confidence) where:
          - label: "Masked" or "Unmasked"
          - confidence: float probability (0.0 to 1.0)
        """
        fh, fw = frame.shape[:2]
        x, y, w, h = box
        
        # Clip bounding box coordinates to image dimensions
        x = max(0, x)
        y = max(0, y)
        w = min(w, fw - x)
        h = min(h, fh - y)
        
        if w <= 0 or h <= 0:
            return "Unmasked", 0.0
            
        # Crop the face
        face_crop = frame[y:y+h, x:x+w]
        
        # Convert BGR to RGB (MobileNetV2 expects RGB)
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        
        # Resize to 224x224
        face_resized = cv2.resize(face_rgb, (224, 224))
        
        # Create input blob. Preprocessing: (pixel - 127.5) / 127.5 => scale = 1/127.5, mean = 127.5
        blob = cv2.dnn.blobFromImage(
            face_resized,
            scalefactor=1.0 / 127.5,
            size=(224, 224),
            mean=(127.5, 127.5, 127.5),
            swapRB=False,
            crop=False
        )
        
        # Run inference
        self.net.setInput(blob)
        preds = self.net.forward()
        
        # Output shape is (1, 2) where class 0 is 'Masked' and class 1 is 'Unmasked'
        probs = preds[0]
        
        # Apply softmax if values are raw logits (sometimes they are already probabilities, but softmax is safe)
        # MobileNetV2 from Keras Dense(2, activation="softmax") already outputs probabilities.
        # Just in case, let's get the max index and score
        masked_prob = float(probs[0])
        unmasked_prob = float(probs[1])
        
        # Sum probabilities to normalise if they are logits, or check if they sum to ~1
        total = masked_prob + unmasked_prob
        if total != 0:
            masked_prob /= total
            unmasked_prob /= total
            
        if masked_prob > unmasked_prob:
            return "Masked", masked_prob
        else:
            return "Unmasked", unmasked_prob
