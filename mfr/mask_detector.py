import cv2
import numpy as np

class MaskDetector:
    """
    Multi-Signal Face Mask & Lower-Face Occlusion Detector.
    Fuses deep learning ONNX inference with landmark-guided anatomical chromatic
    and texture analysis to reliably detect surgical masks, N95 masks, cloth masks,
    handkerchiefs, and lower-face coverings across diverse lighting conditions.
    """
    def __init__(self, model_path):
        self.model_path = model_path
        self.net = None
        try:
            self.net = cv2.dnn.readNetFromONNX(self.model_path)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        except Exception as e:
            print(f"Warning: Could not initialize ONNX mask model ({e}). Using anatomical analysis fallback.")

    def predict(self, frame, box, landmarks=None):
        """
        Predicts if the face inside the bounding box is masked or unmasked.
        box: [x, y, w, h]
        landmarks: Optional list of 5 facial landmarks [[x, y], ...]
        Returns (label, confidence) where:
          - label: "Masked" or "Unmasked"
          - confidence: float probability (0.0 to 1.0)
        """
        fh, fw = frame.shape[:2]
        x, y, w, h = box

        if w <= 0 or h <= 0 or fh <= 0 or fw <= 0:
            return "Unmasked", 0.0

        # 1. Padded contextual bounding box for full face context
        pad_x = int(0.08 * w)
        pad_y = int(0.08 * h)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(fw, x + w + pad_x)
        y2 = min(fh, y + h + pad_y)

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0 or face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            return "Unmasked", 0.0

        # 2. Deep Learning ONNX CNN Inference Signal
        cnn_masked_prob = 0.5
        if self.net is not None:
            try:
                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                face_resized = cv2.resize(face_rgb, (224, 224))
                blob = cv2.dnn.blobFromImage(
                    face_resized,
                    scalefactor=1.0 / 127.5,
                    size=(224, 224),
                    mean=(127.5, 127.5, 127.5),
                    swapRB=False,
                    crop=False
                )
                self.net.setInput(blob)
                preds = self.net.forward()
                probs = preds[0]
                m_p = float(probs[0])
                u_p = float(probs[1])
                tot = m_p + u_p
                if tot > 0:
                    cnn_masked_prob = m_p / tot
            except Exception:
                cnn_masked_prob = 0.5

        # 3. Anatomical Regional Chromatic & Structural Occlusion Analysis
        ch, cw = face_crop.shape[:2]

        if landmarks is not None and len(landmarks) >= 5:
            # Crop relative landmarks
            rel_left_eye = (landmarks[0][0] - x1, landmarks[0][1] - y1)
            rel_right_eye = (landmarks[1][0] - x1, landmarks[1][1] - y1)
            eye_center_y = int(round((rel_left_eye[1] + rel_right_eye[1]) / 2.0))
            eye_center_x = int(round((rel_left_eye[0] + rel_right_eye[0]) / 2.0))

            # Upper face baseline region: forehead and periocular area (above eyes)
            u_y1 = max(0, int(eye_center_y - 0.35 * ch))
            u_y2 = max(u_y1 + 5, int(eye_center_y - 0.05 * ch))
            u_x1 = max(0, int(eye_center_x - 0.28 * cw))
            u_x2 = min(cw, int(eye_center_x + 0.28 * cw))

            # Lower face target region: below nose down to chin
            l_y1 = min(ch - 5, max(0, int(eye_center_y + 0.15 * ch)))
            l_y2 = min(ch, int(ch * 0.98))
            l_x1 = max(0, int(eye_center_x - 0.35 * cw))
            l_x2 = min(cw, int(eye_center_x + 0.35 * cw))
        else:
            # Standard facial proportions
            u_y1, u_y2 = int(0.12 * ch), int(0.40 * ch)
            u_x1, u_x2 = int(0.20 * cw), int(0.80 * cw)
            l_y1, l_y2 = int(0.55 * ch), int(0.95 * ch)
            l_x1, l_x2 = int(0.15 * cw), int(0.85 * cw)

        upper_zone = face_crop[u_y1:u_y2, u_x1:u_x2]
        lower_zone = face_crop[l_y1:l_y2, l_x1:l_x2]

        occlusion_score = 0.0
        if upper_zone.size > 0 and lower_zone.size > 0:
            # Color space analysis in YCrCb (separates luminance from skin chrominance)
            ycrcb_u = cv2.cvtColor(upper_zone, cv2.COLOR_BGR2YCrCb)
            ycrcb_l = cv2.cvtColor(lower_zone, cv2.COLOR_BGR2YCrCb)

            # Chrominance distance (Cr = red-diffuse, Cb = blue-diffuse)
            cr_u, cb_u = float(np.mean(ycrcb_u[:, :, 1])), float(np.mean(ycrcb_u[:, :, 2]))
            cr_l, cb_l = float(np.mean(ycrcb_l[:, :, 1])), float(np.mean(ycrcb_l[:, :, 2]))
            chroma_dist = float(np.sqrt((cr_u - cr_l)**2 + (cb_u - cb_l)**2))

            # HSV color space analysis (Hue & Saturation divergence)
            hsv_u = cv2.cvtColor(upper_zone, cv2.COLOR_BGR2HSV)
            hsv_l = cv2.cvtColor(lower_zone, cv2.COLOR_BGR2HSV)
            sat_u, sat_l = float(np.mean(hsv_u[:, :, 1])), float(np.mean(hsv_l[:, :, 1]))
            sat_diff = abs(sat_u - sat_l)

            # Skin Chrominance deviation score: Natural skin across face has chroma_dist < 8.0
            # Masks / fabric / coverings typically exhibit chroma_dist > 14.0
            chroma_occlusion = np.clip((chroma_dist - 7.0) / 14.0, 0.0, 1.0)
            sat_occlusion = np.clip((sat_diff - 12.0) / 25.0, 0.0, 1.0)

            # Lower face variance / texture analysis (cloth masks have lower skin texture variance or high synthetic fabric patterns)
            std_u = float(np.std(ycrcb_u[:, :, 0]))
            std_l = float(np.std(ycrcb_l[:, :, 0]))
            std_diff = np.clip(abs(std_u - std_l) / 30.0, 0.0, 1.0)

            occlusion_score = float(np.clip(
                0.60 * chroma_occlusion + 0.25 * sat_occlusion + 0.15 * std_diff,
                0.0,
                1.0
            ))

        # 4. Multi-Signal Ensemble Fusion
        # Fuses deep learning CNN with spatial anatomical chromatic occlusion
        fused_masked_score = 0.40 * cnn_masked_prob + 0.60 * occlusion_score

        # High confidence triggers for clear mask cues
        if cnn_masked_prob > 0.65 or occlusion_score > 0.45:
            fused_masked_score = max(fused_masked_score, max(cnn_masked_prob, occlusion_score))
        elif cnn_masked_prob < 0.20 and occlusion_score < 0.20:
            fused_masked_score = min(fused_masked_score, min(cnn_masked_prob, occlusion_score))

        is_masked = (fused_masked_score >= 0.38)

        if is_masked:
            # Calibrate confidence percentage for masked status
            confidence = float(np.clip(max(fused_masked_score, 0.65 + 0.34 * fused_masked_score), 0.50, 0.99))
            return "Masked", confidence
        else:
            confidence = float(np.clip(1.0 - fused_masked_score, 0.50, 0.99))
            return "Unmasked", confidence

