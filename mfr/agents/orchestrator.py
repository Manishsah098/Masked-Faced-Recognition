import cv2
import os
import numpy as np

from .detection_agent import FaceDetectionAgent
from .quality_agent import FaceQualityAgent
from .mask_agent import MaskAnalysisAgent
from .occlusion_agent import OcclusionAgent
from .recognition_agent import RecognitionAgent
from .liveness_agent import LivenessAgent
from .tracking_agent import TemporalTrackingAgent
from .fusion_agent import FusionAgent
from .security_agent import SecurityAgent
from .audit_agent import AuditAgent
from ..utils import ensure_models, get_model_path

class BiometricOrchestrator:
    """
    Master Driver — Orchestrator Agent
    Controls dynamic execution flow, manages agent state dependencies, and returns
    unified multi-agent telemetry payloads for live video feeds.
    """

    def __init__(self, models_dir="models", db_path="db.json"):
        ensure_models()

        yunet_path = get_model_path("yunet.onnx")
        sface_path = get_model_path("sface.onnx")
        mask_path  = get_model_path("mask_detector.onnx")

        # Instantiate specialized agents
        self.detection_agent = FaceDetectionAgent(yunet_path)
        self.quality_agent = FaceQualityAgent()
        self.mask_agent = MaskAnalysisAgent(mask_path)
        self.occlusion_agent = OcclusionAgent()
        self.recognition_agent = RecognitionAgent(sface_path, db_path=db_path)
        self.liveness_agent = LivenessAgent()
        self.tracking_agent = TemporalTrackingAgent()
        self.fusion_agent = FusionAgent()
        self.security_agent = SecurityAgent()
        self.audit_agent = AuditAgent()

        self.frame_count = 0
        self.frame_interval = 1
        self.last_result = None

    def set_strict_mode(self, enabled):
        self.security_agent.strict_mode = enabled

    def set_cosine_threshold(self, threshold):
        self.recognition_agent.set_threshold(threshold)

    def process_frame(self, frame):
        self.frame_count += 1

        # Frame skipping optimization
        if self.last_result is not None and (self.frame_count % self.frame_interval != 0):
            return self.last_result

        # Step 1: Face Detection Agent
        faces = self.detection_agent.process(frame)

        if not faces:
            self.tracking_agent.reset()
            empty_state = {
                'detected': False,
                'candidate': "No Face Detected",
                'status': "NO_FACE",
                'color': "RED",
                'agents': {
                    'detection': {'faces': 0},
                    'quality': {'quality_score': 0.0},
                    'mask': {'mask_status': "N/A"},
                    'occlusion': {'overall_visible_pct': 0.0},
                    'liveness': {'liveness_score': 0.0},
                    'fusion': {'calibrated_confidence': 0.0},
                    'security': {'decision': "NO_FACE", 'risk_level': "NONE"}
                },
                'explanation': "Awaiting face in camera field of view."
            }
            self.last_result = (frame.copy(), empty_state)
            return frame.copy(), empty_state

        # Primary face payload
        face_payload = faces[0]

        # Step 2: Quality Agent
        quality_payload = self.quality_agent.process(frame, face_payload)

        # Step 3: Mask Agent
        mask_payload = self.mask_agent.process(frame, face_payload)

        # Step 4: Occlusion Agent
        occlusion_payload = self.occlusion_agent.process(face_payload, mask_payload)

        # Step 5: Recognition Agent
        recognition_payload = self.recognition_agent.process(frame, face_payload, occlusion_payload)

        # Step 6: Liveness Agent
        liveness_payload = self.liveness_agent.process(face_payload, quality_payload)

        # Step 7: Temporal Tracking Agent
        tracking_payload = self.tracking_agent.process(recognition_payload)

        # Step 8: Fusion Agent
        fusion_payload = self.fusion_agent.process(
            quality_payload, mask_payload, occlusion_payload,
            recognition_payload, liveness_payload, tracking_payload
        )

        # Step 9: Security Agent
        security_payload = self.security_agent.process(
            fusion_payload, mask_payload, liveness_payload, occlusion_payload, quality_payload
        )

        # Step 10: Audit & Explanation Agent
        audit_payload = self.audit_agent.process(
            security_payload, fusion_payload, occlusion_payload,
            quality_payload, liveness_payload, mask_payload
        )

        # Annotate video frame with HUD
        annotated_frame = self.draw_hud(frame.copy(), face_payload, mask_payload, security_payload, fusion_payload)

        consolidated_state = {
            'detected': True,
            'candidate': security_payload['candidate'],
            'status': security_payload['decision'],
            'color': security_payload['color'],
            'confidence': fusion_payload['calibrated_confidence'],
            'explanation': audit_payload['explanation'],
            'agents': {
                'detection': {'faces': len(faces), 'confidence': face_payload['detection_confidence']},
                'quality': quality_payload,
                'mask': mask_payload,
                'occlusion': occlusion_payload,
                'recognition': recognition_payload,
                'liveness': liveness_payload,
                'tracking': tracking_payload,
                'fusion': fusion_payload,
                'security': security_payload,
                'audit': {'timestamp': audit_payload['timestamp']}
            }
        }

        self.last_result = (annotated_frame, consolidated_state)
        return annotated_frame, consolidated_state

    def draw_hud(self, frame, face_p, mask_p, security_p, fusion_p):
        x, y, w, h = face_p['box']

        # Determine box color
        color_map = {
            "GREEN": (0, 212, 170),   # Mint Green
            "BLUE": (255, 158, 75),   # Cyan/Blue
            "ORANGE": (0, 140, 255),  # Safety Amber/Orange
            "RED": (80, 80, 255),     # Crimson Red
            "YELLOW": (0, 215, 255)   # Amber Yellow
        }
        bgr = color_map.get(security_p['color'], (0, 255, 0))

        # Draw bounding box corner brackets
        cv2.rectangle(frame, (x, y), (x + w, y + h), bgr, 2)

        # Top label card
        label = f"{security_p['candidate']} ({fusion_p['calibrated_confidence']}%)"
        cv2.rectangle(frame, (x, y - 28), (x + w, y), bgr, -1)
        cv2.putText(frame, label, (x + 6, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Bottom sublabel card (mask + decision status)
        sublabel = f"{mask_p['mask_status']} | {security_p['decision']}"
        cv2.rectangle(frame, (x, y + h), (x + w, y + h + 22), (20, 20, 20), -1)
        cv2.putText(frame, sublabel, (x + 6, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, bgr, 1)

        return frame
