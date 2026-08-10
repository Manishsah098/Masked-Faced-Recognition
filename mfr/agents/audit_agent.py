import time

class AuditAgent:
    """
    System Intelligence Tier — Agent #10: Audit & AI Explanation Agent
    Generates structured telemetry audit records and synthesizes human-readable
    natural language diagnostic explanations for operators.
    """

    def __init__(self):
        self.logs = []

    def generate_explanation(self, security_p, fusion_p, occlusion_p, quality_p, liveness_p, mask_p):
        decision = security_p['decision']
        candidate = security_p['candidate']
        confidence = fusion_p['calibrated_confidence']
        occlusion = occlusion_p['overall_visible_pct']
        mask_status = mask_p['mask_status']

        if decision == "VERIFIED":
            if mask_status == "Masked":
                return f"Subject {candidate} verified with upper-face strategy ({confidence}% confidence). Mask detected and visible facial area is {occlusion}%."
            else:
                return f"Subject {candidate} verified with full-face baseline ({confidence}% confidence). Full face visible."
        elif decision == "MASK_VIOLATION":
            return f"Access restricted for {candidate}. Safety policy mandates mask wearing, but face was detected unmasked."
        elif decision == "REVIEW_REQUIRED":
            return f"Verification inconclusive ({confidence}% confidence). High occlusion detected ({occlusion}% visible) or low image quality ({quality_p['quality_score']}%). Operator review recommended."
        elif decision == "ACCESS_DENIED":
            if not liveness_p['is_live']:
                return "Access denied due to potential presentation attack (liveness check failed)."
            else:
                return "Access denied: Bounding box face embedding does not match any enrolled profile."
        else:
            return "System awaiting clear face acquisition."

    def process(self, security_p, fusion_p, occlusion_p, quality_p, liveness_p, mask_p):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        explanation = self.generate_explanation(security_p, fusion_p, occlusion_p, quality_p, liveness_p, mask_p)

        audit_entry = {
            'timestamp': timestamp,
            'decision': security_p['decision'],
            'candidate': security_p['candidate'],
            'risk_level': security_p['risk_level'],
            'confidence_pct': fusion_p['calibrated_confidence'],
            'mask_status': mask_p['mask_status'],
            'explanation': explanation
        }

        self.logs.append(audit_entry)
        if len(self.logs) > 200:
            self.logs.pop(0)

        return {
            'timestamp': timestamp,
            'explanation': explanation,
            'recent_logs': self.logs[-50:]
        }
