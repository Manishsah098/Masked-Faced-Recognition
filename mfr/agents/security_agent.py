class SecurityAgent:
    """
    Decision Tier — Agent #9: Security / Risk Agent
    Enforces access control policies and safety mandates. Assigns actionable risk ratings
    (LOW, MEDIUM, HIGH, CRITICAL) and access status decisions.
    """

    def __init__(self, strict_mode=False):
        self.strict_mode = strict_mode

    def process(self, fusion_p, mask_p, liveness_p, occlusion_p, quality_p):
        confidence = fusion_p['calibrated_confidence']
        candidate = fusion_p['candidate']
        is_masked = mask_p['is_masked']
        is_live = liveness_p['is_live']
        strategy = occlusion_p['strategy']

        # Rule 1: Mask Mandate Check
        if self.strict_mode and not is_masked:
            return {
                'decision': "MASK_VIOLATION",
                'risk_level': "MEDIUM",
                'candidate': candidate if candidate != "Unknown" else "Unregistered User",
                'color': "ORANGE",
                'code': 403,
                'message': "Mask Mandate Breach — Face Mask Required"
            }

        # Rule 2: Anti-Spoof Liveness Check
        if not is_live:
            return {
                'decision': "ACCESS_DENIED",
                'risk_level': "CRITICAL",
                'candidate': "Spoof Attack Detected",
                'color': "RED",
                'code': 401,
                'message': "Presentation Attack / Liveness Failure"
            }

        # Rule 3: High Occlusion / Low Quality -> Review Required
        if strategy == "INSUFFICIENT" or not quality_p['is_acceptable']:
            return {
                'decision': "REVIEW_REQUIRED",
                'risk_level': "HIGH",
                'candidate': candidate if candidate != "Unknown" else "Inconclusive",
                'color': "YELLOW",
                'code': 202,
                'message': "Verification Inconclusive — High Occlusion / Low Quality"
            }

        # Rule 4: Standard Identification Check
        if candidate != "Unknown" and confidence >= 50.0:
            return {
                'decision': "VERIFIED",
                'risk_level': "LOW",
                'candidate': candidate,
                'color': "BLUE" if is_masked else "GREEN",
                'code': 200,
                'message': "Access Granted — Profile Verified"
            }
        else:
            return {
                'decision': "ACCESS_DENIED",
                'risk_level': "HIGH",
                'candidate': "Unknown",
                'color': "RED",
                'code': 403,
                'message': "Access Denied — Unregistered Face"
            }
