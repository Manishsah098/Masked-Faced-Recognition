import numpy as np

class FusionAgent:
    """
    Decision Tier — Agent #8: Fusion & Confidence Agent
    Consolidates signals from Perception, Biometric, and Temporal agents
    into a calibrated overall confidence score (0–100%).
    """

    def process(self, quality_p, mask_p, occlusion_p, recognition_p, liveness_p, tracking_p):
        rec_score = recognition_p.get('similarity_pct', 0.0)
        qual_score = quality_p.get('quality_score', 0.0)
        live_score = liveness_p.get('liveness_score', 0.0)
        temp_stability = tracking_p.get('temporal_stability_pct', 0.0)
        occlusion_pct = occlusion_p.get('overall_visible_pct', 0.0)

        is_match = recognition_p.get('is_match', False)
        candidate = tracking_p.get('temporal_candidate', 'Unknown')

        if candidate == "Unknown" or not is_match:
            calibrated_confidence = max(0.0, rec_score * 0.5)
        else:
            # Calibrated weighted score
            calibrated_confidence = float(np.round(
                0.45 * rec_score +
                0.20 * live_score +
                0.15 * qual_score +
                0.10 * temp_stability +
                0.10 * occlusion_pct,
                1
            ))

        return {
            'calibrated_confidence': min(99.9, max(0.0, calibrated_confidence)),
            'candidate': candidate,
            'signals': {
                'recognition_pct': rec_score,
                'liveness_pct': live_score,
                'quality_pct': qual_score,
                'temporal_pct': temp_stability,
                'occlusion_pct': occlusion_pct
            }
        }
