class OcclusionAgent:
    """
    Perception Tier — Agent #4: Occlusion Agent
    Calculates visible facial region percentages across anatomical zones (Forehead, Eyes, Nose, Mouth)
    and selects the optimal feature extraction strategy (FULL_FACE, UPPER_FACE, or INSUFFICIENT).
    """

    def process(self, face_payload, mask_payload):
        landmarks = face_payload['landmarks']
        box = face_payload['box']
        is_masked = mask_payload['is_masked']

        x, y, w, h = box
        left_eye, right_eye, nose, left_mouth, right_mouth = landmarks

        # Zone visibility calculation
        forehead_vis = 95.0
        eyes_vis = 100.0 if (left_eye[1] > y and right_eye[1] > y) else 70.0

        if is_masked:
            nose_vis = 25.0
            mouth_vis = 0.0
        else:
            nose_vis = 95.0
            mouth_vis = 95.0

        # Weighted overall visible facial area percentage
        overall_visible_pct = float(round(
            0.25 * forehead_vis +
            0.35 * eyes_vis +
            0.20 * nose_vis +
            0.20 * mouth_vis,
            1
        ))

        # Select Strategy
        if overall_visible_pct >= 80.0:
            strategy = "FULL_FACE"
        elif overall_visible_pct >= 40.0:
            strategy = "UPPER_FACE"
        else:
            strategy = "INSUFFICIENT"

        return {
            'overall_visible_pct': overall_visible_pct,
            'strategy': strategy,
            'zones': {
                'forehead': forehead_vis,
                'eyes': eyes_vis,
                'nose': nose_vis,
                'mouth': mouth_vis
            }
        }
