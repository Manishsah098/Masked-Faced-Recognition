from collections import deque, Counter

class TemporalTrackingAgent:
    """
    Biometric Tier — Agent #7: Temporal Tracking Agent
    Maintains a 5-frame sliding window of identity candidates and similarity scores
    to smooth out frame-by-frame anomalies.
    """

    def __init__(self, window_size=5):
        self.window_size = window_size
        self.candidate_history = deque(maxlen=window_size)
        self.score_history = deque(maxlen=window_size)

    def process(self, recognition_payload):
        candidate = recognition_payload['candidate']
        score = recognition_payload['similarity_score']

        self.candidate_history.append(candidate)
        self.score_history.append(score)

        if not self.candidate_history:
            return {
                'temporal_candidate': "Unknown",
                'temporal_stability_pct': 0.0,
                'window_count': 0
            }

        # Majority vote
        counts = Counter(self.candidate_history)
        top_candidate, top_count = counts.most_common(1)[0]

        # Calculate average similarity score for the top candidate in window
        scores = [s for c, s in zip(self.candidate_history, self.score_history) if c == top_candidate]
        avg_score = float(sum(scores) / len(scores)) if scores else 0.0

        stability_pct = float(round((top_count / float(len(self.candidate_history))) * 100.0, 1))

        return {
            'temporal_candidate': top_candidate,
            'temporal_avg_score': float(round(avg_score, 4)),
            'temporal_stability_pct': stability_pct,
            'window_count': len(self.candidate_history)
        }

    def reset(self):
        self.candidate_history.clear()
        self.score_history.clear()
