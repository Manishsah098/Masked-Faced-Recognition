import os
import json
import numpy as np

class Database:
    """Manages the registration and matching of user facial embeddings stored locally in a JSON database."""
    def __init__(self, db_path="db.json"):
        self.db_path = db_path
        self.users = {}  # Format: { name: { "full": [...], "upper": [...] } }
        self.load()

    def load(self):
        """Loads registered users and their embeddings from the JSON database file."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    # Convert lists back to numpy arrays for compatibility
                    self.users = {}
                    for name, embs in data.items():
                        self.users[name] = {
                            "full": np.array(embs["full"], dtype=np.float32),
                            "upper": np.array(embs["upper"], dtype=np.float32)
                        }
            except Exception as e:
                print(f"Error loading database: {e}")
                self.users = {}
        else:
            self.users = {}

    def save(self):
        """Serializes and saves the database to the JSON file."""
        try:
            data = {}
            for name, embs in self.users.items():
                data[name] = {
                    "full": embs["full"].tolist(),
                    "upper": embs["upper"].tolist()
                }
            with open(self.db_path, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error saving database: {e}")

    def register_user(self, name, full_emb, upper_emb):
        """Registers or updates a user in the database."""
        self.users[name] = {
            "full": np.array(full_emb, dtype=np.float32),
            "upper": np.array(upper_emb, dtype=np.float32)
        }
        self.save()

    def delete_user(self, name):
        """Deletes a user from the database by name."""
        if name in self.users:
            del self.users[name]
            self.save()
            return True
        return False

    def get_registered_names(self):
        """Returns a list of all registered user names."""
        return list(self.users.keys())

    def match_face(self, live_emb, mode="full", recognizer=None):
        """
        Matches a live face embedding against the database.
        mode: "full" (standard face match) or "upper" (upper-face-only match)
        recognizer: FaceRecognizer instance (to compute similarity and retrieve threshold)
        Returns (best_match_name, score). Returns ("Unknown", 0.0) if no match meets threshold.
        """
        if not self.users or recognizer is None:
            return "Unknown", 0.0

        best_name = "Unknown"
        best_score = -1.0
        
        for name, embs in self.users.items():
            db_emb = embs["full"] if mode == "full" else embs["upper"]
            score = recognizer.compute_similarity(live_emb, db_emb)
            if score > best_score:
                best_score = score
                best_name = name

        # Check if the best match exceeds the recognizer's matching threshold
        # SFace default cosine threshold is typically 0.363
        # However, for upper-face-only match, since we blacked out 40% of the image, the similarity score
        # distribution might slightly shift upwards or downwards. In practice, 0.363 is still a very safe
        # baseline, but we can set a slightly lower threshold for upper-face matching (e.g. 0.33) if needed,
        # or stick to the recognizer's configured threshold.
        threshold = recognizer.cosine_threshold
        if mode == "upper":
            # Upper face has less info, so we can be slightly more lenient or keep same threshold
            threshold = max(0.30, recognizer.cosine_threshold - 0.03)

        if best_score >= threshold:
            return best_name, best_score
            
        return "Unknown", best_score
