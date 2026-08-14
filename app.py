import os
import sys

# Suppress OpenCV C++ DNN internal warnings before cv2 import
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["OPENCV_FFMPEG_LOG_LEVEL"] = "-8"

import time
from datetime import datetime
import threading
import cv2
import numpy as np
import base64
import csv
from io import StringIO
from flask import Flask, render_template, Response, jsonify, request, make_response
from flask_socketio import SocketIO, emit

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import mfr

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'mfr-x-secret-key-2026')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet', ping_timeout=60, ping_interval=25)

class AppState:
    def __init__(self):
        self.orchestrator = None
        self.db = mfr.Database("db.json")

        self.strict_mask_mode = False
        self.logs = []

        # Live State for Frontend
        self.live_name = "-"
        self.live_mask = "-"
        self.live_score = "0.0%"
        self.live_status_msg = "SCANNING..."
        self.live_status_color = "#58a6ff"
        self.live_explanation = "Awaiting face stream..."
        self.agent_telemetry = {}

        # Enrollment State
        self.registering = False
        self.reg_name = ""
        self.reg_frames = []
        self.reg_status_text = "Stand by..."
        self.reg_progress = 0.0

state = AppState()

def sanitize_for_json(obj):
    """Recursively convert numpy arrays → lists and drop non-serializable objects."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items() if not isinstance(v, np.ndarray)}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(i) for i in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return None  # strip raw embeddings; not useful in UI anyway
    return obj


def log_event(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_str = f"[{timestamp}] {message}"
    state.logs.append(log_str)
    print(log_str)
    if len(state.logs) > 100:
        state.logs.pop(0)

def initialize_orchestrator():
    log_event("MFR-X: Initializing Multi-Agent Biometric Orchestrator...")
    try:
        state.orchestrator = mfr.BiometricOrchestrator(models_dir="models", db_path="db.json")
        log_event("MFR-X: 10 AI Specialized Agents Loaded Successfully.")
    except Exception as e:
        log_event(f"ERROR: Failed to initialize Biometric Orchestrator: {e}")

threading.Thread(target=initialize_orchestrator, daemon=True).start()

def finalize_registration():
    state.reg_status_text = "Computing average feature embeddings..."
    log_event(f"SYSTEM: Captured 5 face templates for '{state.reg_name}'. Finalizing enrollment...")

    full_embs = []
    upper_embs = []

    recognizer = state.orchestrator.recognition_agent.recognizer if state.orchestrator else None

    for idx, aligned_face in enumerate(state.reg_frames):
        try:
            if recognizer:
                f_emb = recognizer.extract_feature(aligned_face)
                u_emb = recognizer.extract_upper_face_feature(aligned_face)
                full_embs.append(f_emb)
                upper_embs.append(u_emb)
        except Exception as e:
            print(f"Embedding extraction error on frame {idx}: {e}")

    if len(full_embs) > 0:
        avg_full_emb = np.mean(full_embs, axis=0)
        avg_upper_emb = np.mean(upper_embs, axis=0)
        state.db.register_user(state.reg_name, avg_full_emb, avg_upper_emb)
        log_event(f"DATABASE: Successfully registered profile '{state.reg_name}'")
        state.reg_status_text = f"Enrollment Successful! '{state.reg_name}' registered."
    else:
        state.reg_status_text = "Registration failed. Unable to extract features."
        log_event(f"ERROR: Enrollment failed for '{state.reg_name}'.")

    time.sleep(2)
    state.reg_progress = 0.0
    state.reg_status_text = "Stand by..."

@socketio.on('image')
def handle_image(data):
    if state.orchestrator is None:
        emit('response', {
            'image': data,
            'state': {
                'name': '-',
                'mask': '-',
                'score': '0.0%',
                'status_msg': 'INITIALIZING AI MODELS...',
                'status_color': '#d29922',
                'explanation': 'Loading YuNet & SFace models...',
                'agents': {}
            }
        })
        return

    try:
        img_data = base64.b64decode(data.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        state.orchestrator.set_strict_mode(state.strict_mask_mode)
        annotated_frame, telemetry = state.orchestrator.process_frame(frame)

        if telemetry['detected']:
            agents = telemetry['agents']
            mask_data = agents['mask']
            sec_data = agents['security']

            state.live_name = telemetry['candidate']
            state.live_mask = f"{mask_data['mask_status']} ({mask_data['mask_confidence']}%)"
            state.live_score = f"{telemetry['confidence']:.1f}%"
            state.live_status_msg = sec_data['message']
            state.live_explanation = telemetry['explanation']
            state.agent_telemetry = agents

            color_hex_map = {
                "GREEN": "#2ea043",
                "BLUE": "#58a6ff",
                "ORANGE": "#d29922",
                "RED": "#f85149",
                "YELLOW": "#d29922"
            }
            state.live_status_color = color_hex_map.get(sec_data['color'], "#58a6ff")

            # Handle user registration capture stream
            if state.registering:
                if mask_data['is_masked']:
                    state.reg_status_text = "ALERT: Please remove mask to register!"
                    state.reg_progress = 0.0
                else:
                    try:
                        raw_face = agents['recognition'].get('raw', None)
                        face_p = state.orchestrator.detection_agent.detector.detect(frame)
                        if face_p:
                            aligned_face = state.orchestrator.recognition_agent.recognizer.align_crop(frame, face_p[0]['raw'])
                            state.reg_frames.append(aligned_face)
                            state.reg_progress = (len(state.reg_frames) / 5.0) * 100.0
                            state.reg_status_text = f"Capturing biometric template {len(state.reg_frames)}/5..."

                            if len(state.reg_frames) == 5:
                                state.registering = False
                                threading.Thread(target=finalize_registration, daemon=True).start()
                    except Exception as e:
                        print(f"Registration error: {e}")
        else:
            state.live_name = "-"
            state.live_mask = "-"
            state.live_score = "0.0%"
            state.live_status_msg = "SCANNING CAMERA FEED..."
            state.live_status_color = "#58a6ff"
            state.live_explanation = telemetry['explanation']

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        emit('response', {
            'image': 'data:image/jpeg;base64,' + jpg_as_text,
            'state': {
                'name': state.live_name,
                'mask': state.live_mask,
                'score': state.live_score,
                'status_msg': state.live_status_msg,
                'status_color': state.live_status_color,
                'explanation': state.live_explanation,
                'agents': sanitize_for_json(state.agent_telemetry)
            }
        })
    except Exception as e:
        print(f"Error in handle_image socket handler: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/state')
def api_state():
    return jsonify({
        "name": state.live_name,
        "mask": state.live_mask,
        "score": state.live_score,
        "status_msg": state.live_status_msg,
        "status_color": state.live_status_color,
        "explanation": state.live_explanation,
        "agents": sanitize_for_json(state.agent_telemetry),
        "strict_mode": state.strict_mask_mode
    })

@app.route('/api/logs')
def api_logs():
    return jsonify({"logs": state.logs})

@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.json
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Name is required"}), 400

    state.reg_name = name
    state.reg_frames = []
    state.reg_progress = 0.0
    state.reg_status_text = "Initializing multi-agent biometric acquisition..."
    state.registering = True
    log_event(f"SYSTEM: Initiated profile enrolment for '{name}'")
    return jsonify({"success": True})

@app.route('/api/register_status')
def api_register_status():
    return jsonify({
        "status_text": state.reg_status_text,
        "progress": state.reg_progress,
        "is_registering": state.registering
    })

@app.route('/api/directory')
def api_directory():
    names = state.db.get_registered_names()
    return jsonify({"users": names})

@app.route('/api/delete_user', methods=['POST'])
def api_delete_user():
    data = request.json
    name = data.get("name")
    if name:
        state.db.delete_user(name)
        log_event(f"DATABASE: Deleted user profile '{name}'")
        return jsonify({"success": True})
    return jsonify({"error": "Name not provided"}), 400

@app.route('/api/settings', methods=['POST'])
def api_settings():
    data = request.json
    if "strict_mode" in data:
        state.strict_mask_mode = data["strict_mode"]
    if "threshold" in data and state.orchestrator:
        state.orchestrator.set_cosine_threshold(float(data["threshold"]))
    if "interval" in data and state.orchestrator:
        state.orchestrator.frame_interval = int(data["interval"])
    return jsonify({"success": True})

@app.route('/api/wipe_db', methods=['POST'])
def api_wipe_db():
    if os.path.exists("db.json"):
        os.remove("db.json")
    state.db = mfr.Database("db.json")
    log_event("DATABASE: Cleared facial biometric database file.")
    return jsonify({"success": True})

@app.route('/api/export_logs')
def api_export_logs():
    si = StringIO()
    cw = csv.writer(si)
    cw.writerow(["Timestamp", "Type", "Log Message"])
    for log in state.logs:
        parts = log.split('] ', 1)
        if len(parts) == 2:
            ts = parts[0].replace('[', '')
            msg = parts[1]
            log_type = "SYSTEM"
            if "DETECTED:" in msg: log_type = "DETECTION"
            elif "ALERT:" in msg or "VIOLATION:" in msg: log_type = "ALERT"
            elif "DATABASE:" in msg: log_type = "DATABASE"
            cw.writerow([ts, log_type, msg])
        else:
            cw.writerow(["", "", log])

    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=mfrx_security_audit_log.csv"
    output.headers["Content-type"] = "text/csv"
    return output

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
