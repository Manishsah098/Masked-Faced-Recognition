import os
import sys
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

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import mfr

app = Flask(__name__)
# Disable static file caching so CSS/JS changes reflect immediately in dev
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
# Initialize SocketIO with eventlet
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Global Application State
class AppState:
    def __init__(self):
        self.detector = None
        self.recognizer = None
        self.mask_detector = None
        self.db = mfr.Database()
        
        self.frame_count = 0
        self.detection_interval = 1 # Run on every socket frame for real-time latency
        self.strict_mask_mode = False
        
        self.cached_results = []
        self.logs = []
        self.seen_detections = {}
        
        # Live Stats for Frontend
        self.live_name = "-"
        self.live_mask = "-"
        self.live_score = "0.000"
        self.live_status_msg = "SCANNING..."
        self.live_status_color = "#58a6ff" # blue
        
        # Registration State
        self.registering = False
        self.reg_name = ""
        self.reg_frames = []
        self.reg_status_text = "Stand by..."
        self.reg_progress = 0.0

state = AppState()

def log_event(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_str = f"[{timestamp}] {message}"
    state.logs.append(log_str)
    print(log_str)
    # Keep only last 100 logs in memory
    if len(state.logs) > 100:
        state.logs.pop(0)

def track_detection_events(name, mask_status, score):
    current_time = time.time()
    
    # Expire older trackings
    for k in list(state.seen_detections.keys()):
        if current_time - state.seen_detections[k] > 10.0:
            del state.seen_detections[k]
            
    if name != "Unknown":
        if name not in state.seen_detections:
            state.seen_detections[name] = current_time
            status_msg = f"Masked (Score: {score:.2f})" if mask_status == "Masked" else "Unmasked"
            log_event(f"DETECTED: {name} - {status_msg}")
            
            if state.strict_mask_mode and mask_status == "Unmasked":
                log_event(f"VIOLATION: {name} is unmasked in safety zone!")
    else:
        if "Unknown" not in state.seen_detections or current_time - state.seen_detections["Unknown"] > 6.0:
            state.seen_detections["Unknown"] = current_time
            log_event("ALERT: Unregistered user detected!")

def initialize_models():
    log_event("SYSTEM: Initializing models...")
    try:
        mfr.ensure_models()
        state.detector = mfr.FaceDetector(mfr.get_model_path("yunet.onnx"))
        state.recognizer = mfr.FaceRecognizer(mfr.get_model_path("sface.onnx"))
        state.mask_detector = mfr.MaskDetector(mfr.get_model_path("mask_detector.onnx"))
        log_event("SYSTEM: All ONNX models loaded successfully.")
    except Exception as e:
        log_event(f"ERROR: Model initialization failed: {e}")

# Start model init immediately
threading.Thread(target=initialize_models, daemon=True).start()

def finalize_registration():
    state.reg_status_text = "Computing average feature embeddings..."
    log_event(f"SYSTEM: Captured 5 face templates for {state.reg_name}. Finalizing enrollment...")
    
    full_embs = []
    upper_embs = []
    
    for idx, aligned_face in enumerate(state.reg_frames):
        try:
            f_emb = state.recognizer.extract_feature(aligned_face)
            u_emb = state.recognizer.extract_upper_face_feature(aligned_face)
            full_embs.append(f_emb)
            upper_embs.append(u_emb)
        except Exception as e:
            print(f"Embedding error on frame {idx}: {e}")
            
    if len(full_embs) > 0:
        avg_full_emb = np.mean(full_embs, axis=0)
        avg_upper_emb = np.mean(upper_embs, axis=0)
        state.db.register_user(state.reg_name, avg_full_emb, avg_upper_emb)
        log_event(f"DATABASE: Successfully registered user '{state.reg_name}'")
        state.reg_status_text = f"Enrollment Successful! '{state.reg_name}' registered."
    else:
        state.reg_status_text = "Registration failed. Unable to extract features."
        log_event(f"ERROR: Enrollment failed for {state.reg_name}.")
    
    time.sleep(2)
    state.reg_progress = 0.0
    state.reg_status_text = "Stand by..."

def draw_premium_hud(frame):
    for res in state.cached_results:
        box = res['box']
        landmarks = res['landmarks']
        mask_lbl, mask_conf = res['mask_status']
        name, score = res['identity']
        
        x, y, w, h = box
        
        # Color Theme based on state (BGR for OpenCV)
        if name == "Unknown":
            color_theme = (73, 81, 248)       # Red
        else:
            if mask_lbl == "Masked":
                color_theme = (255, 166, 88)   # Blue
            else:
                if state.strict_mask_mode:
                    color_theme = (34, 153, 210) # Orange
                else:
                    color_theme = (67, 160, 46)  # Green
                    
        cv2.rectangle(frame, (x, y), (x + w, y + h), color_theme, 1, lineType=cv2.LINE_AA)
        
        length = min(20, int(w * 0.15))
        thick = 3
        cv2.line(frame, (x, y), (x + length, y), color_theme, thick, lineType=cv2.LINE_AA)
        cv2.line(frame, (x, y), (x, y + length), color_theme, thick, lineType=cv2.LINE_AA)
        cv2.line(frame, (x + w, y), (x + w - length, y), color_theme, thick, lineType=cv2.LINE_AA)
        cv2.line(frame, (x + w, y), (x + w, y + length), color_theme, thick, lineType=cv2.LINE_AA)
        cv2.line(frame, (x, y + h), (x + length, y + h), color_theme, thick, lineType=cv2.LINE_AA)
        cv2.line(frame, (x, y + h), (x, y + h - length), color_theme, thick, lineType=cv2.LINE_AA)
        cv2.line(frame, (x + w, y + h), (x + w - length, y + h), color_theme, thick, lineType=cv2.LINE_AA)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - length), color_theme, thick, lineType=cv2.LINE_AA)

        for pt in landmarks:
            cv2.circle(frame, tuple(pt), 3, (244, 244, 50), -1, lineType=cv2.LINE_AA)

        overlay = frame.copy()
        card_height = 28
        cv2.rectangle(overlay, (x, y - card_height), (x + w, y), color_theme, -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        desc_text = f"{name} | {mask_lbl} ({int(mask_conf*100)}%)"
        cv2.putText(frame, desc_text, (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, lineType=cv2.LINE_AA)

# SOCKET IO COMMUNICATION
@socketio.on('image')
def handle_image(data):
    if state.detector is None or state.recognizer is None or state.mask_detector is None:
        return
        
    try:
        # Decode base64 image from client webcam
        img_data = base64.b64decode(data.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return
            
        state.frame_count += 1
        
        # Run inference on frame
        faces = state.detector.detect(frame)
        state.cached_results = []
        
        for face in faces:
            box = face['box']
            mask_label, mask_conf = state.mask_detector.predict(frame, box)
            
            try:
                aligned_face = state.recognizer.align_crop(frame, face['raw'])
                ycrcb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2YCrCb)
                lower_skin = np.array([0, 133, 77], dtype=np.uint8)
                upper_skin = np.array([255, 173, 127], dtype=np.uint8)
                skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
                
                forehead_ratio = np.mean(skin_mask[10:45, 35:77] == 255)
                mouth_ratio = np.mean(skin_mask[65:105, 20:92] == 255)
                
                is_occluded = False
                if forehead_ratio > 0.15:
                    if mouth_ratio < 0.45 or (mouth_ratio / forehead_ratio) < 0.55:
                        is_occluded = True
                else:
                    if mouth_ratio < 0.35:
                        is_occluded = True
                        
                if is_occluded and mask_label == "Unmasked":
                    mask_label = "Masked"
                    mask_conf = float(1.0 - mouth_ratio)
                
                if mask_label == "Masked":
                    emb = state.recognizer.extract_upper_face_feature(aligned_face)
                    name, score = state.db.match_face(emb, mode="upper", recognizer=state.recognizer)
                else:
                    emb = state.recognizer.extract_feature(aligned_face)
                    name, score = state.db.match_face(emb, mode="full", recognizer=state.recognizer)
            except Exception as e:
                print(f"Alignment error: {e}")
                name, score = "Unknown", 0.0
                
            state.cached_results.append({
                'box': box, 'landmarks': face['landmarks'],
                'mask_status': (mask_label, mask_conf), 'identity': (name, score),
                'raw': face['raw']
            })
            
            track_detection_events(name, mask_label, score)
            
        # Registration capture workflow
        if state.registering:
            if len(state.cached_results) == 1:
                face_data = state.cached_results[0]
                mask_lbl, _ = face_data['mask_status']
                
                if mask_lbl == "Masked":
                    state.reg_status_text = "ALERT: Remove mask to register!"
                    state.reg_progress = 0.0
                else:
                    try:
                        aligned_face = state.recognizer.align_crop(frame, face_data['raw'])
                        state.reg_frames.append(aligned_face)
                        state.reg_progress = (len(state.reg_frames) / 5.0) * 100.0
                        state.reg_status_text = f"Capturing template {len(state.reg_frames)}/5..."
                        
                        if len(state.reg_frames) == 5:
                            state.registering = False
                            threading.Thread(target=finalize_registration, daemon=True).start()
                    except Exception as e:
                        print(f"Registration error: {e}")
            elif len(state.cached_results) == 0:
                state.reg_status_text = "Please align your face inside the frame."
            else:
                state.reg_status_text = "Multiple faces detected! Stand alone to register."

        # Update live state metrics
        if len(state.cached_results) > 0:
            res = state.cached_results[0]
            mask_lbl, mask_conf = res['mask_status']
            name, score = res['identity']
            
            state.live_name = name
            state.live_mask = f"{mask_lbl} ({int(mask_conf*100)}%)"
            state.live_score = f"{score:.3f}" if score > 0 else "0.000"
            
            if name == "Unknown":
                state.live_status_msg = "UNAUTHORIZED ACCESS"
                state.live_status_color = "#f85149"
            elif mask_lbl == "Unmasked" and state.strict_mask_mode:
                state.live_status_msg = "ACCESS BLOCKED: MASK REQUIRED"
                state.live_status_color = "#d29922"
            else:
                state.live_status_msg = "AUTHORIZED: SECURE"
                state.live_status_color = "#2ea043" if mask_lbl == "Unmasked" else "#58a6ff"
        else:
            state.live_name = "-"
            state.live_mask = "-"
            state.live_score = "-"
            state.live_status_msg = "SCANNING..."
            state.live_status_color = "#58a6ff"

        # Apply HUD overlay
        annotated_frame = frame.copy()
        draw_premium_hud(annotated_frame)
        
        # Encode back to JPEG and emit base64
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        
        emit('response', {
            'image': 'data:image/jpeg;base64,' + jpg_as_text,
            'state': {
                'name': state.live_name,
                'mask': state.live_mask,
                'score': state.live_score,
                'status_msg': state.live_status_msg,
                'status_color': state.live_status_color
            }
        })
    except Exception as e:
        print(f"Error in handle_image: {e}")

# HTTP ROUTES
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
    state.reg_status_text = "Initializing biometric capture..."
    state.registering = True
    log_event(f"SYSTEM: Initiated face profile acquisition for '{name}'")
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
    if "threshold" in data and state.recognizer:
        state.recognizer.cosine_threshold = float(data["threshold"])
    if "interval" in data:
        state.detection_interval = int(data["interval"])
    return jsonify({"success": True})

@app.route('/api/wipe_db', methods=['POST'])
def api_wipe_db():
    if os.path.exists("db.json"):
        os.remove("db.json")
    state.db = mfr.Database()
    log_event("DATABASE: Completely wiped facial database file.")
    return jsonify({"success": True})

@app.route('/api/export_logs')
def api_export_logs():
    """Generates a downloadable CSV audit file of all system events."""
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
    output.headers["Content-Disposition"] = "attachment; filename=system_audit_log.csv"
    output.headers["Content-type"] = "text/csv"
    return output

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
