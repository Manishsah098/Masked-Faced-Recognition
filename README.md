# 🎭 Masked Face Recognition System

A real-time, premium computer vision application for **Masked Face Recognition (MFR)**. It runs entirely on the CPU using OpenCV's native Deep Neural Network (DNN) classes and pre-trained ONNX models, providing high-speed inference without heavy frameworks like TensorFlow or PyTorch.

The system ships in **two modes**:
- **🖥️ Desktop App** — native Tkinter GUI (`main.py`)
- **🌐 Web Server** — Flask + Socket.IO browser interface (`app.py`)

---

## ✨ Key Features

- **Real-Time Biometric Scanner** — Detects faces, maps facial landmarks, and identifies users on a live webcam feed.
- **ONNX Mask Detection Classifier** — Automatically classifies whether a face is `Masked` or `Unmasked`.
- **Adaptive Upper-Face Matching** — Dynamically switches matching strategy based on mask presence:
  - **Unmasked Users** → Full-face embeddings (highest accuracy).
  - **Masked Users** → Blackouts the lower face region (`y >= 65`) of both database templates and the live query crop, comparing only visible upper-face geometry (eyes, eyebrows, forehead).
- **Profile Enrolment Workflow** — Captures 5 face samples, averages their embeddings, and registers a secure biometric template.
- **Safety Compliance Mode** — Mandate face masks; flags and logs violations in real time.
- **System Audit Console** — Logs biometric scans, compliance audits, and access denials with precise timestamps.
- **Exportable Logs** — Download the full audit trail as a `.csv` file (web mode).
- **Interactive Calibration** — Adjust matching sensitivity (SFace Cosine Threshold) and frame-skip interval for CPU optimization.

---

## 🧠 Technical Architecture & Models

Three ONNX deep learning models are downloaded automatically on first launch:

| # | Model | File | Purpose |
|---|-------|------|---------|
| 1 | **YuNet** | `face_detection_yunet_2023mar.onnx` | Sub-millisecond face detector + 5-point landmark extractor |
| 2 | **SFace** | `face_recognition_sface_2021dec.onnx` | Sigmoid-constrained hypersphere face recognition |
| 3 | **MobileNetV2** | `mask_detector.onnx` | Binary mask-compliance classifier |

---

## 📦 Installation & Setup

Ensure you have **Python 3.12 or higher** installed. Tested up to **Python 3.13.7 on Windows**.

### 1. Clone / Open the Workspace

```powershell
git clone https://github.com/Manishsah098/Masked-Faced-Recognition.git
cd "Masked Face Recognition"
```

### 2. Create a Virtual Environment

```powershell
python -m venv .venv
```

### 3. Activate the Environment

| Platform | Shell | Command |
|----------|-------|---------|
| Windows | PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows | CMD | `.venv\Scripts\activate.bat` |
| macOS / Linux | Bash | `source .venv/bin/activate` |

### 4. Install Dependencies

**For the Desktop App:**
```bash
pip install opencv-python numpy pillow
```

**For the Web Server:**
```bash
pip install opencv-python numpy pillow flask flask-socketio eventlet
```

> **Note:** On the first launch, models are automatically downloaded to the local `models/` directory. This may take a moment.

---

## 🚀 Running the Application

### Option A — Desktop App (Tkinter GUI)

```powershell
python main.py
```

### Option B — Web Server (Flask + Socket.IO)

```powershell
python app.py
```

Then open your browser and navigate to:

```
http://localhost:5000
```

> The server listens on `0.0.0.0:5000`, so it is also accessible from other devices on the same local network via `http://<your-ip>:5000`.

---

## 🖱️ Operating Instructions

### 1. Registering a New User Profile

1. Click the **Register User** tab in the left sidebar.
2. Type the person's name in the **Full Name** entry box.
3. Instruct the user to stand in front of the camera and **remove their face mask** *(enrolment enforces unmasked faces for a clean full-face baseline)*.
4. Click **Start Profile Acquisition**.
5. Stay still — the system captures 5 biometric frames, extracts embeddings, averages them, and saves the template. A confirmation status displays when complete.

### 2. Live Scanning & Identification

1. Click the **Live Camera** tab.
2. Stand in front of the camera. A colored HUD box is drawn around your face:

| Box Color | Meaning |
|-----------|---------|
| 🟢 Green | Identified — Unmasked |
| 🔵 Blue | Identified — Masked (Upper-Face Matching active) |
| 🔴 Red | Unregistered face — **Access Denied** |
| 🟠 Orange | Registered but unmasked in Safety Mode — **Violation** |

### 3. Safety Compliance Mode (Mandate Masks)

1. In the **Live Camera** tab, locate the **Access Directives** panel.
2. Toggle **Mandate Face Mask** ON.
3. A registered user detected without a mask will trigger an orange warning and a log entry.

### 4. Viewing Logs & User Directory

- **User Directory** — See all enrolled profiles; delete individual profiles permanently.
- **System Logs** — Live feed of all system events, scan matches, and compliance violations.
- **Export Logs** *(web mode only)* — Download a full `.csv` audit report via the API endpoint `/api/export_logs`.

### 5. Calibrating Sensitivity

Navigate to the **System Settings** tab:

| Setting | Default | Effect |
|---------|---------|--------|
| **Cosine Threshold** | `0.363` | Higher → stricter match, fewer false positives |
| **Frame Skip** | `1` | Higher → less CPU load, lower real-time accuracy |

---

## 🌐 Web API Reference (app.py)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the main web UI |
| `GET` | `/api/state` | Returns live detection state (name, mask, score, status) |
| `GET` | `/api/logs` | Returns the last 100 system log entries |
| `GET` | `/api/directory` | Lists all registered user names |
| `GET` | `/api/register_status` | Returns current registration progress |
| `GET` | `/api/export_logs` | Downloads full audit log as `system_audit_log.csv` |
| `POST` | `/api/register` | Initiates enrolment for a given `name` |
| `POST` | `/api/delete_user` | Deletes a user profile by `name` |
| `POST` | `/api/settings` | Updates `strict_mode`, `threshold`, or `interval` |
| `POST` | `/api/wipe_db` | Wipes the entire biometric database |

**Socket.IO Events:**

| Direction | Event | Payload | Description |
|-----------|-------|---------|-------------|
| Client → Server | `image` | Base64 JPEG string | Send a webcam frame for processing |
| Server → Client | `response` | `{ image, state }` | Annotated frame + live detection state |

---

## 🗂️ Workspace Structure

```
Masked Face Recognition/
├── app.py                  # Flask + Socket.IO web server
├── main.py                 # Tkinter graphical interface and pipeline driver
├── db.json                 # Local serialized biometric database
├── verify_pipeline.py      # Diagnostic / smoke-test script
├── models/                 # Cached ONNX model files (auto-generated on first run)
│   ├── face_detection_yunet_2023mar.onnx
│   ├── face_recognition_sface_2021dec.onnx
│   └── mask_detector.onnx
├── mfr/                    # Core biometric pipeline package
│   ├── __init__.py
│   ├── detector.py         # YuNet face detector wrapper (internally downscaled)
│   ├── recognizer.py       # SFace aligner and upper-face masking logic
│   ├── mask_detector.py    # ONNX mask compliance classifier
│   ├── database.py         # Ser/Des database manager
│   └── utils.py            # Biometric model network downloader
├── templates/
│   └── index.html          # Web UI template (used by app.py)
└── static/                 # Static assets (CSS, JS) for the web UI
```

---

## 🔧 Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: flask_socketio` | Run `pip install flask flask-socketio eventlet` |
| `ModuleNotFoundError: cv2` | Run `pip install opencv-python` |
| Camera not detected | Ensure no other application is using the webcam |
| Models not downloading | Check your internet connection; or manually place ONNX files in `models/` |
| Slow inference | Increase the **Frame Skip** slider in System Settings |
| False positives / negatives | Adjust the **Cosine Threshold** in System Settings |

---

## 📄 License

This project is for academic and research purposes.
