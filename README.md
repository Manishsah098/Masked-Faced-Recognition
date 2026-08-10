# 🎭 MFR-X — Multi-Agent Real-Time Masked Face Recognition

A state-of-the-art, CPU-optimized computer vision and biometric intelligence system for **Multi-Agent Real-Time Occlusion-Aware Face Recognition**. **MFR-X** replaces traditional monolithic biometric pipelines with an orchestrated network of **10 specialized AI agents** that dynamically evaluate face quality, facial occlusion, anti-spoof liveness, temporal consistency, and identity confidence on live video feeds.

---

## 🏛 Multi-Agent System Architecture

```text
                         📷 LIVE CAMERA FEED
                              │
                              ▼
                    ┌───────────────────┐
                    │  ORCHESTRATOR     │
                    │      AGENT        │
                    └─────────┬─────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│ FACE DETECTION│     │ MASK ANALYSIS │     │ QUALITY AGENT │
│    AGENT      │     │     AGENT     │     │               │
└───────┬───────┘     └───────┬───────┘     └───────┬───────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                    ┌──────────────────┐
                    │ OCCLUSION AGENT  │
                    └─────────┬────────┘
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
       ┌──────────────────┐      ┌──────────────────┐
       │ RECOGNITION      │      │ LIVENESS /       │
       │ AGENT            │      │ ANTI-SPOOF AGENT │
       └────────┬─────────┘      └────────┬─────────┘
                │                         │
                └────────────┬────────────┘
                             ▼
                    ┌──────────────────┐
                    │ TEMPORAL         │
                    │ TRACKING AGENT   │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │ FUSION /         │
                    │ CONFIDENCE AGENT │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │ SECURITY / RISK  │
                    │ AGENT            │
                    └────────┬─────────┘
                             ▼
                 ┌───────────┼───────────┐
                 ▼           ▼           ▼
              VERIFIED     REVIEW      UNKNOWN
                 │           │           │
                 └───────────┼───────────┘
                             ▼
                    ┌──────────────────┐
                    │ AUDIT / AI       │
                    │ EXPLANATION AGENT│
                    └──────────────────┘
```

---

## ✨ Key Features & Multi-Agent Breakdown

The **MFR-X** agent ecosystem is divided into four distinct operational tiers:

```text
MFR-X AGENT ECOSYSTEM
├── 🎯 Orchestrator Agent (Master Execution Driver)
│
├── 🔍 Perception Tier
│   ├── 👁️ 1. Face Detection Agent (YuNet)
│   ├── 📐 2. Face Quality Agent (Blur/Pose/Resolution)
│   ├── 😷 3. Mask Analysis Agent (MobileNetV2)
│   └── 🔎 4. Occlusion Agent (Visible Region Computation)
│
├── 🧬 Biometric Tier
│   ├── 🧠 5. Recognition Agent (SFace Adaptive Aligner)
│   ├── 🛡️ 6. Liveness / Anti-Spoof Agent (Presentation Attack Defense)
│   └── 🎥 7. Temporal Tracking Agent (Sliding-Window Identity Smoothing)
│
├── ⚖️ Decision Tier
│   ├── 🧮 8. Fusion & Confidence Agent (Multi-Signal Score Calibration)
│   └── 🚨 9. Security / Risk Agent (Access Policy & Directive Engine)
│
└── 📊 System Intelligence Tier
    └── 📋 10. Audit & AI Explanation Agent (CSV Audit & NL Diagnostic Synthesis)
```

### 1. 🎯 Orchestrator Agent
The central decision engine that controls pipeline routing. It evaluates upstream agent responses and conditionally invokes downstream agents (e.g., bypassing recognition if face quality is below threshold or no face is present), reducing unnecessary CPU compute.

### 2. 👁️ Face Detection Agent
Powered by **YuNet**, this agent detects faces, extracts 5 key facial landmarks (eyes, nose, mouth corners), tracks bounding box geometries, and filters out false positives using spatial constraints.

### 3. 📐 Face Quality Agent
Performs pre-recognition validation by inspecting image sharpness (Laplacian variance), contrast, resolution, and tilt/pose angle. If image quality is insufficient, it triggers a user guidance output (*"Move Closer / Adjust Lighting"*) rather than risking false identifications.

### 4. 😷 Mask Analysis Agent
Driven by a fine-tuned **MobileNetV2** ONNX classifier. Evaluates mask presence and wearing compliance (Proper Mask, Unmasked, Nose Exposed, Mouth Exposed).

### 5. 🔎 Occlusion Agent
Calculates the exact visible proportion of the face across 4 anatomical zones (Forehead, Eyes, Nose, Mouth). Based on visibility ratio, it directs the Recognition Agent to deploy:
- **Full-Face Strategy** (Visibility $> 80\%$)
- **Upper-Face Virtual Masking Strategy** (Visibility $40\%–80\%$)
- **Inconclusive Review Mode** (Visibility $< 40\%$)

### 6. 🧠 Recognition Agent
Utilizes OpenCV's **SFace** deep feature extractor. Generates 128-dimensional hyperspherical embeddings and computes Cosine Similarity against enrolled templates. When face masks are detected, it dynamically masks out the lower face region ($y \ge 65$) of both stored templates and live query crops to align upper-face features (eyes, eyebrows, forehead).

### 7. 🛡️ Liveness / Anti-Spoof Agent
Protects against presentation attacks (photos, phone screens, pre-recorded video). Checks landmark micro-jitter, optical flow continuity, and texture variance over consecutive frames.

### 8. 🎥 Temporal Tracking Agent
Tracks bounding box trajectories and maintains a 5-frame sliding window of identification scores. Eliminates single-frame anomalies and stabilizes identity output (e.g., `Manish (94.2% stability)` across 5 frames).

### 9. 🧮 Fusion & Confidence Agent
Consolidates raw signals from Recognition, Quality, Liveness, Occlusion, and Temporal agents into a single calibrated final confidence metric.

### 10. 🚨 Security / Risk Agent
Enforces access directives (e.g. strict mask mandate, minimum confidence bar). Outputs actionable risk ratings:
- 🟢 `VERIFIED` — Access Granted
- 🟡 `REVIEW REQUIRED` — High Uncertainty / High Occlusion
- 🔴 `ACCESS DENIED` — Unregistered Face / Spoof Detected
- 🟠 `MASK VIOLATION` — Unmasked in Safety Mode

### 11. 📋 Audit & AI Explanation Agent
Generates exportable `.csv` security logs and synthesizes human-readable natural language diagnostic explanations (e.g. *"Verification inconclusive: High facial occlusion (58%) and low liveness confidence. Please adjust mask or move closer to camera"*).

---

## 🧠 Deep Learning ONNX Models

The system relies on three lightweight, high-speed ONNX models running on native OpenCV DNN CPU backends (automatically downloaded on first launch):

| Model | File Name | Architecture | Purpose |
|:---|:---|:---|:---|
| **YuNet** | `face_detection_yunet_2023mar.onnx` | MobileNet-based DNN | Sub-millisecond face detection & landmark localization |
| **SFace** | `face_recognition_sface_2021dec.onnx` | SphereFace / CosFace DNN | 128D face feature embedding & cosine similarity |
| **MobileNetV2** | `mask_detector.onnx` | MobileNetV2 ONNX | Binary mask presence & compliance classifier |

---

## 📦 Installation & Setup

### Prerequisites
- **Python 3.12 or higher** (Tested on Python 3.12 & 3.13 on Windows / Linux / macOS)

### 1. Clone Workspace

```powershell
git clone https://github.com/Manishsah098/Masked-Faced-Recognition.git
cd "Masked Face Recognition"
```

### 2. Set Up Virtual Environment

```powershell
python -m venv .venv
```

**Activate environment:**
- **PowerShell (Windows):** `.venv\Scripts\Activate.ps1`
- **CMD (Windows):** `.venv\Scripts\activate.bat`
- **Bash (macOS / Linux):** `source .venv/bin/activate`

### 3. Install Dependencies

**For Desktop App (Tkinter GUI):**
```bash
pip install opencv-python numpy pillow
```

**For Web Dashboard (Flask + Socket.IO):**
```bash
pip install opencv-python numpy pillow flask flask-socketio eventlet
```

> **Note:** Models automatically download to `models/` on first execution.

---

## 🚀 Running MFR-X

### Option A — Desktop Application (Tkinter Multi-Agent HUD)

```powershell
python main.py
```

### Option B — Web Dashboard (Flask + Socket.IO Telemetry)

```powershell
python app.py
```

Open your web browser and navigate to:
```
http://localhost:5000
```

---

## 🌐 Web API & Telemetry Reference

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/` | Serves real-time multi-agent Web UI |
| `GET` | `/api/state` | Returns live multi-agent detection state & agent breakdowns |
| `GET` | `/api/logs` | Returns last 100 system security logs |
| `GET` | `/api/directory` | Lists all enrolled biometric user profiles |
| `GET` | `/api/export_logs` | Downloads audit log as `system_audit_log.csv` |
| `POST` | `/api/register` | Initiates 5-frame biometric profile enrolment |
| `POST` | `/api/delete_user` | Permanently deletes user profile by name |
| `POST` | `/api/settings` | Updates cosine threshold, frame-skip interval, or safety mode |
| `POST` | `/api/wipe_db` | Clears local serialized biometric database |

**Socket.IO Telemetry Events:**
- `image` (Client $\rightarrow$ Server): Send Base64 webcam frame.
- `response` (Server $\rightarrow$ Client): Returns annotated frame with multi-agent telemetry JSON (`quality`, `occlusion`, `liveness`, `explanation`, `agent_breakdown`).

---

## 🗂 Workspace Architecture

```
Masked Face Recognition/
├── app.py                  # Flask + Socket.IO multi-agent web server
├── main.py                 # Tkinter multi-agent HUD desktop interface
├── db.json                 # Serialized 128D biometric template database
├── verify_pipeline.py      # Diagnostic smoke-test script
├── models/                 # Cached ONNX model binaries (auto-downloaded)
│   ├── face_detection_yunet_2023mar.onnx
│   ├── face_recognition_sface_2021dec.onnx
│   └── mask_detector.onnx
├── mfr/                    # Biometric Pipeline & Multi-Agent Package
│   ├── __init__.py
│   ├── detector.py         # YuNet Face Detection Agent wrapper
│   ├── recognizer.py       # SFace Recognition Agent wrapper
│   ├── mask_detector.py    # MobileNetV2 Mask Analysis Agent wrapper
│   ├── database.py         # Serialized database manager
│   └── utils.py            # Model Downloader & helper utilities
├── templates/
│   └── index.html          # Web UI template with Multi-Agent HUD
└── static/                 # CSS/JS assets for Web UI
```

---

## 📄 License & Attribution

Developed for research and hackathon demonstration.  
Built with **OpenCV**, **Flask**, **Socket.IO**, and **ONNX Runtime**.
