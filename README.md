# Masked Face Recognition System

A real-time, premium computer vision application for **Masked Face Recognition (MFR)**. It runs entirely on the CPU using OpenCV's native Deep Neural Network (DNN) classes and pre-trained ONNX models, providing high-speed inference without heavy frameworks like TensorFlow or PyTorch.

---

## Key Features

*   **Real-Time Biometric Scanner**: Detects faces, maps facial landmarks, and identifies users on a live webcam feed.
*   **ONNX Mask Detection Classifier**: Automatically classifies whether a face is `Masked` or `Unmasked`.
*   **Adaptive Upper-Face Matching**: Dynamically switches matching models based on mask presence:
    *   **Unmasked Users**: Compares full-face embeddings (highest accuracy).
    *   **Masked Users**: Blackouts the lower face region (`y >= 65`) of both the database templates and live query crops, comparing only the visible upper-face geometry (eyes, eyebrows, forehead) to bypass mask occlusion.
*   **Profile Enrolment Workflow**: Automatically captures 5 face samples and averages their embeddings to minimize noise and register a secure template.
*   **System Audit Console**: Logs biometric scans, safety compliance audits, and access denials with precise timestamps.
*   **Interactive Calibration Settings**: Adjust matching sensitivity (SFace Cosine Threshold) and frame skips for CPU optimization.

---

## Technical Architecture & Models

The application automatically downloads and runs three optimized deep learning models in ONNX format:
1.  **YuNet** (`face_detection_yunet_2023mar.onnx`): Sub-millisecond face and 5-point landmark detector.
2.  **SFace** (`face_recognition_sface_2021dec.onnx`): Sigmoid-constrained hypersphere face recognition model.
3.  **MobileNetV2 Mask Detector** (`mask_detector.onnx`): Binary classifier trained to identify face mask compliance.

---

## Installation & Setup

Ensure you have **Python 3.12 or higher** installed. (Tested up to Python 3.13.7 on Windows).

1.  **Clone or Open the Workspace** in your terminal.
2.  **Create a Virtual Environment**:
    ```powershell
    python -m venv .venv
    ```
3.  **Activate the Environment**:
    *   **Windows (PowerShell)**:
        ```powershell
        .venv\Scripts\Activate.ps1
        ```
    *   **Windows (CMD)**:
        ```cmd
        .venv\Scripts\activate.bat
        ```
    *   **macOS / Linux**:
        ```bash
        source .venv/bin/activate
        ```
4.  **Install Required Dependencies**:
    ```bash
    pip install opencv-python numpy pillow
    ```

---

## Operating Instructions

Run the application:
```powershell
python main.py
```
*Note: On the first launch, the application will take a moment to download the required ONNX model files to a local `models/` directory.*

### 1. Registering a New User Profile
1.  Click the **Register User** tab in the left sidebar.
2.  Type the person's name in the **Full Name** entry box.
3.  Instruct the user to stand in front of the camera and **remove their face mask**. *(The system enforces unmasked faces during enrolment to capture a clean full-face baseline).*
4.  Click **Start Profile Acquisition**.
5.  Keep still. The system will track your landmarks, capture 5 distinct biometric frames, extract embeddings, and save the averaged template. A confirmation status will display once complete.

### 2. Live Scanning & Identification
1.  Click the **Live Camera** tab in the sidebar.
2.  Stand in front of the camera. The scanner will run and draw a colored HUD box around your face:
    *   **Green Bounding Box**: You are identified as a registered user (Unmasked).
    *   **Blue Bounding Box**: You are identified as a registered user (Masked). The system has successfully switched to **Upper Face Matching** mode to verify your identity.
    *   **Red Bounding Box**: The face is unregistered and marked as **Unknown** (Access Denied).

### 3. Safety Compliance Mode (Mandate Masks)
1.  In the **Live Camera** tab, look at the **Access Directives** panel on the right.
2.  Toggle **Mandate Face Mask** ON.
3.  If a registered user stands in front of the camera without a mask, the system will trigger a warning: the box turns **Orange** and registers a safety violation in the log console.

### 4. Viewing Logs & Directory
*   **User Directory**: Go to this tab to see all enrolled names and templates. You can remove individual profiles permanently by clicking the **Delete** button.
*   **System Logs**: View the live log of all system activities, network loadings, scan matches, and compliance violations.

### 5. Calibrating Sensitivity
If you experience false positives (misidentifying strangers) or false negatives (failing to recognize yourself), go to the **System Settings** tab:
*   **Cosine Threshold Slider**: Adjust the match boundary (default `0.363`). Higher values require more precise facial matches. Lower values are more lenient.
*   **Frame Skip Slider**: Increase the skip interval (e.g. run inference every 3 or 5 frames) to reduce CPU load on older systems.

---

## Workspace Structure

```
├── main.py                 # Tkinter graphical interface and pipeline driver
├── db.json                 # Local serialized biometric database
├── models/                 # Cached ONNX model files (auto-generated)
│   ├── yunet.onnx
│   ├── sface.onnx
│   └── mask_detector.onnx
├── mfr/                    # Core biometric pipeline package
│   ├── __init__.py
│   ├── detector.py         # YuNet face detector wrapper (internally downscaled)
│   ├── recognizer.py       # SFace aligner and upper-face masking logic
│   ├── mask_detector.py    # ONNX mask compliance classifier
│   ├── database.py         # Ser/Des database manager
│   └── utils.py            # Biometric model network downloader
└── verify_pipeline.py      # Diagnostic check script
```
