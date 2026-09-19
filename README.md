# MFR-X — Multi-Agent Real-Time Masked Face Recognition

**MFR-X (Multi-Agent Face Recognition eXperience)** is a CPU-optimized computer vision and biometric verification system designed for **real-time face recognition under facial occlusion**.

Instead of relying on a single monolithic recognition pipeline, MFR-X uses a coordinated network of specialized AI agents to analyze different aspects of a live camera feed, including:

* Face detection
* Image quality
* Mask detection
* Facial occlusion
* Face recognition
* Liveness and anti-spoofing
* Temporal consistency
* Confidence estimation
* Security and risk assessment
* Audit logging and AI-generated explanations

The system is designed to make recognition decisions more robust by combining multiple independent signals before producing a final verification result.

---

## Overview

Real-world face recognition systems often encounter challenging conditions such as:

* Face masks and partial facial occlusion
* Poor lighting
* Blurry or low-resolution images
* Different face orientations
* Presentation attacks using photographs or screens
* Temporary recognition errors in individual video frames

MFR-X addresses these challenges through a **multi-agent architecture** in which each agent performs a dedicated task and contributes its results to the overall verification pipeline.

### Core Workflow

```text
Live Camera Feed
       │
       ▼
Orchestrator Agent
       │
       ├── Face Detection Agent
       ├── Face Quality Agent
       └── Mask Analysis Agent
                │
                ▼
         Occlusion Agent
                │
        ┌───────┴────────┐
        ▼                ▼
 Recognition Agent   Liveness Agent
        │                │
        └───────┬────────┘
                ▼
       Temporal Tracking
                │
                ▼
     Fusion & Confidence
                │
                ▼
        Security / Risk
                │
        ┌───────┼────────┐
        ▼       ▼        ▼
     Verified  Review  Unknown
                │
                ▼
      Audit & Explanation
```

---

# System Architecture

MFR-X is organized into four major operational tiers.

```text
MFR-X
│
├── Orchestrator Agent
│
├── Perception Tier
│   ├── Face Detection Agent
│   ├── Face Quality Agent
│   ├── Mask Analysis Agent
│   └── Occlusion Agent
│
├── Biometric Tier
│   ├── Recognition Agent
│   ├── Liveness / Anti-Spoof Agent
│   └── Temporal Tracking Agent
│
├── Decision Tier
│   ├── Fusion & Confidence Agent
│   └── Security / Risk Agent
│
└── System Intelligence Tier
    └── Audit & AI Explanation Agent
```

---

# Multi-Agent Components

## 1. Orchestrator Agent

The **Orchestrator Agent** acts as the central execution controller for the MFR-X pipeline.

It evaluates the output of upstream agents and determines which downstream agents need to be executed.

For example, the system can avoid unnecessary recognition processing when:

* No face is detected
* Image quality is below the required threshold
* The detected face does not satisfy processing requirements

This conditional execution helps reduce unnecessary CPU computation.

---

## 2. Face Detection Agent

The Face Detection Agent uses **YuNet** for real-time face detection and facial landmark localization.

### Responsibilities

* Detect faces from live camera frames
* Locate bounding boxes
* Extract five facial landmarks
* Apply spatial filtering to reduce false detections

---

## 3. Face Quality Agent

Before performing biometric recognition, the Face Quality Agent evaluates whether the captured face is suitable for recognition.

### Quality Parameters

* Image sharpness
* Contrast
* Resolution
* Face orientation
* Pose / tilt

When the image does not satisfy the required quality conditions, the system can provide guidance such as:

> "Move closer / Adjust lighting"

This helps reduce unreliable recognition attempts.

---

## 4. Mask Analysis Agent

The Mask Analysis Agent uses a **MobileNetV2-based ONNX classifier** to analyze mask presence and compliance.

It classifies facial mask conditions such as:

* Proper Mask
* Unmasked
* Nose Exposed
* Mouth Exposed

---

## 5. Occlusion Agent

The Occlusion Agent estimates the visible proportion of the face across four anatomical regions:

* Forehead
* Eyes
* Nose
* Mouth

Based on the calculated visibility ratio, the system selects an appropriate recognition strategy.

| Visibility | Recognition Strategy       |
| ---------- | -------------------------- |
| > 80%      | Full-Face Recognition      |
| 40–80%     | Upper-Face Virtual Masking |
| < 40%      | Inconclusive Review        |

This allows the recognition pipeline to adapt to different levels of facial occlusion.

---

## 6. Recognition Agent

The Recognition Agent uses **OpenCV SFace** for deep facial feature extraction and identity matching.

The system generates **128-dimensional face embeddings** and compares them with enrolled biometric templates using cosine similarity.

For masked faces, the recognition pipeline can focus on visible upper-face features such as:

* Eyes
* Eyebrows
* Forehead

This adaptive approach allows the system to continue recognition when part of the face is covered.

---

## 7. Liveness / Anti-Spoof Agent

The Liveness Agent is responsible for detecting potential presentation attacks.

It evaluates temporal visual information using signals such as:

* Landmark micro-movement
* Optical-flow continuity
* Texture variation
* Consecutive-frame analysis

The goal is to distinguish a live subject from potential attacks involving:

* Printed photographs
* Phone screens
* Pre-recorded video

---

## 8. Temporal Tracking Agent

The Temporal Tracking Agent improves recognition stability across consecutive frames.

It maintains a **5-frame sliding window** of identification results and uses temporal information to reduce the effect of isolated frame-level errors.

Example:

```text
Frame 1 → Manish
Frame 2 → Manish
Frame 3 → Manish
Frame 4 → Manish
Frame 5 → Manish

Result → Stable Identity
```

---

## 9. Fusion & Confidence Agent

The Fusion & Confidence Agent combines signals from multiple agents, including:

* Recognition
* Image quality
* Liveness
* Occlusion
* Temporal tracking

These signals are consolidated into a final confidence value used by the decision pipeline.

---

## 10. Security / Risk Agent

The Security / Risk Agent applies configured access policies and security rules.

Possible system states include:

| Status            | Description                                                          |
| ----------------- | -------------------------------------------------------------------- |
| `VERIFIED`        | Identity successfully verified                                       |
| `REVIEW REQUIRED` | Recognition confidence or image conditions require additional review |
| `ACCESS DENIED`   | Unknown identity or detected spoofing condition                      |
| `MASK VIOLATION`  | Mask requirement not satisfied in safety mode                        |

---

## 11. Audit & AI Explanation Agent

The Audit & AI Explanation Agent provides system transparency and diagnostic information.

It generates:

* CSV security logs
* Verification events
* Agent-level diagnostic information
* Human-readable explanations

Example:

```text
Verification inconclusive:
High facial occlusion and low liveness confidence.
Please adjust the mask or move closer to the camera.
```

---

# Deep Learning Models

MFR-X uses lightweight ONNX models designed for efficient CPU-based inference.

| Model           | Architecture                   | Purpose                                       |
| --------------- | ------------------------------ | --------------------------------------------- |
| **YuNet**       | MobileNet-based DNN            | Face detection and landmark localization      |
| **SFace**       | SphereFace / CosFace-based DNN | Face feature extraction and identity matching |
| **MobileNetV2** | MobileNetV2 ONNX               | Mask detection and compliance classification  |

### Model Files

```text
models/
├── face_detection_yunet_2023mar.onnx
├── face_recognition_sface_2021dec.onnx
└── mask_detector.onnx
```

The models are automatically downloaded on first execution.

---

# Technology Stack

### Computer Vision & AI

* Python
* OpenCV
* OpenCV DNN
* YuNet
* SFace
* MobileNetV2
* ONNX

### Backend

* Flask
* Flask-SocketIO
* Eventlet

### Frontend / Desktop

* HTML
* CSS
* JavaScript
* Tkinter
* Socket.IO

### Data & Logging

* JSON-based biometric database
* CSV audit logs

---

# Installation

## Prerequisites

* Python **3.12 or higher**
* Webcam / camera
* Windows, Linux, or macOS

---

## 1. Clone the Repository

```bash
git clone https://github.com/Manishsah098/Masked-Faced-Recognition.git
cd "Masked Face Recognition"
```

---

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### Windows — PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

### Windows — CMD

```cmd
.venv\Scripts\activate.bat
```

### macOS / Linux

```bash
source .venv/bin/activate
```

---

## 3. Install Dependencies

### Desktop Application

```bash
pip install opencv-python numpy pillow
```

### Web Dashboard

```bash
pip install opencv-python numpy pillow flask flask-socketio eventlet
```

The required ONNX models are downloaded automatically when the application is executed for the first time.

---

# Running the Application

MFR-X provides two execution modes.

## Desktop Application

Launch the Tkinter-based multi-agent interface:

```bash
python main.py
```

---

## Web Dashboard

Start the Flask + Socket.IO server:

```bash
python app.py
```

Then open:

```text
http://localhost:5000
```

The web interface provides real-time system telemetry and multi-agent processing information.

---

# Web API

MFR-X exposes a REST API for system monitoring and management.

| Method | Endpoint           | Description                                 |
| ------ | ------------------ | ------------------------------------------- |
| `GET`  | `/`                | Serves the real-time web interface          |
| `GET`  | `/api/state`       | Returns current multi-agent detection state |
| `GET`  | `/api/logs`        | Returns recent security logs                |
| `GET`  | `/api/directory`   | Lists enrolled biometric profiles           |
| `GET`  | `/api/export_logs` | Downloads the audit log                     |
| `POST` | `/api/register`    | Registers a biometric profile               |
| `POST` | `/api/delete_user` | Deletes a biometric profile                 |
| `POST` | `/api/settings`    | Updates system configuration                |
| `POST` | `/api/wipe_db`     | Clears the local biometric database         |

---

# Real-Time Telemetry

The system uses **Socket.IO** for real-time communication between the web client and server.

### Client → Server

```text
image
```

Sends a Base64-encoded camera frame.

### Server → Client

```text
response
```

Returns the processed frame along with multi-agent telemetry such as:

```text
quality
occlusion
liveness
explanation
agent_breakdown
```

---

# Project Structure

```text
Masked Face Recognition/
│
├── app.py
├── main.py
├── db.json
├── verify_pipeline.py
│
├── models/
│   ├── face_detection_yunet_2023mar.onnx
│   ├── face_recognition_sface_2021dec.onnx
│   └── mask_detector.onnx
│
├── mfr/
│   ├── __init__.py
│   ├── detector.py
│   ├── recognizer.py
│   ├── mask_detector.py
│   ├── database.py
│   └── utils.py
│
├── templates/
│   └── index.html
│
└── static/
    └── CSS / JavaScript assets
```

### Core Modules

| File                 | Responsibility                         |
| -------------------- | -------------------------------------- |
| `app.py`             | Flask + Socket.IO web server           |
| `main.py`            | Tkinter desktop interface              |
| `detector.py`        | YuNet face detection                   |
| `recognizer.py`      | SFace recognition                      |
| `mask_detector.py`   | Mask classification                    |
| `database.py`        | Biometric profile management           |
| `utils.py`           | Model downloading and helper functions |
| `verify_pipeline.py` | Pipeline diagnostic testing            |

---

# Key Capabilities

* Real-time face detection
* Mask-aware face recognition
* Facial occlusion analysis
* Adaptive recognition strategies
* CPU-optimized inference
* Liveness / anti-spoof analysis
* Temporal identity stabilization
* Multi-signal confidence fusion
* Configurable security policies
* Real-time web telemetry
* Desktop monitoring interface
* CSV audit logging
* AI-generated diagnostic explanations

---

# Example Decision Pipeline

```text
Camera Frame
     │
     ▼
Face Detection
     │
     ▼
Quality Assessment
     │
     ├── Poor Quality ──► User Guidance
     │
     ▼
Mask Analysis
     │
     ▼
Occlusion Analysis
     │
     ├── High Visibility ──► Full-Face Recognition
     │
     ├── Partial Visibility ──► Upper-Face Recognition
     │
     └── Very Low Visibility ──► Review Required
     │
     ▼
Liveness Verification
     │
     ▼
Temporal Tracking
     │
     ▼
Confidence Fusion
     │
     ▼
Security / Risk Evaluation
     │
     ├── VERIFIED
     ├── REVIEW REQUIRED
     ├── ACCESS DENIED
     └── MASK VIOLATION
     │
     ▼
Audit & Explanation
```

---

# Performance-Oriented Design

MFR-X is designed with CPU efficiency in mind.

The multi-agent architecture allows the system to make conditional processing decisions rather than executing every component for every frame.

For example:

```text
No Face
   ↓
Stop Processing

Poor Quality
   ↓
Request Better Capture

Valid Face
   ↓
Continue Recognition Pipeline
```

This approach helps reduce unnecessary computation during real-time operation.

---

# Use Cases

MFR-X can serve as a research and demonstration platform for applications involving:

* Access-control research
* Mask-aware biometric verification
* Smart surveillance research
* Computer vision experimentation
* Anti-spoofing research
* Real-time biometric systems
* Multi-agent AI architectures
* Hackathon demonstrations
* Edge / CPU-based AI applications

---

# Research & Hackathon Project

MFR-X was developed as a **research and hackathon demonstration project** exploring the combination of:

> **Computer Vision + Biometrics + Multi-Agent AI + Real-Time Processing**

The project demonstrates how a complex biometric pipeline can be decomposed into specialized agents that independently analyze different aspects of the input before contributing to a final system decision.

---

# Privacy & Security Considerations

Because this system processes biometric information, deployments should consider:

* User consent
* Secure storage of biometric templates
* Access control
* Data retention policies
* Encryption
* Appropriate legal and regulatory requirements
* Protection of audit logs
* Secure deletion of biometric data

This repository is intended primarily for **research, development, and demonstration purposes**.

---

# License & Attribution

Developed for research and hackathon demonstration.

Built with:

* **OpenCV**
* **Flask**
* **Flask-SocketIO**
* **ONNX**
* **Python**

---

# Author

**Manish Sah**

Computer Science & Engineering

GitHub: `Manishsah098`

---

## Project Summary

**MFR-X is a real-time, CPU-optimized, multi-agent biometric verification system that combines face detection, mask analysis, occlusion assessment, recognition, liveness detection, temporal tracking, confidence fusion, and security evaluation into a unified pipeline.**

The architecture demonstrates a modular approach to building robust computer vision systems where specialized AI agents collaborate to produce a final, explainable verification result.
