import tkinter as tk
from tkinter import messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import sys
import time
from datetime import datetime
import threading

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import mfr

class MFRSystemApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Masked Face Recognition System")
        self.root.geometry("1280x750")
        self.root.minsize(1024, 650)
        
        # Modern Dark Palette Design
        self.bg_main = "#0d1117"        # Dark charcoal blue
        self.bg_panel = "#161b22"       # Deeper panel blue
        self.bg_card = "#21262d"        # Slate card color
        self.border_color = "#30363d"   # Soft slate outline
        self.text_primary = "#ffffff"   # High contrast text
        self.text_secondary = "#8b949e" # Muted text
        self.accent_blue = "#58a6ff"    # Glowing cyan-blue (Masked / Active)
        self.accent_green = "#2ea043"   # Secure Green (Unmasked matched)
        self.accent_red = "#f85149"     # Warning Red (Unknown)
        self.accent_orange = "#d29922"  # Alert Orange (Safety Warning)
        
        self.root.configure(bg=self.bg_main)
        
        # Pipeline State
        self.is_camera_active = False
        self.models_ready = False
        self.cap = None
        self.orchestrator = None  # MFR-X Multi-Agent Orchestrator
        self.db = mfr.Database()
        self.log_db = mfr.DetectionLog("detection_log.db")
        
        self.cached_results = []
        self.latest_telemetry = {}  # Full multi-agent telemetry
        self.frame_count = 0
        self.detection_interval = 3      # Run inference every 3 frames for smooth UI
        self.strict_mask_mode = tk.BooleanVar(value=False)
        self.active_tab = "camera"
        
        # User Registration Flow State
        self.registering = False
        self.reg_name = ""
        self.reg_frames = []
        self.reg_status_text = tk.StringVar(value="Stand by...")
        self.reg_progress = tk.DoubleVar(value=0.0)
        
        # Logging State
        self.logs = []
        self.seen_detections = {}        # Track recently seen users to prevent spam
        
        # Setup UI Theme & show splash landing page
        self.setup_styles()
        self.show_splash_screen()
        
        # Initialize models immediately in background so they are ready when user clicks Start
        threading.Thread(target=self.initialize_models, daemon=True).start()

    def setup_styles(self):
        """Sets up custom ttk styling for a clean flat design."""
        style = ttk.Style()
        style.theme_use('default')
        
        # Configure Notebook/Tabs
        style.configure('TNotebook', background=self.bg_main, borderwidth=0)
        style.configure('TNotebook.Tab', background=self.bg_panel, foreground=self.text_secondary, borderwidth=0, padding=(15, 8))
        style.map('TNotebook.Tab', background=[('selected', self.bg_main)], foreground=[('selected', self.text_primary)])
        
        # Configure Progressbar
        style.configure("TProgressbar", thickness=8, troughcolor=self.bg_panel, background=self.accent_blue)
        style.map("TProgressbar", background=[('active', self.accent_blue)])

    def initialize_models(self):
        """Downloads (if necessary) and loads the MFR-X Multi-Agent Orchestrator."""
        self.log_event("SYSTEM: Initializing MFR-X Multi-Agent Orchestrator...")
        try:
            self.orchestrator = mfr.BiometricOrchestrator(models_dir="models", db_path="db.json")
            self.orchestrator.frame_interval = 1  # Desktop handles its own interval via detection_interval
            self.db = self.orchestrator.recognition_agent.db
            
            self.log_event("SYSTEM: 10 AI Specialized Agents loaded successfully.")
            self.reg_status_text.set("System Ready")
            self.models_ready = True
        except Exception as e:
            self.log_event(f"ERROR: Model initialization failed: {e}")
            self.root.after(0, lambda: messagebox.showerror("Initialization Error", f"Failed to initialize models:\n{e}"))

    def show_splash_screen(self):
        """Creates and displays the full-window premium landing/splash screen."""
        self.splash_frame = tk.Frame(self.root, bg=self.bg_main)
        self.splash_frame.place(relx=0, rely=0, relwidth=1, relheight=1)
        
        # === CENTER CONTAINER ===
        center = tk.Frame(self.splash_frame, bg=self.bg_main)
        center.place(relx=0.5, rely=0.48, anchor="center")
        
        # Brand accent icon
        tk.Label(center, text="\u25c6", font=("Segoe UI", 22),
                 fg=self.accent_blue, bg=self.bg_main).pack()
        
        # Main brand title
        tk.Label(center, text="Face Analyst",
                 font=("Segoe UI", 56, "bold"),
                 fg=self.text_primary, bg=self.bg_main).pack(pady=(4, 0))
        
        # Glowing blue underline bar
        tk.Frame(center, bg=self.accent_blue, height=3, width=360).pack(pady=(8, 0))
        
        # Tagline
        tk.Label(center,
                 text="Advanced Biometric Identity \u2022 Masked Face Recognition \u2022 Security Compliance",
                 font=("Segoe UI", 10),
                 fg=self.text_secondary, bg=self.bg_main).pack(pady=(14, 48))
        
        # === FEATURE CARDS ROW ===
        cards_row = tk.Frame(center, bg=self.bg_main)
        cards_row.pack(pady=(0, 48))
        
        features = [
            ("Real-Time\nDetection",  "YuNet DNN engine\nfor fast face detection"),
            ("Mask\nRecognition",     "MobileNetV2 ONNX\ncompliance classifier"),
            ("Profile\nEnrolment",    "5-sample biometric\ntemplate averaging"),
            ("Security\nAudit Log",   "Timestamped activity\n& violation logging"),
        ]
        
        for title, desc in features:
            card = tk.Frame(cards_row, bg=self.bg_panel,
                            highlightbackground=self.border_color,
                            highlightthickness=1,
                            padx=22, pady=20,
                            width=188, height=118)
            card.pack(side="left", padx=9)
            card.pack_propagate(False)
            
            tk.Label(card, text=title, font=("Segoe UI", 10, "bold"),
                     fg=self.accent_blue, bg=self.bg_panel, justify="left").pack(anchor="w", pady=(0, 6))
            tk.Label(card, text=desc, font=("Segoe UI", 8),
                     fg=self.text_secondary, bg=self.bg_panel, justify="left").pack(anchor="w")
            
            # Hover glow on card border
            def _enter(e, c=card): c.config(highlightbackground=self.accent_blue)
            def _leave(e, c=card): c.config(highlightbackground=self.border_color)
            card.bind("<Enter>", _enter)
            card.bind("<Leave>", _leave)
            for child in card.winfo_children():
                child.bind("<Enter>", _enter)
                child.bind("<Leave>", _leave)
        
        # === LAUNCH BUTTON ===
        self.start_btn = tk.Button(
            center,
            text="  \u25ba   Launch System  ",
            font=("Segoe UI", 13, "bold"),
            bg=self.accent_blue,
            fg="#ffffff",
            activebackground="#1f6feb",
            activeforeground="#ffffff",
            bd=0, relief="flat",
            cursor="hand2",
            command=self.launch_main_app,
            padx=40, pady=15
        )
        self.start_btn.pack()
        self.start_btn.bind("<Enter>", lambda e: self.start_btn.config(bg="#1f6feb"))
        self.start_btn.bind("<Leave>", lambda e: self.start_btn.config(bg=self.accent_blue))
        
        # === FOOTER ===
        tk.Label(self.splash_frame,
                 text="Powered by OpenCV DNN  \u2022  ONNX Runtime  \u2022  YuNet  \u2022  SFace  \u2022  MobileNetV2",
                 font=("Segoe UI", 8),
                 fg=self.text_secondary,
                 bg=self.bg_main).place(relx=0.5, rely=0.95, anchor="center")

    def launch_main_app(self):
        """Destroys the splash screen and transitions into the main application UI."""
        self.splash_frame.destroy()
        self.setup_ui_layout()
        # switch_to_camera packs the camera tab frame into the layout AND starts the webcam
        self.root.after(200, self.switch_to_camera)

    def log_event(self, message):
        """Appends a timestamped log to the list and updates the logs view if active."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_str = f"[{timestamp}] {message}"
        self.logs.append(log_str)
        print(log_str)
        
        # Schedule GUI update for log panel
        self.root.after(0, self.update_logs_ui)

    def setup_ui_layout(self):
        """Creates the main window sidebar and right content panel."""
        # Main Container
        self.main_container = tk.Frame(self.root, bg=self.bg_main)
        self.main_container.pack(fill="both", expand=True)
        
        # 1. Left Sidebar Frame
        self.sidebar = tk.Frame(self.main_container, bg=self.bg_panel, width=240, bd=0)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        
        # Sidebar Logo Section
        logo_frame = tk.Frame(self.sidebar, bg=self.bg_panel, pady=25)
        logo_frame.pack(fill="x")
        
        logo_lbl = tk.Label(logo_frame, text="Face Analyst", font=("Segoe UI", 18, "bold"), fg=self.accent_blue, bg=self.bg_panel)
        logo_lbl.pack()
        sub_logo_lbl = tk.Label(logo_frame, text="MASKED FACE RECOGNITION", font=("Segoe UI", 8, "bold"), fg=self.text_secondary, bg=self.bg_panel)
        sub_logo_lbl.pack(pady=2)
        
        divider = tk.Frame(self.sidebar, bg=self.border_color, height=1)
        divider.pack(fill="x", padx=15, pady=10)
        
        # Navigation Buttons
        self.nav_buttons = {}
        navs = [
            ("camera", " Live Camera", self.switch_to_camera),
            ("register", " Register User", self.switch_to_register),
            ("directory", " User Directory", self.switch_to_directory),
            ("logs", " System Logs", self.switch_to_logs),
            ("settings", " System Settings", self.switch_to_settings)
        ]
        
        for key, text, command in navs:
            btn = tk.Button(
                self.sidebar,
                text=text,
                font=("Segoe UI", 11),
                anchor="w",
                padx=20,
                bg=self.bg_panel,
                fg=self.text_secondary,
                activebackground=self.bg_card,
                activeforeground=self.text_primary,
                bd=0,
                relief="flat",
                command=command
            )
            btn.pack(fill="x", pady=2, padx=10)
            self.style_button(btn, self.bg_card, self.text_primary, self.bg_panel, self.text_secondary)
            self.nav_buttons[key] = btn
            
        # Sidebar Footer
        self.system_status_var = tk.StringVar(value="Initializing...")
        footer = tk.Frame(self.sidebar, bg=self.bg_panel, pady=15)
        footer.pack(side="bottom", fill="x")
        
        status_dot = tk.Label(footer, text="●", fg=self.accent_orange, bg=self.bg_panel, font=("Segoe UI", 10))
        status_dot.pack(side="left", padx=(20, 5))
        self.status_dot_lbl = status_dot
        
        status_lbl = tk.Label(footer, textvariable=self.system_status_var, font=("Segoe UI", 9), fg=self.text_secondary, bg=self.bg_panel)
        status_lbl.pack(side="left")
        
        # 2. Right Content Pane
        self.content_pane = tk.Frame(self.main_container, bg=self.bg_main, padx=25, pady=25)
        self.content_pane.pack(side="right", fill="both", expand=True)
        
        # Initialize Frames for tabs
        self.create_camera_frame()
        self.create_register_frame()
        self.create_directory_frame()
        self.create_logs_frame()
        self.create_settings_frame()
        
        # Set Active Tab styling
        self.set_active_nav_styling("camera")

    def style_button(self, btn, active_bg, active_fg, normal_bg, normal_fg):
        """Attaches clean hover event listeners to buttons."""
        btn.config(cursor="hand2")
        btn.bind("<Enter>", lambda e: btn.config(bg=active_bg, fg=active_fg) if self.active_tab != btn_key_from_val(btn) else None)
        btn.bind("<Leave>", lambda e: btn.config(bg=normal_bg, fg=normal_fg) if self.active_tab != btn_key_from_val(btn) else None)
        
        def btn_key_from_val(b):
            for k, v in self.nav_buttons.items():
                if v == b:
                    return k
            return ""

    def set_active_nav_styling(self, active_key):
        """Highlights the active tab navigation button on the sidebar."""
        self.active_tab = active_key
        for key, btn in self.nav_buttons.items():
            if key == active_key:
                btn.config(bg=self.accent_blue, fg=self.text_primary, font=("Segoe UI", 11, "bold"))
            else:
                btn.config(bg=self.bg_panel, fg=self.text_secondary, font=("Segoe UI", 11))

    # --- TAB SWITCHING LOGIC ---
    def switch_to_camera(self):
        self.hide_all_tabs()
        self.set_active_nav_styling("camera")
        self.camera_tab_frame.pack(fill="both", expand=True)
        self.start_camera()

    def switch_to_register(self):
        self.hide_all_tabs()
        self.set_active_nav_styling("register")
        self.register_tab_frame.pack(fill="both", expand=True)
        self.start_camera()

    def switch_to_directory(self):
        self.hide_all_tabs()
        self.set_active_nav_styling("directory")
        self.directory_tab_frame.pack(fill="both", expand=True)
        self.stop_camera()
        self.refresh_directory_table()

    def switch_to_logs(self):
        self.hide_all_tabs()
        self.set_active_nav_styling("logs")
        self.logs_tab_frame.pack(fill="both", expand=True)
        self.stop_camera()
        self.update_logs_ui()

    def switch_to_settings(self):
        self.hide_all_tabs()
        self.set_active_nav_styling("settings")
        self.settings_tab_frame.pack(fill="both", expand=True)
        self.stop_camera()

    def hide_all_tabs(self):
        """Hides pack configurations of all tabs."""
        self.camera_tab_frame.pack_forget()
        self.register_tab_frame.pack_forget()
        self.directory_tab_frame.pack_forget()
        self.logs_tab_frame.pack_forget()
        self.settings_tab_frame.pack_forget()

    # --- WEBCAM STREAMING ENGINE ---
    def start_camera(self):
        """Starts the video capture thread if not already running."""
        if self.is_camera_active:
            return
        
        self.is_camera_active = True
        self.cap = cv2.VideoCapture(0)
        
        # Test if camera was opened successfully
        if not self.cap.isOpened():
            self.log_event("ERROR: Failed to open webcam. Retrying with index 1...")
            self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                self.log_event("ERROR: No webcam detected.")
                self.is_camera_active = False
                self.system_status_var.set("Camera Error")
                self.status_dot_lbl.config(fg=self.accent_red)
                return
                
        self.system_status_var.set("Camera Active")
        self.status_dot_lbl.config(fg=self.accent_green)
        
        # Trigger the frame reading loop
        self.update_video_frame()

    def stop_camera(self):
        """Releases the camera and shuts down the update loop."""
        self.is_camera_active = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.system_status_var.set("System Standby")
        self.status_dot_lbl.config(fg=self.accent_blue)
        
        # Draw placeholder screen on active canvases
        self.draw_camera_placeholder(self.cam_canvas, "Camera Suspended")
        self.draw_camera_placeholder(self.reg_cam_canvas, "Camera Suspended")

    def draw_camera_placeholder(self, canvas, text):
        """Clears canvas and draws a custom dark placeholder."""
        canvas.delete("all")
        w = int(canvas.cget("width"))
        h = int(canvas.cget("height"))
        canvas.create_rectangle(0, 0, w, h, fill=self.bg_panel, outline=self.border_color)
        canvas.create_text(w//2, h//2, text=text, font=("Segoe UI", 12), fill=self.text_secondary)

    def update_video_frame(self):
        """The main real-time loop that fetches, processes, and overlays webcam frames."""
        if not self.is_camera_active or self.cap is None:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.log_event("WARNING: Empty frame received from webcam.")
            self.root.after(30, self.update_video_frame)
            return
            
        # Flip frame horizontally for natural mirror alignment
        frame = cv2.flip(frame, 1)
        self.frame_count += 1
        
        # Core Pipeline: MFR-X Multi-Agent Inference
        if self.orchestrator is not None:
            if self.frame_count % self.detection_interval == 0:
                self.orchestrator.set_strict_mode(self.strict_mask_mode.get())
                annotated_frame, telemetry = self.orchestrator.process_frame(frame)
                self.latest_telemetry = telemetry
                
                new_results = []
                if telemetry['detected']:
                    agents = telemetry['agents']
                    det = agents['detection']
                    mask_data = agents['mask']
                    rec_data = agents['recognition']
                    sec_data = agents['security']
                    
                    # Translate multi-agent telemetry back to legacy cached_results format
                    # for HUD drawing and info panel compatibility
                    faces = self.orchestrator.detection_agent.process(frame)
                    for face_p in faces:
                        mask_label = mask_data['mask_status']
                        mask_conf = mask_data['mask_confidence'] / 100.0
                        
                        candidate = sec_data.get('candidate', 'Unknown')
                        score = rec_data.get('similarity_score', 0.0)
                        
                        new_results.append({
                            'box': face_p['box'],
                            'landmarks': face_p['landmarks'],
                            'mask_status': (mask_label, mask_conf),
                            'identity': (candidate, score),
                            'raw': face_p['raw']
                        })
                        
                        self.track_detection_events(candidate, mask_label, score)
                
                self.cached_results = new_results
                
            # If in Registration Mode, save captured frames
            if self.registering and len(self.cached_results) == 1:
                # Only register if exactly one face is detected
                face_data = self.cached_results[0]
                mask_lbl, _ = face_data['mask_status']
                
                if mask_lbl == "Masked":
                    self.reg_status_text.set("ALERT: Remove mask to register!")
                    self.reg_progress.set(0.0)
                else:
                    try:
                        recognizer = self.orchestrator.recognition_agent.recognizer
                        aligned_face = recognizer.align_crop(frame, face_data['raw'])
                        self.reg_frames.append(aligned_face)
                        progress = len(self.reg_frames) / 5.0 * 100.0
                        self.reg_progress.set(progress)
                        self.reg_status_text.set(f"Capturing biometric template {len(self.reg_frames)}/5...")
                        
                        if len(self.reg_frames) == 5:
                            self.registering = False
                            threading.Thread(target=self.finalize_registration, daemon=True).start()
                    except Exception as e:
                        print(f"Error during registration capture: {e}")
            elif self.registering and len(self.cached_results) == 0:
                self.reg_status_text.set("Please align your face inside the frame.")
            elif self.registering and len(self.cached_results) > 1:
                self.reg_status_text.set("Multiple faces detected! Stand alone to register.")
                
        # Draw Overlays on Display Frame
        annotated_frame = frame.copy()
        self.draw_premium_hud(annotated_frame)
        
        # Convert BGR (OpenCV) to RGB (PIL) for Canvas rendering
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        # Route output to current active tab's canvas
        if self.active_tab == "camera":
            self.render_to_canvas(self.cam_canvas, annotated_frame_rgb)
        elif self.active_tab == "register":
            self.render_to_canvas(self.reg_cam_canvas, annotated_frame_rgb)
            
        # Loop iteration
        self.root.after(20, self.update_video_frame)

    def track_detection_events(self, name, mask_status, score):
        """Intelligently handles event logging for new detections."""
        current_time = time.time()
        
        # Expire older trackings
        for k in list(self.seen_detections.keys()):
            if current_time - self.seen_detections[k] > 10.0:  # 10s timeout
                del self.seen_detections[k]
                
        # Log if user is new or hasn't been logged in last 10s
        if name != "Unknown":
            if name not in self.seen_detections:
                self.seen_detections[name] = current_time
                status_msg = f"Masked (Score: {score:.2f})" if mask_status == "Masked" else "Unmasked"
                self.log_event(f"DETECTED: {name} - {status_msg}")
                self.log_db.log_detection(name, mask_status, score)
                
                # Check for strict safety warnings
                if self.strict_mask_mode.get() and mask_status == "Unmasked":
                    self.log_event(f"VIOLATION: {name} is unmasked in safety zone!")
        else:
            if "Unknown" not in self.seen_detections or current_time - self.seen_detections["Unknown"] > 6.0:
                self.seen_detections["Unknown"] = current_time
                self.log_event("ALERT: Unregistered user detected!")

    def render_to_canvas(self, canvas, rgb_img):
        """Resizes the RGB image and displays it on the target Canvas."""
        h, w = rgb_img.shape[:2]
        c_w = int(canvas.cget("width"))
        c_h = int(canvas.cget("height"))
        
        # Calculate aspect ratio
        scale = min(c_w/w, c_h/h)
        n_w = int(w * scale)
        n_h = int(h * scale)
        
        resized = cv2.resize(rgb_img, (n_w, n_h))
        pil_img = Image.fromarray(resized)
        tk_img = ImageTk.PhotoImage(image=pil_img)
        
        canvas.delete("all")
        # Center image in Canvas
        canvas.create_image(c_w//2, c_h//2, anchor="center", image=tk_img)
        canvas.image = tk_img # Maintain reference

    # --- ADVANCED HUD RENDERER (OpenCV overlays) ---
    def draw_premium_hud(self, frame):
        """Draws glowing brackets, landmarks, and information cards on the camera stream."""
        for res in self.cached_results:
            box = res['box']
            landmarks = res['landmarks']
            mask_lbl, mask_conf = res['mask_status']
            name, score = res['identity']
            
            x, y, w, h = box
            
            # 1. Set Color Theme based on state
            if name == "Unknown":
                color_theme = (73, 81, 248)       # Red BGR
                status_text = "ACCESS DENIED"
            else:
                if mask_lbl == "Masked":
                    color_theme = (255, 166, 88)   # Blue BGR
                    status_text = f"SECURE ({name})"
                else:
                    if self.strict_mask_mode.get():
                        color_theme = (34, 153, 210) # Orange BGR
                        status_text = "WARNING: NO MASK"
                    else:
                        color_theme = (67, 160, 46)  # Green BGR
                        status_text = f"SECURE ({name})"
                        
            # Update Live UI overlay metrics panel (on the right of video stream)
            if self.active_tab == "camera":
                self.update_info_panel(name, mask_lbl, mask_conf, score)

            # 2. Draw Semi-Transparent Face Bounding Box
            # We draw a thin line first
            cv2.rectangle(frame, (x, y), (x + w, y + h), color_theme, 1, lineType=cv2.LINE_AA)
            
            # 3. Draw Cyberpunk Corner Brackets
            length = min(20, int(w * 0.15))
            thick = 3
            # Top-Left Corner
            cv2.line(frame, (x, y), (x + length, y), color_theme, thick, lineType=cv2.LINE_AA)
            cv2.line(frame, (x, y), (x, y + length), color_theme, thick, lineType=cv2.LINE_AA)
            # Top-Right Corner
            cv2.line(frame, (x + w, y), (x + w - length, y), color_theme, thick, lineType=cv2.LINE_AA)
            cv2.line(frame, (x + w, y), (x + w, y + length), color_theme, thick, lineType=cv2.LINE_AA)
            # Bottom-Left Corner
            cv2.line(frame, (x, y + h), (x + length, y + h), color_theme, thick, lineType=cv2.LINE_AA)
            cv2.line(frame, (x, y + h), (x, y + h - length), color_theme, thick, lineType=cv2.LINE_AA)
            # Bottom-Right Corner
            cv2.line(frame, (x + w, y + h), (x + w - length, y + h), color_theme, thick, lineType=cv2.LINE_AA)
            cv2.line(frame, (x + w, y + h), (x + w, y + h - length), color_theme, thick, lineType=cv2.LINE_AA)

            # 4. Draw Face Landmarks (Glowing Dots)
            # Landmarks: left eye, right eye, nose tip, left mouth, right mouth
            for pt in landmarks:
                cv2.circle(frame, tuple(pt), 3, (244, 244, 50), -1, lineType=cv2.LINE_AA)  # Glowing Cyan dots

            # 5. Draw Info Card Banner (Glassmorphism effect)
            overlay = frame.copy()
            card_height = 28
            # Header card
            cv2.rectangle(overlay, (x, y - card_height), (x + w, y), color_theme, -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            
            # Label texts
            desc_text = f"{name} | {mask_lbl} ({int(mask_conf*100)}%)"
            cv2.putText(frame, desc_text, (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    def update_info_panel(self, name, mask, mask_conf, score):
        """Updates the statistics cards displayed on the right pane of the camera tab."""
        self.info_name_var.set(name)
        self.info_mask_var.set(f"{mask} ({int(mask_conf*100)}%)")
        self.info_score_var.set(f"{score:.3f}" if score > 0 else "0.000")
        
        # Update Status Summary Label
        if name == "Unknown":
            self.live_status_summary_var.set("UNAUTHORIZED ACCESS")
            self.live_status_summary_lbl.config(fg=self.accent_red)
        elif mask == "Unmasked" and self.strict_mask_mode.get():
            self.live_status_summary_var.set("ACCESS BLOCKED: MASK REQUIRED")
            self.live_status_summary_lbl.config(fg=self.accent_orange)
        else:
            self.live_status_summary_var.set("AUTHORIZED: SECURE")
            self.live_status_summary_lbl.config(fg=self.accent_green)

    # --- TAB FRAME CREATORS ---
    
    # 1. CAMERA TAB
    def create_camera_frame(self):
        self.camera_tab_frame = tk.Frame(self.content_pane, bg=self.bg_main)
        
        # Header Area
        header = tk.Frame(self.camera_tab_frame, bg=self.bg_main)
        header.pack(fill="x", pady=(0, 20))
        
        title = tk.Label(header, text="Real-Time Face Scanner", font=("Segoe UI", 20, "bold"), fg=self.text_primary, bg=self.bg_main)
        title.pack(anchor="w")
        subtitle = tk.Label(header, text="Scanning for identities and verifying face mask compliance", font=("Segoe UI", 10), fg=self.text_secondary, bg=self.bg_main)
        subtitle.pack(anchor="w", pady=(2, 0))
        
        # Main Work Panel (Video Feed + Side info)
        work_panel = tk.Frame(self.camera_tab_frame, bg=self.bg_main)
        work_panel.pack(fill="both", expand=True)
        
        # Webcam Feed Canvas Container (Left)
        cam_container = tk.Frame(work_panel, bg=self.bg_panel, bd=1, highlightbackground=self.border_color, highlightthickness=1)
        cam_container.pack(side="left", fill="both", expand=True, padx=(0, 15))
        
        self.cam_canvas = tk.Canvas(cam_container, bg=self.bg_panel, bd=0, highlightthickness=0, width=720, height=480)
        self.cam_canvas.pack(padx=10, pady=10, fill="both", expand=True)
        self.draw_camera_placeholder(self.cam_canvas, "Initializing Video Feed...")
        
        # Info & Settings Panel (Right)
        right_panel = tk.Frame(work_panel, bg=self.bg_main, width=280)
        right_panel.pack(side="right", fill="y")
        right_panel.pack_propagate(False)
        
        # Status Card (Glow card)
        status_card = tk.Frame(right_panel, bg=self.bg_panel, bd=1, highlightbackground=self.border_color, highlightthickness=1, pady=15, padx=15)
        status_card.pack(fill="x", pady=(0, 15))
        
        tk.Label(status_card, text="SYSTEM AUTHORIZATION", font=("Segoe UI", 9, "bold"), fg=self.text_secondary, bg=self.bg_panel).pack(anchor="w")
        
        self.live_status_summary_var = tk.StringVar(value="SCANNING...")
        self.live_status_summary_lbl = tk.Label(status_card, textvariable=self.live_status_summary_var, font=("Segoe UI", 14, "bold"), fg=self.accent_blue, bg=self.bg_panel)
        self.live_status_summary_lbl.pack(anchor="w", pady=(8, 0))
        
        # Face Detail Stats Card
        stats_card = tk.Frame(right_panel, bg=self.bg_panel, bd=1, highlightbackground=self.border_color, highlightthickness=1, pady=15, padx=15)
        stats_card.pack(fill="x", pady=(0, 15))
        
        tk.Label(stats_card, text="DETECTION STATISTICS", font=("Segoe UI", 9, "bold"), fg=self.text_secondary, bg=self.bg_panel).pack(anchor="w", pady=(0, 10))
        
        # Grid variables
        self.info_name_var = tk.StringVar(value="-")
        self.info_mask_var = tk.StringVar(value="-")
        self.info_score_var = tk.StringVar(value="-")
        
        stats_grid = tk.Frame(stats_card, bg=self.bg_panel)
        stats_grid.pack(fill="x")
        
        labels = [("Identity:", self.info_name_var), ("Mask Status:", self.info_mask_var), ("Sim Score:", self.info_score_var)]
        for r, (label_txt, var) in enumerate(labels):
            tk.Label(stats_grid, text=label_txt, font=("Segoe UI", 9), fg=self.text_secondary, bg=self.bg_panel).grid(row=r, column=0, sticky="w", pady=4)
            tk.Label(stats_grid, textvariable=var, font=("Segoe UI", 10, "bold"), fg=self.text_primary, bg=self.bg_panel).grid(row=r, column=1, sticky="w", padx=10, pady=4)
            
        # Security Directives Card (Safety checks toggling)
        directives_card = tk.Frame(right_panel, bg=self.bg_panel, bd=1, highlightbackground=self.border_color, highlightthickness=1, pady=15, padx=15)
        directives_card.pack(fill="x")
        
        tk.Label(directives_card, text="ACCESS DIRECTIVES", font=("Segoe UI", 9, "bold"), fg=self.text_secondary, bg=self.bg_panel).pack(anchor="w", pady=(0, 12))
        
        # Checkbox for strict mask check mode
        mask_mode_cb = tk.Checkbutton(
            directives_card, 
            text="Mandate Face Mask", 
            variable=self.strict_mask_mode,
            onvalue=True, 
            offvalue=False,
            font=("Segoe UI", 10),
            bg=self.bg_panel,
            fg=self.text_primary,
            selectcolor=self.bg_card,
            activebackground=self.bg_panel,
            activeforeground=self.text_primary,
            bd=0
        )
        mask_mode_cb.pack(anchor="w")
        
        directive_desc = tk.Label(directives_card, text="When enabled, unmasked faces will trigger an orange alert block even if the user is registered.", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.bg_panel, wraplength=220, justify="left")
        directive_desc.pack(anchor="w", pady=(5, 0))

    # 2. REGISTER USER TAB
    def create_register_frame(self):
        self.register_tab_frame = tk.Frame(self.content_pane, bg=self.bg_main)
        
        # Header Area
        header = tk.Frame(self.register_tab_frame, bg=self.bg_main)
        header.pack(fill="x", pady=(0, 20))
        
        title = tk.Label(header, text="Enroll New Profile", font=("Segoe UI", 20, "bold"), fg=self.text_primary, bg=self.bg_main)
        title.pack(anchor="w")
        subtitle = tk.Label(header, text="Scan facial geometry to create a secure template database file.", font=("Segoe UI", 10), fg=self.text_secondary, bg=self.bg_main)
        subtitle.pack(anchor="w", pady=(2, 0))
        
        # Layout Split: Form (Left) vs Register Preview Camera (Right)
        split_panel = tk.Frame(self.register_tab_frame, bg=self.bg_main)
        split_panel.pack(fill="both", expand=True)
        
        # Form Container (Left)
        form_container = tk.Frame(split_panel, bg=self.bg_panel, bd=1, highlightbackground=self.border_color, highlightthickness=1, padx=25, pady=25)
        form_container.pack(side="left", fill="both", expand=True, padx=(0, 15))
        
        tk.Label(form_container, text="REGISTRATION METADATA", font=("Segoe UI", 9, "bold"), fg=self.text_secondary, bg=self.bg_panel).pack(anchor="w", pady=(0, 20))
        
        # Name Input field
        tk.Label(form_container, text="Full Name", font=("Segoe UI", 10), fg=self.text_primary, bg=self.bg_panel).pack(anchor="w", pady=(0, 5))
        self.reg_name_entry = tk.Entry(form_container, font=("Segoe UI", 11), bg=self.bg_card, fg=self.text_primary, insertbackground=self.text_primary, bd=1, relief="solid", highlightthickness=0)
        self.reg_name_entry.config(highlightbackground=self.border_color, highlightcolor=self.accent_blue)
        self.reg_name_entry.pack(fill="x", ipady=8, pady=(0, 20))
        
        # Capture Trigger Button
        self.reg_btn = tk.Button(
            form_container,
            text="Start Profile Acquisition",
            font=("Segoe UI", 11, "bold"),
            bg=self.accent_blue,
            fg=self.text_primary,
            activebackground="#1f6feb",
            activeforeground=self.text_primary,
            bd=0,
            relief="flat",
            command=self.start_registration_capture
        )
        self.reg_btn.pack(fill="x", ipady=10, pady=(0, 30))
        self.style_button(self.reg_btn, "#1f6feb", self.text_primary, self.accent_blue, self.text_primary)
        
        # Capture Status monitor
        status_box = tk.Frame(form_container, bg=self.bg_card, bd=1, highlightbackground=self.border_color, highlightthickness=1, pady=15, padx=15)
        status_box.pack(fill="x")
        
        tk.Label(status_box, text="ACQUISITION STATE", font=("Segoe UI", 8, "bold"), fg=self.text_secondary, bg=self.bg_card).pack(anchor="w")
        
        status_lbl = tk.Label(status_box, textvariable=self.reg_status_text, font=("Segoe UI", 10, "bold"), fg=self.accent_blue, bg=self.bg_card, wraplength=350, justify="left")
        status_lbl.pack(anchor="w", pady=(10, 8))
        
        self.reg_progress_bar = ttk.Progressbar(status_box, variable=self.reg_progress, mode="determinate", style="TProgressbar")
        self.reg_progress_bar.pack(fill="x", pady=5)
        
        # Register Camera Preview Panel (Right)
        preview_container = tk.Frame(split_panel, bg=self.bg_panel, bd=1, highlightbackground=self.border_color, highlightthickness=1, width=420)
        preview_container.pack(side="right", fill="y")
        preview_container.pack_propagate(False)
        
        tk.Label(preview_container, text="ALIGNMENT VIEWPORT", font=("Segoe UI", 9, "bold"), fg=self.text_secondary, bg=self.bg_panel, pady=15).pack()
        
        self.reg_cam_canvas = tk.Canvas(preview_container, bg=self.bg_panel, bd=0, highlightthickness=0, width=380, height=280)
        self.reg_cam_canvas.pack(padx=20, fill="both", expand=True)
        self.draw_camera_placeholder(self.reg_cam_canvas, "Connecting to camera viewport...")
        
        instructions = (
            "Calibration Guidelines:\n"
            "1. Stand still facing the camera directly.\n"
            "2. Make sure you are UNMASKED during registration.\n"
            "3. Ensure the room has balanced lighting.\n"
            "4. The system will auto-capture 5 face geometry samples."
        )
        instruct_lbl = tk.Label(preview_container, text=instructions, font=("Segoe UI", 9), fg=self.text_secondary, bg=self.bg_panel, justify="left", wraplength=350, pady=20)
        instruct_lbl.pack()

    def start_registration_capture(self):
        """Starts the capture routine for registration."""
        if self.orchestrator is None:
            messagebox.showerror("Error", "Models are still initializing. Please wait.")
            return
            
        name = self.reg_name_entry.get().strip()
        if not name:
            messagebox.showwarning("Input Required", "Please enter the user's name before registering.")
            return
            
        # Clean state for registration capture
        self.reg_name = name
        self.reg_frames = []
        self.reg_progress.set(0.0)
        self.reg_status_text.set("Initializing biometric capture...")
        self.registering = True
        self.reg_btn.config(state="disabled", text="Acquiring Biometrics...")
        self.log_event(f"SYSTEM: Initiated face profile acquisition for '{name}'")

    def finalize_registration(self):
        """Processes the 5 captured face crops, computes their average embeddings, and saves to database."""
        self.reg_status_text.set("Computing average feature embeddings...")
        self.log_event(f"SYSTEM: Captured 5 face templates for {self.reg_name}. Finalizing enrollment...")
        
        recognizer = self.orchestrator.recognition_agent.recognizer if self.orchestrator else None
        if recognizer is None:
            self.reg_status_text.set("Registration failed. Orchestrator not ready.")
            self.root.after(0, lambda: self.reg_btn.config(state="normal", text="Start Profile Acquisition"))
            return
        
        full_embs = []
        upper_embs = []
        
        for idx, aligned_face in enumerate(self.reg_frames):
            try:
                f_emb = recognizer.extract_feature(aligned_face)
                u_emb = recognizer.extract_upper_face_feature(aligned_face)
                full_embs.append(f_emb)
                upper_embs.append(u_emb)
            except Exception as e:
                print(f"Embedding error on frame {idx}: {e}")
                
        if len(full_embs) > 0:
            avg_full_emb = np.mean(full_embs, axis=0)
            avg_upper_emb = np.mean(upper_embs, axis=0)
            
            self.db.register_user(self.reg_name, avg_full_emb, avg_upper_emb)
            
            self.log_event(f"DATABASE: Successfully registered user '{self.reg_name}'")
            self.reg_status_text.set(f"Enrollment Successful! '{self.reg_name}' registered.")
            
            self.root.after(0, lambda: self.reg_name_entry.delete(0, tk.END))
        else:
            self.reg_status_text.set("Registration failed. Unable to extract features.")
            self.log_event(f"ERROR: Enrollment failed for {self.reg_name}.")
            
        self.root.after(0, lambda: self.reg_btn.config(state="normal", text="Start Profile Acquisition"))

    # 3. USER DIRECTORY TAB
    def create_directory_frame(self):
        self.directory_tab_frame = tk.Frame(self.content_pane, bg=self.bg_main)
        
        # Header Area
        header = tk.Frame(self.directory_tab_frame, bg=self.bg_main)
        header.pack(fill="x", pady=(0, 20))
        
        title = tk.Label(header, text="User Directory", font=("Segoe UI", 20, "bold"), fg=self.text_primary, bg=self.bg_main)
        title.pack(anchor="w")
        subtitle = tk.Label(header, text="Manage enrolled facial identities and credential keys", font=("Segoe UI", 10), fg=self.text_secondary, bg=self.bg_main)
        subtitle.pack(anchor="w", pady=(2, 0))
        
        # Main Table Card Background
        table_card = tk.Frame(self.directory_tab_frame, bg=self.bg_panel, bd=1, highlightbackground=self.border_color, highlightthickness=1, padx=20, pady=20)
        table_card.pack(fill="both", expand=True)
        
        # Scrollable user list frame
        self.user_table_frame = tk.Frame(table_card, bg=self.bg_panel)
        self.user_table_frame.pack(fill="both", expand=True)

    def refresh_directory_table(self):
        """Cleans and re-populates the directory list of users with Delete buttons."""
        for widget in self.user_table_frame.winfo_children():
            widget.destroy()
            
        # Headers
        headers = [("Profile ID / Name", 0.3), ("Times Detected", 0.25), ("Last Seen Time", 0.3), ("Operations", 0.15)]
        headers_frame = tk.Frame(self.user_table_frame, bg=self.bg_card, height=35)
        headers_frame.pack(fill="x", pady=(0, 10))
        
        # Grid columns
        headers_frame.columnconfigure(0, weight=3)
        headers_frame.columnconfigure(1, weight=2)
        headers_frame.columnconfigure(2, weight=3)
        headers_frame.columnconfigure(3, weight=2)
        
        for c, (txt, weight) in enumerate(headers):
            lbl = tk.Label(headers_frame, text=txt, font=("Segoe UI", 9, "bold"), fg=self.text_secondary, bg=self.bg_card, anchor="w", padx=10)
            lbl.grid(row=0, column=c, sticky="nsew", pady=6)
            
        registered_names = self.db.get_registered_names()
        
        if not registered_names:
            no_users_lbl = tk.Label(self.user_table_frame, text="No users enrolled. Click 'Register User' in the sidebar to add profiles.", font=("Segoe UI", 11), fg=self.text_secondary, bg=self.bg_panel, pady=40)
            no_users_lbl.pack()
            return
            
        # Draw user rows
        # Add scrollable canvas inside user_table_frame if directory grows large
        scroll_canvas = tk.Canvas(self.user_table_frame, bg=self.bg_panel, bd=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.user_table_frame, orient="vertical", command=scroll_canvas.yview)
        scrollable_frame = tk.Frame(scroll_canvas, bg=self.bg_panel)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: scroll_canvas.configure(
                scrollregion=scroll_canvas.bbox("all")
            )
        )
        
        scroll_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=750)
        scroll_canvas.configure(yscrollcommand=scrollbar.set)
        
        scroll_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        for r, name in enumerate(registered_names):
            row_frame = tk.Frame(scrollable_frame, bg=self.bg_panel)
            row_frame.pack(fill="x", pady=3)
            
            row_frame.columnconfigure(0, weight=3)
            row_frame.columnconfigure(1, weight=2)
            row_frame.columnconfigure(2, weight=3)
            row_frame.columnconfigure(3, weight=2)
            
            # Fetch stats from log db
            summary = self.log_db.get_person_summary(name)
            det_count = summary["detection_count"] if summary else 0
            last_seen = summary["last_seen"] if summary else "Never"
            
            # Name
            name_lbl = tk.Label(row_frame, text=name, font=("Segoe UI", 10, "bold"), fg=self.text_primary, bg=self.bg_panel, anchor="w", padx=10)
            name_lbl.grid(row=0, column=0, sticky="nsew", pady=8)
            
            # Times Detected
            det_lbl = tk.Label(row_frame, text=f"{det_count} times", font=("Segoe UI", 9, "bold"), fg=self.accent_blue if det_count > 0 else self.text_secondary, bg=self.bg_panel, anchor="w", padx=10)
            det_lbl.grid(row=0, column=1, sticky="nsew", pady=8)
            
            # Last Seen Time
            last_lbl = tk.Label(row_frame, text=last_seen, font=("Segoe UI", 9, "bold"), fg=self.accent_green if det_count > 0 else self.text_secondary, bg=self.bg_panel, anchor="w", padx=10)
            last_lbl.grid(row=0, column=2, sticky="nsew", pady=8)
            
            # Delete Button
            del_btn = tk.Button(
                row_frame,
                text="Delete",
                font=("Segoe UI", 9),
                bg="#30363d",
                fg=self.accent_red,
                activebackground=self.accent_red,
                activeforeground=self.text_primary,
                bd=0,
                relief="flat",
                command=lambda n=name: self.delete_user_profile(n)
            )
            del_btn.grid(row=0, column=3, sticky="e", padx=10, pady=4, ipadx=8, ipady=3)
            self.style_button(del_btn, self.accent_red, self.text_primary, "#30363d", self.accent_red)
            
            # Bottom row divider
            row_divider = tk.Frame(scrollable_frame, bg=self.border_color, height=1)
            row_divider.pack(fill="x", pady=2)

    def delete_user_profile(self, name):
        """Confirms and deletes a user profile from database."""
        confirm = messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete profile '{name}' from the MFR database?")
        if confirm:
            self.db.delete_user(name)
            self.log_db.delete_person(name)
            self.log_event(f"DATABASE: Deleted user profile '{name}' and detection history")
            self.refresh_directory_table()

    # 4. SYSTEM LOGS TAB
    def create_logs_frame(self):
        self.logs_tab_frame = tk.Frame(self.content_pane, bg=self.bg_main)
        
        # Header Area
        header = tk.Frame(self.logs_tab_frame, bg=self.bg_main)
        header.pack(fill="x", pady=(0, 20))
        
        title = tk.Label(header, text="System Log Console", font=("Segoe UI", 20, "bold"), fg=self.text_primary, bg=self.bg_main)
        title.pack(anchor="w")
        subtitle = tk.Label(header, text="View real-time authorization audits and system security events", font=("Segoe UI", 10), fg=self.text_secondary, bg=self.bg_main)
        subtitle.pack(anchor="w", pady=(2, 0))
        
        # Logs Container Panel
        log_panel = tk.Frame(self.logs_tab_frame, bg=self.bg_panel, bd=1, highlightbackground=self.border_color, highlightthickness=1, padx=20, pady=20)
        log_panel.pack(fill="both", expand=True)
        
        # Log Text Box
        self.log_text_widget = tk.Text(log_panel, font=("Consolas", 10), bg=self.bg_main, fg=self.text_secondary, insertbackground=self.text_primary, bd=0, wrap="word", state="disabled")
        scrollbar = ttk.Scrollbar(log_panel, orient="vertical", command=self.log_text_widget.yview)
        self.log_text_widget.configure(yscrollcommand=scrollbar.set)
        
        self.log_text_widget.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Custom coloring in logs text widget
        self.log_text_widget.tag_config("SYSTEM", foreground=self.text_secondary)
        self.log_text_widget.tag_config("DETECTED", foreground=self.accent_green)
        self.log_text_widget.tag_config("ALERT", foreground=self.accent_red, font=("Consolas", 10, "bold"))
        self.log_text_widget.tag_config("WARNING", foreground=self.accent_orange)
        self.log_text_widget.tag_config("DATABASE", foreground=self.accent_blue)

    def update_logs_ui(self):
        """Flushes logs to log text widget with styled tags."""
        if not hasattr(self, 'log_text_widget'):
            return
            
        self.log_text_widget.config(state="normal")
        self.log_text_widget.delete("1.0", tk.END)
        
        for log in self.logs:
            tag = "SYSTEM"
            if "DETECTED:" in log:
                tag = "DETECTED"
            elif "ALERT:" in log or "VIOLATION:" in log or "ERROR:" in log:
                tag = "ALERT"
            elif "WARNING:" in log:
                tag = "WARNING"
            elif "DATABASE:" in log:
                tag = "DATABASE"
                
            self.log_text_widget.insert(tk.END, log + "\n", tag)
            
        self.log_text_widget.see(tk.END)
        self.log_text_widget.config(state="disabled")

    # 5. SYSTEM SETTINGS TAB
    def create_settings_frame(self):
        self.settings_tab_frame = tk.Frame(self.content_pane, bg=self.bg_main)
        
        # Header Area
        header = tk.Frame(self.settings_tab_frame, bg=self.bg_main)
        header.pack(fill="x", pady=(0, 20))
        
        title = tk.Label(header, text="Calibration & Settings", font=("Segoe UI", 20, "bold"), fg=self.text_primary, bg=self.bg_main)
        title.pack(anchor="w")
        subtitle = tk.Label(header, text="Fine-tune deep neural network confidence bounds and clear memory caches", font=("Segoe UI", 10), fg=self.text_secondary, bg=self.bg_main)
        subtitle.pack(anchor="w", pady=(2, 0))
        
        settings_panel = tk.Frame(self.settings_tab_frame, bg=self.bg_panel, bd=1, highlightbackground=self.border_color, highlightthickness=1, padx=25, pady=25)
        settings_panel.pack(fill="both", expand=True)
        
        # Setting Row 1: Face Recognizer Similarity Bound (Threshold)
        tk.Label(settings_panel, text="Biometric Matching Sensitivity (SFace Cosine Threshold)", font=("Segoe UI", 10, "bold"), fg=self.text_primary, bg=self.bg_panel).pack(anchor="w", pady=(0, 5))
        tk.Label(settings_panel, text="Adjusting this value dictates the matching tolerance. Higher values demand precise features (reduces False Positives but increases False Negatives). Recommended default is 0.363.", font=("Segoe UI", 9), fg=self.text_secondary, bg=self.bg_panel, wraplength=600, justify="left").pack(anchor="w", pady=(0, 10))
        
        self.threshold_slider_var = tk.DoubleVar(value=0.363)
        self.threshold_lbl_var = tk.StringVar(value="Current Boundary: 0.363")
        
        slider_frame = tk.Frame(settings_panel, bg=self.bg_panel)
        slider_frame.pack(fill="x", pady=(0, 30))
        
        slider = ttk.Scale(slider_frame, from_=0.15, to=0.60, variable=self.threshold_slider_var, orient="horizontal", command=self.on_threshold_change)
        slider.pack(side="left", fill="x", expand=True, padx=(0, 20))
        
        threshold_lbl = tk.Label(slider_frame, textvariable=self.threshold_lbl_var, font=("Segoe UI", 10, "bold"), fg=self.accent_blue, bg=self.bg_panel, width=22, anchor="e")
        threshold_lbl.pack(side="right")
        
        # Setting Row 2: Performance Inference Speed Interval
        tk.Label(settings_panel, text="Webcam Scanner Refresh Skips (Frame Skip Interval)", font=("Segoe UI", 10, "bold"), fg=self.text_primary, bg=self.bg_panel).pack(anchor="w", pady=(0, 5))
        tk.Label(settings_panel, text="Configure how frequently the neural network runs checks. Skipping frames decreases CPU burden and ensures smooth feed (30 FPS). Lower values give faster response but high CPU usage.", font=("Segoe UI", 9), fg=self.text_secondary, bg=self.bg_panel, wraplength=600, justify="left").pack(anchor="w", pady=(0, 10))
        
        self.interval_slider_var = tk.IntVar(value=3)
        self.interval_lbl_var = tk.StringVar(value="Run inference every 3 frames")
        
        slider_frame2 = tk.Frame(settings_panel, bg=self.bg_panel)
        slider_frame2.pack(fill="x", pady=(0, 30))
        
        slider2 = ttk.Scale(slider_frame2, from_=1, to=10, variable=self.interval_slider_var, orient="horizontal", command=self.on_interval_change)
        slider2.pack(side="left", fill="x", expand=True, padx=(0, 20))
        
        interval_lbl = tk.Label(slider_frame2, textvariable=self.interval_lbl_var, font=("Segoe UI", 10, "bold"), fg=self.accent_blue, bg=self.bg_panel, width=28, anchor="e")
        interval_lbl.pack(side="right")
        
        # Setting Row 3: Database wipe button
        tk.Label(settings_panel, text="Database Administration", font=("Segoe UI", 10, "bold"), fg=self.text_primary, bg=self.bg_panel).pack(anchor="w", pady=(0, 10))
        
        action_frame = tk.Frame(settings_panel, bg=self.bg_panel)
        action_frame.pack(fill="x", pady=(0, 10))
        
        wipe_db_btn = tk.Button(
            action_frame,
            text="Wipe Database",
            font=("Segoe UI", 10, "bold"),
            bg="#30363d",
            fg=self.accent_red,
            activebackground=self.accent_red,
            activeforeground=self.text_primary,
            bd=0,
            relief="flat",
            command=self.wipe_database
        )
        wipe_db_btn.pack(side="left", ipady=8, ipadx=15)
        self.style_button(wipe_db_btn, self.accent_red, self.text_primary, "#30363d", self.accent_red)

    def on_threshold_change(self, val):
        """Callback when SFace cosine threshold slider updates."""
        curr_val = self.threshold_slider_var.get()
        self.threshold_lbl_var.set(f"Current Boundary: {curr_val:.3f}")
        if self.orchestrator is not None:
            self.orchestrator.set_cosine_threshold(curr_val)

    def on_interval_change(self, val):
        """Callback when frame skip interval slider updates."""
        curr_val = int(round(self.interval_slider_var.get()))
        self.interval_slider_var.set(curr_val)
        self.interval_lbl_var.set(f"Run inference every {curr_val} frames")
        self.detection_interval = curr_val

    def wipe_database(self):
        """Wipes the database file and resets memory."""
        confirm = messagebox.askyesno("Confirm Database Wipe", "CRITICAL WARNING:\n\nAre you sure you want to completely erase the MFR database? All enrolled user profiles and their corresponding templates will be deleted permanently.")
        if confirm:
            if os.path.exists("db.json"):
                os.remove("db.json")
            self.db = mfr.Database()
            self.log_db.clear_all()
            if self.orchestrator is not None:
                # Reload the orchestrator's internal db reference
                self.orchestrator.recognition_agent.db = self.db
            self.log_event("DATABASE: Completely wiped facial database file and detection log database.")
            messagebox.showinfo("Database Cleaned", "Biometric face database and detection logs successfully wiped clean.")

# --- MAIN INVOCATOR ---
if __name__ == "__main__":
    root = tk.Tk()
    app = MFRSystemApp(root)
    
    # Custom window close event to safely release resources
    def on_close():
        app.stop_camera()
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
