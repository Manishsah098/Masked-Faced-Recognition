document.addEventListener('DOMContentLoaded', () => {
    // --- CONNECT TO SOCKET.IO SERVER ---
    const socket = io();
    const statusText = document.getElementById('system-status-text');
    const statusDot = document.getElementById('system-status-dot');

    socket.on('connect', () => {
        if (statusText) statusText.innerText = "Swarm Online (10 Agents)";
        if (statusDot) {
            statusDot.className = "dot active";
            statusDot.style.backgroundColor = 'var(--emerald)';
        }
    });

    socket.on('disconnect', () => {
        if (statusText) statusText.innerText = "Pipeline Disconnected";
        if (statusDot) {
            statusDot.className = "dot";
            statusDot.style.backgroundColor = 'var(--red)';
        }
    });

    // --- NAVIGATION ---
    const navItems = document.querySelectorAll('.nav-links li');
    const views = document.querySelectorAll('.view');

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            navItems.forEach(n => n.classList.remove('active'));
            views.forEach(v => {
                v.classList.remove('active-view');
                v.classList.add('hidden-view');
            });
            
            item.classList.add('active');
            const targetId = item.getAttribute('data-target');
            const targetView = document.getElementById(targetId);
            if (targetView) {
                targetView.classList.remove('hidden-view');
                targetView.classList.add('active-view');
            }

            if (targetId === 'directory-view') loadDirectory();
            if (targetId === 'logs-view') fetchLogs();
        });
    });

    // --- CHART.JS COMPLIANCE STATISTICS ---
    let complianceChart = null;
    let complianceStats = {
        masked: 0,
        unmasked: 0,
        violations: 0
    };
    let totalScans = 0;

    function initChart() {
        const chartElem = document.getElementById('complianceChart');
        if (!chartElem) return;
        const ctx = chartElem.getContext('2d');
        complianceChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Compliant (Masked)', 'Standard (Unmasked)', 'Violations'],
                datasets: [{
                    data: [0, 0, 0],
                    backgroundColor: ['#00f0ff', '#00e676', '#ff3366'],
                    borderColor: '#0d131d',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right',
                        labels: {
                            color: '#8b99ad',
                            font: { size: 9.5, family: 'Inter' },
                            boxWidth: 8
                        }
                    }
                },
                cutout: '68%'
            }
        });
    }

    function updateChartStats(name, mask, status_msg) {
        if (name === '-') return;
        totalScans++;
        
        const totalScansElem = document.getElementById('total-scans-text');
        if (totalScansElem) totalScansElem.innerText = `${totalScans} verified scans`;

        if (status_msg.includes('VIOLATION') || status_msg.includes('BLOCKED') || status_msg.includes('DENIED')) {
            complianceStats.violations++;
        } else if (mask.includes('Mask')) {
            complianceStats.masked++;
        } else {
            complianceStats.unmasked++;
        }
        
        if (complianceChart) {
            complianceChart.data.datasets[0].data = [
                complianceStats.masked,
                complianceStats.unmasked,
                complianceStats.violations
            ];
            complianceChart.update();
        }
    }

    initChart();

    // --- TEXT-TO-SPEECH (TTS) AUDIO ALERTS ---
    let voiceEnabled = localStorage.getItem('mfr_voice_alerts') !== 'false';
    let lastVoiceTime = 0;

    const btnVoiceToggle = document.getElementById('btn-voice-toggle');
    const voiceIcon = document.getElementById('voice-icon');

    function updateVoiceBtnState() {
        if (!btnVoiceToggle) return;
        if (voiceEnabled) {
            btnVoiceToggle.classList.add('active');
            if (voiceIcon) voiceIcon.className = 'fa-solid fa-volume-high';
        } else {
            btnVoiceToggle.classList.remove('active');
            if (voiceIcon) voiceIcon.className = 'fa-solid fa-volume-xmark';
        }
    }

    if (btnVoiceToggle) {
        updateVoiceBtnState();
        btnVoiceToggle.addEventListener('click', () => {
            voiceEnabled = !voiceEnabled;
            localStorage.setItem('mfr_voice_alerts', voiceEnabled);
            updateVoiceBtnState();
        });
    }

    function speak(text) {
        if (!voiceEnabled) return;
        const now = Date.now();
        // Cooldown of 5 seconds to prevent spam/stutter
        if (now - lastVoiceTime < 5000) return;
        lastVoiceTime = now;
        
        try {
            if ('speechSynthesis' in window) {
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.rate = 1.0;
                utterance.pitch = 1.0;
                window.speechSynthesis.speak(utterance);
            }
        } catch (e) {
            console.warn("TTS error:", e);
        }
    }

    function handleVoiceAlerts(name, mask, status_msg) {
        if (name === '-') return;
        
        if (status_msg.includes('VIOLATION') || status_msg.includes('BLOCKED')) {
            speak(`Notice: Face mask required, ${name}`);
        } else if (name === 'Unknown' || status_msg.includes('DENIED')) {
            speak("Access Denied: Unregistered profile");
        } else if (status_msg.includes('GRANTED') || status_msg.includes('VERIFIED')) {
            speak(`Access Granted: Welcome ${name}`);
        }
    }

    // --- BROWSER WEBCAM STREAMING & FPS METRICS ---
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('hidden-canvas');
    const ctx = canvas.getContext('2d');
    const feedImg = document.getElementById('video-feed');
    const cameraSelect = document.getElementById('camera-select');
    const btnMirror = document.getElementById('btn-mirror-cam');
    const btnFlipV = document.getElementById('btn-flipv-cam');
    const btnRefresh = document.getElementById('btn-refresh-cam');

    let isProcessing = false;
    let lastSendTime = 0;
    let currentStream = null;
    let loopRunning = false;

    // Performance FPS tracker
    let frameCount = 0;
    let lastFpsCalc = Date.now();
    let currentFps = 30;

    // Persisted preferences
    let selectedCameraId = localStorage.getItem('mfr_camera_id') || '';
    let mirrorH = localStorage.getItem('mfr_mirror_h') === 'true';
    let flipV = localStorage.getItem('mfr_flip_v') === 'true';

    if (btnMirror && mirrorH) btnMirror.classList.add('active');
    if (btnFlipV && flipV) btnFlipV.classList.add('active');

    async function enumerateCameraDevices() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const videoDevices = devices.filter(d => d.kind === 'videoinput');
            
            if (cameraSelect) {
                cameraSelect.innerHTML = '';
                if (videoDevices.length === 0) {
                    const opt = document.createElement('option');
                    opt.value = '';
                    opt.innerText = 'No camera found';
                    cameraSelect.appendChild(opt);
                    return;
                }

                videoDevices.forEach((dev, idx) => {
                    const opt = document.createElement('option');
                    opt.value = dev.deviceId;
                    opt.innerText = dev.label || `Camera ${idx + 1}`;
                    if (dev.deviceId === selectedCameraId) {
                        opt.selected = true;
                    }
                    cameraSelect.appendChild(opt);
                });

                if (!videoDevices.some(d => d.deviceId === selectedCameraId) && videoDevices.length > 0) {
                    selectedCameraId = videoDevices[0].deviceId;
                    cameraSelect.value = selectedCameraId;
                }
            }
        } catch (err) {
            console.warn("Could not enumerate devices:", err);
        }
    }

    async function startCamera(deviceId = null) {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            if (statusText) statusText.innerText = "Camera Requires HTTPS/Localhost";
            return;
        }

        if (currentStream) {
            currentStream.getTracks().forEach(track => track.stop());
            currentStream = null;
        }

        const chosenId = deviceId || selectedCameraId;
        const constraints = {
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 }
            }
        };

        if (chosenId) {
            constraints.video.deviceId = { exact: chosenId };
        } else {
            constraints.video.facingMode = "user";
        }

        try {
            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            onStreamSuccess(stream, chosenId);
        } catch (err) {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                onStreamSuccess(stream);
            } catch (err2) {
                console.error("Camera access failed:", err2);
                if (statusText) statusText.innerText = "Camera Access Denied";
            }
        }
    }

    function onStreamSuccess(stream, requestedId = null) {
        currentStream = stream;
        video.srcObject = stream;
        video.muted = true;
        video.playsInline = true;

        const activeTrack = stream.getVideoTracks()[0];
        if (activeTrack) {
            const settings = activeTrack.getSettings ? activeTrack.getSettings() : {};
            if (settings.deviceId) {
                selectedCameraId = settings.deviceId;
                localStorage.setItem('mfr_camera_id', selectedCameraId);
            }
        }

        enumerateCameraDevices();

        const playPromise = video.play();
        if (playPromise !== undefined) {
            playPromise.then(() => {
                if (!loopRunning) {
                    loopRunning = true;
                    requestAnimationFrame(captureLoop);
                }
            }).catch(() => {
                if (!loopRunning) {
                    loopRunning = true;
                    requestAnimationFrame(captureLoop);
                }
            });
        } else if (!loopRunning) {
            loopRunning = true;
            requestAnimationFrame(captureLoop);
        }
    }

    if (cameraSelect) {
        cameraSelect.addEventListener('change', (e) => {
            selectedCameraId = e.target.value;
            localStorage.setItem('mfr_camera_id', selectedCameraId);
            startCamera(selectedCameraId);
        });
    }

    if (btnMirror) {
        btnMirror.addEventListener('click', () => {
            mirrorH = !mirrorH;
            localStorage.setItem('mfr_mirror_h', mirrorH);
            btnMirror.classList.toggle('active', mirrorH);
        });
    }

    if (btnFlipV) {
        btnFlipV.addEventListener('click', () => {
            flipV = !flipV;
            localStorage.setItem('mfr_flip_v', flipV);
            btnFlipV.classList.toggle('active', flipV);
        });
    }

    if (btnRefresh) {
        btnRefresh.addEventListener('click', () => {
            startCamera(selectedCameraId);
        });
    }

    if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
        navigator.mediaDevices.addEventListener('devicechange', enumerateCameraDevices);
    }

    startCamera();

    function captureLoop() {
        const now = Date.now();
        if ((!isProcessing || (now - lastSendTime > 1000)) && socket.connected) {
            sendFrame();
        }
        setTimeout(() => {
            requestAnimationFrame(captureLoop);
        }, 50);
    }

    function sendFrame() {
        if (!video.videoWidth || !video.videoHeight || video.paused || video.ended) return;
        isProcessing = true;
        lastSendTime = Date.now();
        
        canvas.width = 640;
        canvas.height = 480;

        ctx.save();
        ctx.translate(mirrorH ? 640 : 0, flipV ? 480 : 0);
        ctx.scale(mirrorH ? -1 : 1, flipV ? -1 : 1);
        ctx.drawImage(video, 0, 0, 640, 480);
        ctx.restore();

        const dataUrl = canvas.toDataURL('image/jpeg', 0.55);
        socket.emit('image', dataUrl);
    }

    // --- HANDLE MULTI-AGENT RESPONSE FROM SERVER ---
    socket.on('response', (data) => {
        isProcessing = false;
        const now = Date.now();
        const latency = now - lastSendTime;

        // Update FPS Counter
        frameCount++;
        if (now - lastFpsCalc >= 1000) {
            currentFps = frameCount;
            frameCount = 0;
            lastFpsCalc = now;
            const fpsElem = document.getElementById('fps-display');
            const latElem = document.getElementById('latency-display');
            if (fpsElem) fpsElem.innerText = `${currentFps} FPS`;
            if (latElem) latElem.innerText = `${latency} ms`;
        }

        // Render Annotated HUD frame
        if (feedImg && data.image) {
            feedImg.src = data.image;
        }

        const state = data.state;
        if (!state) return;

        // Top Status & Candidate
        const statName = document.getElementById('stat-name');
        const statScore = document.getElementById('stat-score');
        const authText = document.getElementById('auth-status-text');
        const authBox = document.getElementById('auth-status-box');
        const confBar = document.getElementById('confidence-bar');
        const xaiText = document.getElementById('xai-explanation');

        if (statName) statName.innerText = state.name || '-';
        if (statScore) statScore.innerText = state.score || '0.0%';
        if (authText) {
            authText.innerText = state.status_msg || 'SCANNING...';
            authText.style.color = state.status_color || 'var(--cyan)';
        }
        if (authBox) {
            authBox.style.borderLeftColor = state.status_color || 'var(--cyan)';
        }

        // Numeric Confidence Score Parse
        if (confBar) {
            const rawScore = parseFloat((state.score || '0').replace('%', ''));
            confBar.style.width = Math.min(100, Math.max(0, rawScore)) + '%';
        }

        // Explainable AI Reasoning
        if (xaiText && state.explanation) {
            xaiText.innerText = state.explanation;
        }

        // 10-Agent Live Telemetry Unpacking
        if (state.agents) {
            const agents = state.agents;

            // 1. Quality Agent
            if (agents.quality) {
                const qScore = document.getElementById('stat-quality-score');
                const qStatus = document.getElementById('stat-quality-status');
                const qBar = document.getElementById('quality-bar');
                if (qScore) qScore.innerText = agents.quality.sharpness ? `${agents.quality.sharpness.toFixed(1)}` : `${agents.quality.score || 0}%`;
                if (qStatus) qStatus.innerText = agents.quality.status || 'OK';
                if (qBar) qBar.style.width = Math.min(100, (agents.quality.score || 0)) + '%';
            }

            // 2. Mask Agent
            if (agents.mask) {
                const mVal = document.getElementById('stat-mask');
                const mBadge = document.getElementById('stat-mask-badge');
                const mBar = document.getElementById('mask-bar');
                if (mVal) mVal.innerText = state.mask;
                if (mBadge) mBadge.innerText = agents.mask.is_masked ? 'MASKED' : 'UNMASKED';
                if (mBar) mBar.style.width = (agents.mask.mask_confidence || 0) + '%';
            }

            // 3. Occlusion Strategy Agent
            if (agents.occlusion) {
                const sBadge = document.getElementById('stat-strategy-badge');
                const visVal = document.getElementById('stat-visibility');
                const visBar = document.getElementById('visibility-bar');
                if (sBadge) sBadge.innerText = agents.occlusion.strategy || 'FULL';
                if (visVal) visVal.innerText = `${(agents.occlusion.visibility || 100).toFixed(0)}%`;
                if (visBar) visBar.style.width = `${agents.occlusion.visibility || 100}%`;
            }

            // 4. Liveness Anti-Spoof Agent
            if (agents.liveness) {
                const lStatus = document.getElementById('stat-liveness-status');
                const lScore = document.getElementById('stat-liveness-score');
                const lBar = document.getElementById('liveness-bar');
                if (lStatus) lStatus.innerText = agents.liveness.status || 'LIVE';
                if (lScore) lScore.innerText = `${(agents.liveness.score || 95).toFixed(1)}%`;
                if (lBar) lBar.style.width = `${agents.liveness.score || 95}%`;
            }
        }

        // Voice alert trigger
        handleVoiceAlerts(state.name, state.mask, state.status_msg);

        // Chart stats trigger
        updateChartStats(state.name, state.mask, state.status_msg);
    });

    // --- REGISTRATION / ENROLLMENT LOGIC ---
    const btnStartReg = document.getElementById('btn-start-reg');
    let regPollInterval = null;

    if (btnStartReg) {
        btnStartReg.addEventListener('click', async () => {
            const nameInput = document.getElementById('reg-name');
            const name = nameInput ? nameInput.value.trim() : '';
            if (!name) return alert("Please enter a subject full name first.");

            btnStartReg.disabled = true;
            btnStartReg.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Acquiring Biometrics...';

            try {
                await fetch('/api/register', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ name })
                });

                if (regPollInterval) clearInterval(regPollInterval);
                regPollInterval = setInterval(pollRegistrationStatus, 400);
            } catch (e) {
                alert("Biometric enrollment request failed.");
                btnStartReg.disabled = false;
                btnStartReg.innerHTML = '<i class="fa-solid fa-camera-retro"></i> Start Multi-Frame Acquisition';
            }
        });
    }

    async function pollRegistrationStatus() {
        try {
            const res = await fetch('/api/register_status');
            const data = await res.json();
            
            const regText = document.getElementById('reg-status-text');
            const regBar = document.getElementById('reg-progress-bar');
            const regPct = document.getElementById('reg-percentage');

            if (regText) regText.innerText = data.status_text;
            if (regBar) regBar.style.width = data.progress + "%";
            if (regPct) regPct.innerText = Math.round(data.progress) + "%";

            if (!data.is_registering && data.progress === 0 && !data.status_text.includes("ALERT")) {
                clearInterval(regPollInterval);
                if (btnStartReg) {
                    btnStartReg.disabled = false;
                    btnStartReg.innerHTML = '<i class="fa-solid fa-camera-retro"></i> Start Multi-Frame Acquisition';
                }
                const nameInput = document.getElementById('reg-name');
                if (nameInput) nameInput.value = '';
                loadDirectory();
            }
        } catch (e) {
            console.error("Poll registration error:", e);
        }
    }

    // --- SYSTEM LOGS CONSOLE ---
    let logsInterval = setInterval(fetchLogs, 2500);

    async function fetchLogs() {
        const logsView = document.getElementById('logs-view');
        if (!logsView || !logsView.classList.contains('active-view')) return;
        try {
            const res = await fetch('/api/logs');
            const data = await res.json();
            
            const container = document.getElementById('logs-container');
            if (!container) return;
            container.innerHTML = '';
            
            data.logs.forEach(log => {
                const div = document.createElement('div');
                div.className = 'log-entry';
                
                if (log.includes('DETECTED:') || log.includes('GRANTED')) div.classList.add('log-detected');
                else if (log.includes('ALERT:') || log.includes('VIOLATION:') || log.includes('ERROR:') || log.includes('DENIED')) div.classList.add('log-alert');
                else if (log.includes('WARNING:')) div.classList.add('log-warning');
                else if (log.includes('DATABASE:') || log.includes('REGISTERED:')) div.classList.add('log-database');
                else div.classList.add('log-system');
                
                div.innerText = log;
                container.appendChild(div);
            });
            
            container.scrollTop = container.scrollHeight;
        } catch (e) {
            console.error("Logs fetch error:", e);
        }
    }

    // --- DIRECTORY LOGIC & SEARCH ---
    let allDirectoryUsers = [];

    async function loadDirectory() {
        try {
            const res = await fetch('/api/directory');
            const data = await res.json();
            allDirectoryUsers = data.users || [];
            
            const countElem = document.getElementById('nav-user-count');
            if (countElem) countElem.innerText = allDirectoryUsers.length;

            renderDirectoryTable(allDirectoryUsers);
        } catch (e) {
            console.error("Load directory error:", e);
        }
    }

    function renderDirectoryTable(users) {
        const tbody = document.getElementById('directory-tbody');
        if (!tbody) return;
        tbody.innerHTML = '';
        
        if (users.length === 0) {
            tbody.innerHTML = '<tr><td colspan="4" style="text-align:center; padding:32px; color:var(--text-dim);"><i class="fa-solid fa-folder-open" style="font-size:24px; margin-bottom:8px; display:block;"></i>No enrolled biometric profiles found.</td></tr>';
            return;
        }

        users.forEach(name => {
            const initials = name.split(' ').map(n => n[0]).join('').toUpperCase().substring(0, 2) || 'ID';
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>
                    <div class="user-identity-cell">
                        <div class="user-avatar-circle">${initials}</div>
                        <div>
                            <strong>${name}</strong>
                            <div style="font-family:var(--font-mono); font-size:10px; color:var(--text-dim);">ID: SFace-128D</div>
                        </div>
                    </div>
                </td>
                <td><span class="status-badge-pill status-ok"><i class="fa-solid fa-check"></i> 128-D EMBEDDING</span></td>
                <td><span class="status-badge-pill status-ok"><i class="fa-solid fa-mask"></i> PERIOCULAR READY</span></td>
                <td><button class="btn-table-delete" onclick="deleteUser('${name}')"><i class="fa-solid fa-trash"></i> Revoke</button></td>
            `;
            tbody.appendChild(tr);
        });
    }

    const searchInput = document.getElementById('directory-search');
    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            const query = e.target.value.toLowerCase().trim();
            const filtered = allDirectoryUsers.filter(u => u.toLowerCase().includes(query));
            renderDirectoryTable(filtered);
        });
    }

    window.deleteUser = async function(name) {
        if (!confirm(`Are you sure you want to revoke biometric credential for '${name}'?`)) return;
        try {
            await fetch('/api/delete_user', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ name })
            });
            loadDirectory();
        } catch (e) {
            console.error("Delete user error:", e);
        }
    };

    // Initial load of directory count
    loadDirectory();

    // --- SETTINGS & CALIBRATION LOGIC ---
    const strictToggle = document.getElementById('strict-mode-toggle');
    if (strictToggle) {
        strictToggle.addEventListener('change', async (e) => {
            await fetch('/api/settings', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ strict_mode: e.target.checked })
            });
        });
    }

    const threshSlider = document.getElementById('threshold-slider');
    const threshVal = document.getElementById('threshold-val');
    const presetBadges = document.querySelectorAll('.preset-badge');

    if (threshSlider) {
        threshSlider.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value).toFixed(3);
            if (threshVal) threshVal.innerText = val;
        });

        threshSlider.addEventListener('change', async (e) => {
            const val = parseFloat(e.target.value).toFixed(3);
            await updateThreshold(val);
        });
    }

    presetBadges.forEach(badge => {
        badge.addEventListener('click', async () => {
            presetBadges.forEach(b => b.classList.remove('active'));
            badge.classList.add('active');
            const targetThresh = badge.getAttribute('data-threshold');
            if (threshSlider) threshSlider.value = targetThresh;
            if (threshVal) threshVal.innerText = targetThresh;
            await updateThreshold(targetThresh);
        });
    });

    async function updateThreshold(val) {
        try {
            await fetch('/api/settings', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ threshold: val })
            });
        } catch (e) {
            console.error("Threshold update error:", e);
        }
    }

    const intervalSlider = document.getElementById('interval-slider');
    const intervalVal = document.getElementById('interval-val');

    if (intervalSlider) {
        intervalSlider.addEventListener('input', (e) => {
            if (intervalVal) intervalVal.innerText = `${e.target.value} frames`;
        });

        intervalSlider.addEventListener('change', async (e) => {
            const val = e.target.value;
            await fetch('/api/settings', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ interval: val })
            });
        });
    }

    const btnWipe = document.getElementById('btn-wipe-db');
    if (btnWipe) {
        btnWipe.addEventListener('click', async () => {
            if (confirm("CRITICAL SECURITY ACTION: Permanently wipe all biometric vector records in db.json?")) {
                await fetch('/api/wipe_db', { method: 'POST' });
                alert("Biometric database wiped successfully.");
                loadDirectory();
            }
        });
    }
});
