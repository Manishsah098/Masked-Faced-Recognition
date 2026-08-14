document.addEventListener('DOMContentLoaded', () => {
    // --- CONNECT TO SOCKET.IO SERVER ---
    const socket = io();
    const statusText = document.getElementById('system-status-text');
    const statusDot = document.getElementById('system-status-dot');

    socket.on('connect', () => {
        statusText.innerText = "System Online";
        statusDot.className = "dot active";
        statusDot.style.backgroundColor = '#2ea043';
    });

    socket.on('disconnect', () => {
        statusText.innerText = "System Disconnected";
        statusDot.className = "dot";
        statusDot.style.backgroundColor = '#f85149';
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
            document.getElementById(targetId).classList.remove('hidden-view');
            document.getElementById(targetId).classList.add('active-view');

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

    function initChart() {
        const ctx = document.getElementById('complianceChart').getContext('2d');
        complianceChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Masked', 'Unmasked', 'Violations'],
                datasets: [{
                    data: [0, 0, 0],
                    backgroundColor: ['#58a6ff', '#2ea043', '#f85149'],
                    borderColor: '#161b22',
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
                            color: '#8b949e',
                            font: { size: 9, family: 'Inter' },
                            boxWidth: 8
                        }
                    }
                },
                cutout: '65%'
            }
        });
    }

    function updateChartStats(name, mask, status_msg) {
        if (name === '-') return;
        
        if (status_msg.includes('VIOLATION') || status_msg.includes('BLOCKED')) {
            complianceStats.violations++;
        } else if (mask.includes('Masked')) {
            complianceStats.masked++;
        } else if (mask.includes('Unmasked')) {
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
    let lastVoiceTime = 0;

    function speak(text) {
        const now = Date.now();
        // Strict global cooldown of 5 seconds to let the browser speak the full sentence
        // without stuttering, clipping, or canceling itself due to detection noise
        if (now - lastVoiceTime < 5000) {
            return;
        }
        lastVoiceTime = now;
        
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.0;
        window.speechSynthesis.speak(utterance);
    }

    function handleVoiceAlerts(name, mask, status_msg) {
        if (name === '-') return;
        
        if (status_msg.includes('VIOLATION') || status_msg.includes('BLOCKED')) {
            speak(`Warning: Please wear a mask, ${name}`);
        } else if (name === 'Unknown') {
            speak("Access Denied: Unregistered user");
        } else {
            speak(`Access Granted: Welcome ${name}`);
        }
    }

    // --- BROWSER WEBCAM STREAMING (SOCKET.IO) ---
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('hidden-canvas');
    const ctx = canvas.getContext('2d');
    const feedImg = document.getElementById('video-feed');
    
    let isProcessing = false;
    let lastSendTime = 0;

    function startCamera() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            console.error("Camera API unavailable. Webcams require HTTPS or localhost.");
            statusText.innerText = "Camera Error: Requires HTTPS";
            statusDot.style.backgroundColor = '#f85149';
            return;
        }

        const primaryConstraints = {
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: "user"
            }
        };

        navigator.mediaDevices.getUserMedia(primaryConstraints)
            .then(onStreamSuccess)
            .catch(err => {
                console.warn("Primary camera constraints failed, attempting fallback:", err);
                navigator.mediaDevices.getUserMedia({ video: true })
                    .then(onStreamSuccess)
                    .catch(err2 => {
                        console.error("Camera access failed completely:", err2);
                        statusText.innerText = "Camera Access Error/Denied";
                        statusDot.style.backgroundColor = '#f85149';
                    });
            });
    }

    function onStreamSuccess(stream) {
        video.srcObject = stream;
        video.muted = true;
        video.playsInline = true;
        const playPromise = video.play();
        if (playPromise !== undefined) {
            playPromise.then(() => {
                requestAnimationFrame(captureLoop);
            }).catch(err => {
                console.warn("video.play() auto-play prevented:", err);
                requestAnimationFrame(captureLoop);
            });
        } else {
            requestAnimationFrame(captureLoop);
        }
    }

    startCamera();

    function captureLoop() {
        const now = Date.now();
        // Pull-based sync: Only send next frame if the previous one finished, or if 1s elapsed (timeout fallback)
        if ((!isProcessing || (now - lastSendTime > 1000)) && socket.connected) {
            sendFrame();
        }
        // Limit processing loop to ~15 FPS max to optimize CPU usage
        setTimeout(() => {
            requestAnimationFrame(captureLoop);
        }, 60);
    }

    function sendFrame() {
        if (!video.videoWidth || !video.videoHeight || video.paused || video.ended) return;
        isProcessing = true;
        lastSendTime = Date.now();
        
        canvas.width = 640;
        canvas.height = 480;
        ctx.drawImage(video, 0, 0, 640, 480);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.5); // Highly compressed JPEG for speed
        socket.emit('image', dataUrl);
    }

    // Handle annotated responses from server
    socket.on('response', (data) => {
        isProcessing = false; // Release lock to allow next frame capture
        
        // Render HUD frame
        feedImg.src = data.image;

        // Render variables
        document.getElementById('stat-name').innerText = data.state.name;
        document.getElementById('stat-mask').innerText = data.state.mask;
        document.getElementById('stat-score').innerText = data.state.score;
        
        const authText = document.getElementById('auth-status-text');
        const authBox = document.getElementById('auth-status-box');
        authText.innerText = data.state.status_msg;
        authText.style.color = data.state.status_color;
        authBox.style.borderLeftColor = data.state.status_color;

        // Voice alert trigger
        handleVoiceAlerts(data.state.name, data.state.mask, data.state.status_msg);

        // Chart stats trigger
        updateChartStats(data.state.name, data.state.mask, data.state.status_msg);
    });

    // --- REGISTRATION LOGIC ---
    const btnStartReg = document.getElementById('btn-start-reg');
    let regPollInterval = null;

    btnStartReg.addEventListener('click', async () => {
        const name = document.getElementById('reg-name').value;
        if (!name) return alert("Please enter a name first.");

        btnStartReg.disabled = true;
        btnStartReg.innerText = "Acquiring Biometrics...";

        try {
            await fetch('/api/register', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ name })
            });

            if (regPollInterval) clearInterval(regPollInterval);
            regPollInterval = setInterval(pollRegistrationStatus, 500);
        } catch (e) {
            alert("Registration request failed.");
            btnStartReg.disabled = false;
        }
    });

    async function pollRegistrationStatus() {
        try {
            const res = await fetch('/api/register_status');
            const data = await res.json();
            
            document.getElementById('reg-status-text').innerText = data.status_text;
            document.getElementById('reg-progress-bar').style.width = data.progress + "%";

            if (!data.is_registering && data.progress === 0 && data.status_text !== "ALERT: Remove mask to register!") {
                clearInterval(regPollInterval);
                btnStartReg.disabled = false;
                btnStartReg.innerText = "Start Profile Acquisition";
                document.getElementById('reg-name').value = '';
            }
        } catch (e) {
            console.error(e);
        }
    }

    // --- LOGS LOGIC ---
    let logsInterval = setInterval(fetchLogs, 2000);

    async function fetchLogs() {
        if (!document.getElementById('logs-view').classList.contains('active-view')) return;
        try {
            const res = await fetch('/api/logs');
            const data = await res.json();
            
            const container = document.getElementById('logs-container');
            container.innerHTML = '';
            
            data.logs.forEach(log => {
                const div = document.createElement('div');
                div.className = 'log-entry';
                
                if (log.includes('DETECTED:')) div.classList.add('log-detected');
                else if (log.includes('ALERT:') || log.includes('VIOLATION:') || log.includes('ERROR:')) div.classList.add('log-alert');
                else if (log.includes('WARNING:')) div.classList.add('log-warning');
                else if (log.includes('DATABASE:')) div.classList.add('log-database');
                else div.classList.add('log-system');
                
                div.innerText = log;
                container.appendChild(div);
            });
            
            container.scrollTop = container.scrollHeight;
        } catch (e) {
            console.error("Logs error", e);
        }
    }

    // --- DIRECTORY LOGIC ---
    async function loadDirectory() {
        try {
            const res = await fetch('/api/directory');
            const data = await res.json();
            
            const tbody = document.getElementById('directory-tbody');
            tbody.innerHTML = '';
            
            if (data.users.length === 0) {
                tbody.innerHTML = '<tr><td colspan="4" style="text-align:center; padding:30px;">No users enrolled.</td></tr>';
                return;
            }

            data.users.forEach(name => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><strong>${name}</strong></td>
                    <td><span class="status-badge status-active">● ACTIVE</span></td>
                    <td><span class="status-badge status-adaptive">● ADAPTIVE MASKED</span></td>
                    <td><button class="btn-small" onclick="deleteUser('${name}')">Delete</button></td>
                `;
                tbody.appendChild(tr);
            });
        } catch (e) {
            console.error(e);
        }
    }

    window.deleteUser = async function(name) {
        if (!confirm(`Delete profile '${name}'?`)) return;
        try {
            await fetch('/api/delete_user', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ name })
            });
            loadDirectory();
        } catch (e) {
            console.error(e);
        }
    }

    // --- SETTINGS LOGIC ---
    const strictToggle = document.getElementById('strict-mode-toggle');
    strictToggle.addEventListener('change', async (e) => {
        await fetch('/api/settings', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ strict_mode: e.target.checked })
        });
    });

    const threshSlider = document.getElementById('threshold-slider');
    threshSlider.addEventListener('change', async (e) => {
        const val = parseFloat(e.target.value).toFixed(3);
        document.getElementById('threshold-val').innerText = val;
        await fetch('/api/settings', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ threshold: val })
        });
    });

    const intervalSlider = document.getElementById('interval-slider');
    intervalSlider.addEventListener('change', async (e) => {
        const val = e.target.value;
        document.getElementById('interval-val').innerText = `${val} frames`;
        await fetch('/api/settings', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ interval: val })
        });
    });

    const btnWipe = document.getElementById('btn-wipe-db');
    btnWipe.addEventListener('click', async () => {
        if (confirm("CRITICAL WARNING: Wipe entire database?")) {
            await fetch('/api/wipe_db', { method: 'POST' });
            alert("Database wiped.");
        }
    });

});
