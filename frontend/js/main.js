document.addEventListener('DOMContentLoaded', () => {

    // ── Navigation ──
    const navLinks = document.querySelectorAll('.nav-links li');
    const views = document.querySelectorAll('.view');

    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');
            const target = link.getAttribute('data-target');
            views.forEach(v => {
                v.classList.remove('active-view');
                if (v.id === target) v.classList.add('active-view');
            });
            if (target === 'users') loadUsers();
            if (target === 'logs') { loadDepartmentFilter(); loadLogs(); }
            if (target === 'checkin') loadCheckinPage();
        });
    });

    // ── Auto-refresh dashboard ──
    loadDashboardStats();
    loadActivity();
    loadCameraStatus();
    loadCameraList();
    setInterval(loadDashboardStats, 5000);
    setInterval(loadActivity, 1500);
    setInterval(loadCameraStatus, 5000);
    autoScanLoop();

    // ── Recover the MJPEG stream if the connection drops ──
    const stream = document.getElementById('video-stream');
    stream.addEventListener('error', () => setTimeout(reloadVideoStream, 3000));

    // ── Registration: preview picked files ──
    document.getElementById('reg-image').addEventListener('change', renderRegPreview);

    // ── Registration form (multi-image upload) ──
    document.getElementById('register-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        const name = document.getElementById('reg-name').value.trim();
        const department = document.getElementById('reg-department').value.trim();
        const fileInput = document.getElementById('reg-image');
        const msgDiv = document.getElementById('reg-message');
        const progressDiv = document.getElementById('reg-progress');
        const progressFill = document.getElementById('reg-progress-fill');
        const progressText = document.getElementById('reg-progress-text');
        const submitBtn = document.getElementById('reg-submit-btn');

        const totalPhotos = capturedShots.length + fileInput.files.length;
        if (!name) {
            msgDiv.textContent = '❌ Please enter a name.';
            msgDiv.style.color = 'var(--danger)';
            return;
        }
        if (totalPhotos === 0) {
            msgDiv.textContent = '❌ Capture from the camera or choose at least one photo.';
            msgDiv.style.color = 'var(--danger)';
            return;
        }

        const formData = new FormData();
        capturedShots.forEach((blob, i) => {
            formData.append('files', blob, `capture_${i + 1}.jpg`);
        });
        for (let i = 0; i < fileInput.files.length; i++) {
            formData.append('files', fileInput.files[i]);
        }

        submitBtn.disabled = true;
        progressDiv.style.display = 'block';
        progressFill.style.width = '30%';
        progressText.textContent = `Analyzing ${totalPhotos} photo(s) and generating face vectors...`;
        msgDiv.textContent = '';

        try {
            progressFill.style.width = '60%';
            const res = await fetch(`/api/register?name=${encodeURIComponent(name)}&department=${encodeURIComponent(department)}`, {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            progressFill.style.width = '100%';

            if (res.ok) {
                progressText.textContent = 'Done!';
                msgDiv.textContent = '✅ ' + data.message;
                msgDiv.style.color = 'var(--success)';
                showToast(`Registered: ${name}`, 'success');
                setTimeout(() => {
                    closeRegisterModal();
                    if (document.getElementById('users').classList.contains('active-view')) loadUsers();
                }, 2000);
            } else {
                msgDiv.textContent = '❌ ' + (data.detail || 'Registration failed');
                msgDiv.style.color = 'var(--danger)';
                progressDiv.style.display = 'none';
            }
        } catch (err) {
            msgDiv.textContent = '❌ Error connecting to server';
            msgDiv.style.color = 'var(--danger)';
            progressDiv.style.display = 'none';
        }
        submitBtn.disabled = false;
    });
});


// ═══════════════ Modals ═══════════════

function openRegisterModal() {
    document.getElementById('register-modal').style.display = 'flex';
    document.getElementById('reg-message').textContent = '';
    document.getElementById('reg-progress').style.display = 'none';
    document.getElementById('register-form').reset();
    clearCaptures();
}
function closeRegisterModal() {
    document.getElementById('register-modal').style.display = 'none';
}
function openImageModal(src) {
    document.getElementById('image-modal-img').src = src;
    document.getElementById('image-modal').style.display = 'flex';
}
function closeImageModal() {
    document.getElementById('image-modal').style.display = 'none';
}


// ═══════════════ Camera ═══════════════

function reloadVideoStream() {
    const img = document.getElementById('video-stream');
    // Cache-buster forces the browser to re-open the MJPEG connection.
    img.src = '/api/video_feed?t=' + Date.now();
}

async function loadCameraStatus() {
    const badge = document.getElementById('camera-badge');
    const errorEl = document.getElementById('camera-error');
    try {
        const res = await fetch('/api/camera/status');
        const s = await res.json();
        if (s.connected) {
            badge.classList.remove('offline');
            badge.innerHTML = '<i class="fa-solid fa-circle"></i> LIVE';
            errorEl.style.display = 'none';
        } else {
            badge.classList.add('offline');
            badge.innerHTML = '<i class="fa-solid fa-circle"></i> NO CAMERA';
            errorEl.textContent = s.error || `Connecting to camera ${s.index}...`;
            errorEl.style.display = 'block';
        }
    } catch (e) {
        badge.classList.add('offline');
        badge.innerHTML = '<i class="fa-solid fa-circle"></i> OFFLINE';
        errorEl.textContent = 'Cannot reach the AIFACE server.';
        errorEl.style.display = 'block';
    }
}

async function loadCameraList(showMsg = false) {
    const select = document.getElementById('camera-select');
    select.disabled = true;
    if (showMsg) showToast('กำลังค้นหากล้อง...', 'info');
    try {
        const res = await fetch('/api/camera/list');
        const data = await res.json();
        const cams = data.cameras || [];
        // Keep the selected camera listed even if it can't be probed right now.
        if (!cams.some(c => c.index === data.current)) {
            cams.push({ index: data.current, name: `Camera ${data.current}` });
            cams.sort((a, b) => a.index - b.index);
        }
        select.innerHTML = cams.map(c =>
            `<option value="${c.index}">${c.name}</option>`
        ).join('');
        select.value = data.current;
        if (showMsg) showToast(`พบกล้อง ${data.cameras.length} ตัว`, 'success');
    } catch (e) {
        select.innerHTML = '<option value="">ไม่พบกล้อง</option>';
    }
    select.disabled = false;
}

async function selectCamera(index) {
    if (index === '') return;
    showToast(`กำลังเปลี่ยนเป็น Camera ${index}...`, 'info');
    try {
        const res = await fetch('/api/camera/select', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ index: parseInt(index, 10) })
        });
        if (!res.ok) throw new Error();
        setTimeout(() => { reloadVideoStream(); loadCameraStatus(); }, 1500);
    } catch (e) {
        showToast('เปลี่ยนกล้องไม่สำเร็จ', 'error');
        loadCameraList();
    }
}

async function restartCamera() {
    showToast('Reconnecting camera...', 'info');
    try {
        await fetch('/api/camera/restart', { method: 'POST' });
        setTimeout(() => { reloadVideoStream(); loadCameraStatus(); }, 1500);
    } catch (e) {
        showToast('Failed to reconnect camera', 'error');
    }
}


// ═══════════════ Registration captures ═══════════════

let capturedShots = [];

function updateRegCount() {
    const picked = document.getElementById('reg-image').files.length;
    const total = capturedShots.length + picked;
    document.getElementById('reg-count').textContent =
        `${total} photo(s) ready` + (capturedShots.length ? ` (${capturedShots.length} from camera)` : '');
}

function renderRegPreview() {
    const preview = document.getElementById('reg-preview');
    preview.innerHTML = '';

    capturedShots.forEach((blob, i) => {
        const img = document.createElement('img');
        img.src = URL.createObjectURL(blob);
        img.alt = `Capture ${i + 1}`;
        img.title = 'Captured from camera';
        preview.appendChild(img);
    });

    const files = document.getElementById('reg-image').files;
    for (let i = 0; i < files.length; i++) {
        const reader = new FileReader();
        reader.onload = (ev) => {
            const img = document.createElement('img');
            img.src = ev.target.result;
            img.alt = `Photo ${i + 1}`;
            preview.appendChild(img);
        };
        reader.readAsDataURL(files[i]);
    }
    updateRegCount();
}

async function captureFromCamera(silent = false) {
    try {
        const res = await fetch('/api/snapshot?t=' + Date.now());
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            showToast(err.detail || 'Camera not ready', 'error');
            return false;
        }
        capturedShots.push(await res.blob());
        renderRegPreview();
        if (!silent) showToast(`Captured photo ${capturedShots.length}`, 'success');
        return true;
    } catch (e) {
        showToast('Could not capture from camera', 'error');
        return false;
    }
}

async function captureBurst() {
    const btn = document.getElementById('btn-burst');
    btn.disabled = true;
    const original = btn.innerHTML;
    for (let i = 1; i <= 5; i++) {
        btn.innerHTML = `<i class="fa-solid fa-camera"></i> Capturing ${i}/5...`;
        const ok = await captureFromCamera(true);
        if (!ok) break;
        // Pause so the user can change angle between shots.
        await new Promise(r => setTimeout(r, 900));
    }
    btn.innerHTML = original;
    btn.disabled = false;
    showToast(`${capturedShots.length} photo(s) ready`, 'success');
}

function clearCaptures() {
    capturedShots = [];
    document.getElementById('reg-image').value = '';
    renderRegPreview();
}


// ═══════════════ Data Loading ═══════════════

let lastActivityKey = undefined;
let lastActivityListKey = null;
// The scan response shows its own overlay (with the snapshot). While a scan
// request is in flight, or its overlay is still playing, the activity poll
// must not play a second one for the same event.
let scanInFlight = false;
let lastScanOverlayAt = 0;

function showScanResult(data) {
    lastScanOverlayAt = Date.now();
    showScanSnapshot(data.image, data.face_found);
    triggerAccessOverlay({ status: data.granted ? 'Access Granted' : 'Access Denied', name: data.name, confidence: data.granted ? data.confidence : null });
    loadActivity();
    loadDashboardStats();
}

async function scanEntry() {
    const btn = document.getElementById('btn-enter-scan');
    if (btn.disabled) return;
    btn.disabled = true;
    const original = btn.innerHTML;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> กำลังสแกน...';

    scanInFlight = true;
    try {
        const res = await fetch('/api/scan', { method: 'POST' });
        const data = await res.json();
        if (res.ok) {
            showScanResult(data);
        } else {
            showToast(data.detail || 'สแกนไม่สำเร็จ กล้องไม่พร้อม', 'error');
        }
    } catch (e) {
        showToast('เชื่อมต่อเซิร์ฟเวอร์ไม่ได้', 'error');
    }
    scanInFlight = false;

    // Re-enable only once the overlay has finished playing — entry is a
    // deliberate one-shot action, not a continuous loop, so the next person
    // must press the button again.
    setTimeout(() => {
        btn.disabled = false;
        btn.innerHTML = original;
    }, 3300);
}

// ── Auto scan: poll /api/scan?auto=1 while the dashboard is on screen. The
// server only runs the face engine once a face is in view ("idle" otherwise)
// and stays silent ("skipped") until a good-quality face isn't inside its
// cooldown, so this only fires once per person/visit. With nobody in view
// the poll slows down to rest.
const AUTO_SCAN_INTERVAL = 400;
const AUTO_SCAN_IDLE_INTERVAL = 1000;

async function autoScanTick() {
    const btn = document.getElementById('btn-enter-scan');
    const onDashboard = document.getElementById('dashboard').classList.contains('active-view');
    if (document.hidden || !onDashboard || btn.disabled) return AUTO_SCAN_IDLE_INTERVAL;

    scanInFlight = true;
    try {
        const res = await fetch('/api/scan?auto=1', { method: 'POST' });
        if (!res.ok) return AUTO_SCAN_IDLE_INTERVAL;
        const data = await res.json();
        if (data.status === 'idle') return AUTO_SCAN_IDLE_INTERVAL;
        if (data.status !== 'success') return AUTO_SCAN_INTERVAL;

        // Already logged server-side, so always show it — even if the manual
        // button happens to be busy.
        btn.disabled = true;
        showScanResult(data);
        setTimeout(() => { btn.disabled = false; }, 3300);
    } catch (e) {
        /* server busy/restarting — next tick retries */
    } finally {
        scanInFlight = false;
    }
    return AUTO_SCAN_INTERVAL;
}

async function autoScanLoop() {
    const delay = await autoScanTick();
    setTimeout(autoScanLoop, delay);
}

function showScanSnapshot(image, faceFound) {
    const snap = document.getElementById('scan-snapshot');
    const overlay = document.getElementById('access-overlay');
    clearTimeout(snap._hideTimer);
    if (!image) { snap.classList.remove('show'); overlay.classList.remove('with-box'); return; }
    snap.src = image;
    snap.classList.add('show');
    overlay.classList.toggle('with-box', !!faceFound);
    snap._hideTimer = setTimeout(() => snap.classList.remove('show'), 3200);
}

function triggerAccessOverlay(item) {
    const overlay = document.getElementById('access-overlay');
    const granted = item.status.includes('Granted');

    overlay.classList.remove('show', 'granted', 'denied');
    void overlay.offsetWidth; // restart the CSS animation

    overlay.querySelector('.access-icon i').className = granted
        ? 'fa-solid fa-circle-check' : 'fa-solid fa-circle-xmark';
    overlay.querySelector('.access-title').textContent = granted ? 'ACCESS GRANTED' : 'ACCESS DENIED';
    overlay.querySelector('.access-name').textContent = (item.name && item.name !== 'Unknown') ? item.name : '';
    overlay.querySelector('.access-confidence').textContent =
        (granted && item.confidence != null) ? `ความแม่นยำ ${Number(item.confidence).toFixed(1)}%` : '';
    overlay.querySelector('.access-sub').textContent = granted
        ? 'เชิญเข้าห้องได้เลย · You may enter'
        : 'ไม่รู้จักใบหน้านี้ · Face not recognized';

    overlay.classList.add(granted ? 'granted' : 'denied', 'show');
    clearTimeout(overlay._hideTimer);
    overlay._hideTimer = setTimeout(() => overlay.classList.remove('show'), 3200);
}

async function loadActivity() {
    if (!document.getElementById('dashboard').classList.contains('active-view')) return;
    try {
        const res = await fetch('/api/activity');
        const items = await res.json();
        const container = document.getElementById('activity-list');

        if (items && items.length > 0) {
            const latest = items[0];
            const key = `${latest.time}|${latest.name}|${latest.status}|${latest.confidence}`;
            // Skip the very first load so an old event already on screen
            // doesn't replay the animation the moment the dashboard opens.
            if (lastActivityKey !== undefined && key !== lastActivityKey) {
                if (!scanInFlight && Date.now() - lastScanOverlayAt > 3500) {
                    triggerAccessOverlay({ ...latest, confidence: latest.status.includes('Granted') ? latest.confidence : null });
                }
            }
            lastActivityKey = key;
        }

        // Rebuild only when the list changed — re-rendering it every poll
        // reloaded every thumbnail and replayed every row's slide-in animation.
        const listKey = JSON.stringify(items);
        if (listKey === lastActivityListKey) return;
        lastActivityListKey = listKey;

        if (!items || items.length === 0) {
            container.innerHTML = '<p class="no-activity">Waiting for detections...</p>';
            return;
        }
        container.innerHTML = items.map(item => {
            const badgeClass = item.status.includes('Granted') ? 'granted' : 'denied';
            const thumbHtml = item.image
                ? `<img class="activity-thumb" src="/api/captured/${item.image}" alt="${item.name}" onclick="openImageModal('/api/captured/${item.image}')" />`
                : `<div class="activity-thumb-placeholder"><i class="fa-solid fa-user"></i></div>`;
            return `<div class="activity-item">
                ${thumbHtml}
                <div class="activity-info">
                    <div class="activity-name">${item.name}</div>
                    <div class="activity-meta">${item.time} · ${item.confidence}%</div>
                </div>
                <span class="activity-badge ${badgeClass}">${item.status.includes('Granted') ? 'GRANTED' : 'DENIED'}</span>
            </div>`;
        }).join('');
    } catch (e) { console.error('loadActivity:', e); }
}

async function loadUsers() {
    try {
        const res = await fetch('/api/users');
        const users = await res.json();
        const tbody = document.querySelector('#users-table tbody');
        tbody.innerHTML = users.map(u => `
            <tr>
                <td>${u.id}</td>
                <td><strong>${u.name}</strong></td>
                <td>${u.department ? `<span class="encoding-count">${u.department}</span>` : '<span style="color:var(--text-muted)">—</span>'}</td>
                <td><span class="encoding-count">${u.encoding_count} vectors</span></td>
                <td>${u.registered_at}</td>
                <td><button class="btn-danger" onclick="deleteUser(${u.id})"><i class="fa-solid fa-trash"></i> Delete</button></td>
            </tr>
        `).join('');
        document.getElementById('stat-users').textContent = users.length;
    } catch (e) { console.error('loadUsers:', e); }
}

async function deleteUser(id) {
    if (!confirm("Delete this user? Their face data will be removed from recognition.")) return;
    try {
        await fetch(`/api/users/${id}`, { method: 'DELETE' });
        loadUsers();
        showToast('User deleted', 'success');
    } catch (e) { showToast('Failed to delete', 'error'); }
}

async function loadDepartmentFilter() {
    try {
        const res = await fetch('/api/departments');
        const departments = await res.json();
        const select = document.getElementById('logs-department-filter');
        const current = select.value;
        select.innerHTML = '<option value="">All Departments</option>' +
            departments.map(d => `<option value="${d}">${d}</option>`).join('');
        select.value = current;
    } catch (e) { console.error('loadDepartmentFilter:', e); }
}

let logsSearchTimer = null;
function loadLogsDebounced() {
    clearTimeout(logsSearchTimer);
    logsSearchTimer = setTimeout(loadLogs, 300);
}

function clearLogSearch() {
    document.getElementById('logs-name-search').value = '';
    document.getElementById('logs-date-from').value = '';
    document.getElementById('logs-date-to').value = '';
    loadLogs();
}

async function loadLogs() {
    try {
        const params = new URLSearchParams();
        const department = document.getElementById('logs-department-filter')?.value || '';
        const name = document.getElementById('logs-name-search')?.value.trim() || '';
        const dateFrom = document.getElementById('logs-date-from')?.value || '';
        const dateTo = document.getElementById('logs-date-to')?.value || '';
        if (department) params.set('department', department);
        if (name) params.set('name', name);
        if (dateFrom) params.set('date_from', dateFrom);
        if (dateTo) params.set('date_to', dateTo);
        const qs = params.toString();
        const res = await fetch('/api/logs' + (qs ? `?${qs}` : ''));
        const logs = await res.json();
        const grid = document.getElementById('logs-grid');

        if (logs.length === 0) {
            grid.innerHTML = (name || dateFrom || dateTo)
                ? '<p class="no-activity" style="grid-column:1/-1;padding:3rem;">ไม่พบประวัติที่ตรงกับการค้นหา</p>'
                : '<p class="no-activity" style="grid-column:1/-1;padding:3rem;">No access logs yet. The AI will record faces automatically.</p>';
            return;
        }

        let unknownCount = 0;
        const today = new Date().toISOString().split('T')[0];

        grid.innerHTML = logs.map(l => {
            if (l.name === 'Unknown' && l.timestamp.startsWith(today)) unknownCount++;
            const badgeClass = l.status.includes('Granted') || l.status.includes('Clock') ? 'granted' : 'denied';
            const conf = l.confidence || 0;
            const confClass = conf >= 70 ? 'high' : conf >= 40 ? 'medium' : 'low';
            const imageHtml = l.image_path
                ? `<img class="log-card-image" src="/api/captured/${l.image_path}" alt="${l.name}" onclick="openImageModal('/api/captured/${l.image_path}')" />`
                : `<div class="log-card-placeholder"><i class="fa-solid fa-user-secret"></i></div>`;
            const typeBadge = l.check_type === 'IN' ? '<span class="status-badge granted">CLOCK IN</span>'
                : l.check_type === 'OUT' ? '<span class="status-badge denied">CLOCK OUT</span>' : '';
            const gpsHtml = (l.latitude != null && l.longitude != null)
                ? `<a class="log-card-gps" href="https://www.google.com/maps?q=${l.latitude},${l.longitude}" target="_blank" rel="noopener"><i class="fa-solid fa-location-dot"></i> Map</a>`
                : '';
            return `<div class="log-card">
                <button class="log-card-delete" onclick="deleteLog(${l.id})" title="Delete this log"><i class="fa-solid fa-xmark"></i></button>
                ${imageHtml}
                <div class="log-card-body">
                    <div class="log-card-name">${l.name}</div>
                    <div class="log-card-row">
                        <span class="status-badge ${badgeClass}">${l.status}</span>
                        <span class="log-card-confidence ${confClass}">${conf > 0 ? conf.toFixed(0) + '%' : '—'}</span>
                    </div>
                    <div class="log-card-row">
                        ${typeBadge}
                        ${l.department ? `<span class="log-card-dept">${l.department}</span>` : ''}
                    </div>
                    <div class="log-card-row">
                        <span class="log-card-time"><i class="fa-regular fa-clock"></i> ${l.timestamp}</span>
                        ${gpsHtml}
                    </div>
                </div>
            </div>`;
        }).join('');

        // A filtered list would skew the dashboard's "unknowns today" count.
        if (!qs) document.getElementById('stat-unknowns').textContent = unknownCount;
    } catch (e) { console.error('loadLogs:', e); }
}


// ═══════════════ Check-In (self-service attendance) ═══════════════

async function loadCheckinPage() {
    try {
        const res = await fetch('/api/users');
        const users = await res.json();
        const select = document.getElementById('checkin-user');
        const current = select.value;
        select.innerHTML = '<option value="">— Choose your name —</option>' +
            users.map(u => `<option value="${u.name}">${u.name}${u.department ? ' (' + u.department + ')' : ''}</option>`).join('');
        select.value = current;
    } catch (e) { console.error('loadCheckinPage:', e); }

    document.getElementById('checkin-user').onchange = loadCheckinLastStatus;
    loadCheckinLastStatus();
}

async function loadCheckinLastStatus() {
    const name = document.getElementById('checkin-user').value;
    const box = document.getElementById('checkin-last-status');
    if (!name) { box.style.display = 'none'; return; }
    try {
        const res = await fetch(`/api/checkin/status?name=${encodeURIComponent(name)}`);
        const data = await res.json();
        box.style.display = 'block';
        box.textContent = data.last_timestamp
            ? `Last: Clock ${data.last_check_type} at ${data.last_timestamp}`
            : 'No previous check-in/out on record.';
    } catch (e) { box.style.display = 'none'; }
}

function getGeolocation() {
    return new Promise((resolve) => {
        if (!navigator.geolocation) { resolve(null); return; }
        navigator.geolocation.getCurrentPosition(
            (pos) => resolve({ latitude: pos.coords.latitude, longitude: pos.coords.longitude }),
            () => resolve(null),
            { timeout: 5000 }
        );
    });
}

async function doCheckin(checkType) {
    const name = document.getElementById('checkin-user').value;
    const msgDiv = document.getElementById('checkin-message');
    const previewImg = document.getElementById('checkin-preview-img');
    const previewPlaceholder = document.getElementById('checkin-preview-placeholder');
    msgDiv.textContent = '';

    if (!name) {
        msgDiv.textContent = '❌ Please select your name first.';
        msgDiv.style.color = 'var(--danger)';
        return;
    }

    const btnIn = document.getElementById('btn-checkin-in');
    const btnOut = document.getElementById('btn-checkin-out');
    btnIn.disabled = true; btnOut.disabled = true;
    msgDiv.textContent = 'Scanning face...';
    msgDiv.style.color = 'var(--text-muted)';

    try {
        const [snapRes, geo] = await Promise.all([
            fetch('/api/snapshot?t=' + Date.now()),
            getGeolocation()
        ]);
        if (!snapRes.ok) {
            const err = await snapRes.json().catch(() => ({}));
            throw new Error(err.detail || 'Camera not ready');
        }
        const blob = await snapRes.blob();
        previewImg.src = URL.createObjectURL(blob);
        previewImg.style.display = 'block';
        previewPlaceholder.style.display = 'none';

        const formData = new FormData();
        formData.append('file', blob, 'checkin.jpg');
        let url = `/api/checkin?name=${encodeURIComponent(name)}&check_type=${checkType}`;
        if (geo) url += `&latitude=${geo.latitude}&longitude=${geo.longitude}`;

        const res = await fetch(url, { method: 'POST', body: formData });
        const data = await res.json();

        if (res.ok) {
            msgDiv.textContent = `✅ Clock ${checkType} confirmed for ${data.name} (${data.confidence}%)${geo ? '' : ' — GPS unavailable'}`;
            msgDiv.style.color = 'var(--success)';
            showToast(`Clock ${checkType}: ${data.name}`, 'success');
            loadCheckinLastStatus();
        } else {
            msgDiv.textContent = '❌ ' + (data.detail || 'Check-in failed');
            msgDiv.style.color = 'var(--danger)';
        }
    } catch (e) {
        msgDiv.textContent = '❌ ' + e.message;
        msgDiv.style.color = 'var(--danger)';
    }
    btnIn.disabled = false; btnOut.disabled = false;
}


// ═══════════════ Log Deletion ═══════════════

async function deleteLog(id) {
    try {
        await fetch(`/api/logs/${id}`, { method: 'DELETE' });
        loadLogs();
        showToast('Log deleted', 'success');
    } catch (e) { showToast('Failed to delete log', 'error'); }
}

async function deleteUnknownLogs() {
    if (!confirm("Delete ALL 'Unknown' (mismatched) logs and their images?")) return;
    try {
        const res = await fetch('/api/logs/bulk/unknown', { method: 'DELETE' });
        const data = await res.json();
        loadLogs();
        showToast(`Deleted ${data.deleted} unknown logs`, 'success');
    } catch (e) { showToast('Failed to delete', 'error'); }
}

async function deleteTodayLogs() {
    const today = new Date().toISOString().split('T')[0];
    if (!confirm(`Delete ALL logs for today (${today})?`)) return;
    try {
        const res = await fetch(`/api/logs/bulk/date/${today}`, { method: 'DELETE' });
        const data = await res.json();
        loadLogs();
        showToast(`Deleted ${data.deleted} logs for today`, 'success');
    } catch (e) { showToast('Failed to delete', 'error'); }
}


// ═══════════════ Utilities ═══════════════

async function loadDashboardStats() {
    if (!document.getElementById('dashboard').classList.contains('active-view')) return;
    try {
        const [usersRes, logsRes] = await Promise.all([fetch('/api/users'), fetch('/api/logs')]);
        const users = await usersRes.json();
        const logs = await logsRes.json();
        document.getElementById('stat-users').textContent = users.length;
        const today = new Date().toISOString().split('T')[0];
        document.getElementById('stat-unknowns').textContent = logs.filter(l => l.name === 'Unknown' && l.timestamp.startsWith(today)).length;
    } catch (e) { console.error('loadDashboardStats:', e); }
}

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.style.cssText = `
        position:fixed;bottom:20px;right:20px;z-index:200;
        padding:12px 20px;border-radius:10px;
        font-size:0.85rem;font-weight:500;color:white;
        animation:slideIn 0.3s ease;
        background:${type === 'success' ? 'rgba(34,197,94,0.9)' : type === 'error' ? 'rgba(239,68,68,0.9)' : 'rgba(99,102,241,0.9)'};
        backdrop-filter:blur(8px);box-shadow:0 8px 24px rgba(0,0,0,0.3);
    `;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => { toast.style.opacity = '0'; toast.style.transition = 'opacity 0.3s'; setTimeout(() => toast.remove(), 300); }, 3000);
}
