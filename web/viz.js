const previewCanvas = document.getElementById('preview');
const previewCtx = previewCanvas.getContext('2d');
const canvas = document.getElementById('timeline');
const ctx = canvas.getContext('2d');
const sidebar = document.getElementById('sidebar');
const btnStart = document.getElementById('btnStart');
const btnStop = document.getElementById('btnStop');
const recStatus = document.getElementById('recStatus');

let ws = null;
let students = new Map(); // track_id -> {id, state, last_seen, img, history: [{ts, state}]}
let windowSize = 60; // seconds
let now = Date.now() / 1000;

// Colors
const COLORS = {
    'awake': '#4CAF50',
    'drowsy': '#FF9800',
    'down': '#F44336',
    'unknown': '#9E9E9E'
};

function init() {
    resize();
    window.addEventListener('resize', resize);
    connect();
    requestAnimationFrame(loop);

    document.getElementById('windowSel').onchange = e => windowSize = parseInt(e.target.value);
    
    btnStart.onclick = async () => {
        try {
            const res = await fetch('/api/session/start', {method: 'POST'});
            if(res.ok) {
                setRecording(true);
                students.clear();
                sidebar.innerHTML = '';
            }
        } catch(e) { console.error(e); }
    };

    btnStop.onclick = async () => {
        try {
            const res = await fetch('/api/session/stop', {method: 'POST'});
            if(res.ok) setRecording(false);
        } catch(e) { console.error(e); }
    };
    
    // Check initial status
    fetch('/api/session/status').then(r=>r.json()).then(d => {
        if(d.is_recording) setRecording(true);
    });
}

function setRecording(isRec) {
    btnStart.disabled = isRec;
    btnStop.disabled = !isRec;
    recStatus.style.display = isRec ? 'inline' : 'none';
}

function resize() {
    // preview: take previewWrap size
    const previewWrap = document.getElementById('previewWrap');
    const timelineWrap = document.getElementById('timelineWrap');
    const pw = previewWrap.clientWidth;
    const ph = previewWrap.clientHeight;
    previewCanvas.width = pw;
    previewCanvas.height = ph;

    const tw = timelineWrap.clientWidth;
    const th = timelineWrap.clientHeight;
    canvas.width = tw;
    canvas.height = th;
}

function connect() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.onopen = () => document.getElementById('status').innerText = 'connected';
    ws.onclose = () => {
        document.getElementById('status').innerText = 'disconnected';
        setTimeout(connect, 2000);
    };
    ws.onmessage = msg => {
        try {
            const data = JSON.parse(msg.data);
            if (data.type === 'batch' && Array.isArray(data.events)) {
                data.events.forEach(e => processEvent(e));
            } else {
                processEvent(data);
            }
        } catch(e) { console.warn(e); }
    };
}

function processEvent(data) {
    if (data.type === 'frame_data') {
        handleFrameData(data);
        // draw preview image immediately if provided
        if (data.image_base64) drawPreviewImage(data.image_base64, data.faces);
    }
}

function handleFrameData(data) {
    now = data.ts;
    
    // Update students
    data.faces.forEach(face => {
        const tid = face.track_id;
        if (!students.has(tid)) {
            students.set(tid, {
                id: tid,
                state: face.state,
                last_seen: now,
                img: null,
                history: []
            });
            createStudentCard(tid);
        }
        
        const s = students.get(tid);
        s.state = face.state;
        s.last_seen = now;
        s.history.push({ts: now, state: face.state});
        
        // Keep history manageable
        if(s.history.length > 1000) s.history.shift();
        
        // Update DOM
        updateStudentCard(s, data.image_base64);
    });
}

function drawPreviewImage(b64, faces) {
    const img = new Image();
    img.onload = () => {
        // draw image to preview canvas
        previewCtx.clearRect(0,0,previewCanvas.width, previewCanvas.height);
        // fit image preserving aspect
        const iw = img.width, ih = img.height;
        const cw = previewCanvas.width, ch = previewCanvas.height;
        // compute scale
        const scale = Math.min(cw/iw, ch/ih);
        const dw = iw * scale, dh = ih * scale;
        const dx = (cw - dw)/2, dy = (ch - dh)/2;
        previewCtx.drawImage(img, 0, 0, iw, ih, dx, dy, dw, dh);

        // overlay faces
        if (Array.isArray(faces)) {
            faces.forEach(f => {
                // try to find bbox
                let bbox = null;
                if (f.bbox) bbox = f.bbox;
                else if (f.box) bbox = f.box;
                else if (f.rect) bbox = f.rect;
                else if (f.bounding_box) bbox = f.bounding_box;

                if (!bbox) return;

                // bbox can be dict or array
                let x=0,y=0,w=0,h=0;
                if (Array.isArray(bbox) || bbox instanceof Array) {
                    [x,y,w,h] = bbox;
                } else if (typeof bbox === 'object') {
                    x = bbox.x ?? bbox.left ?? bbox[0] ?? 0;
                    y = bbox.y ?? bbox.top ?? bbox[1] ?? 0;
                    w = bbox.w ?? bbox.width ?? bbox[2] ?? 0;
                    h = bbox.h ?? bbox.height ?? bbox[3] ?? 0;
                }

                // determine if normalized (0..1)
                let norm = false;
                if (x <= 1 && y <=1 && w <=1 && h <=1) norm = true;

                let drawX = x, drawY = y, drawW = w, drawH = h;
                if (norm) {
                    drawX = dx + x * dw;
                    drawY = dy + y * dh;
                    drawW = w * dw;
                    drawH = h * dh;
                } else {
                    // assume bbox in original image pixel coords; need to scale
                    const sx = scale; // mapping image->canvas
                    drawX = dx + x * sx;
                    drawY = dy + y * sx;
                    drawW = w * sx;
                    drawH = h * sx;
                }

                // draw rectangle and label
                previewCtx.strokeStyle = 'lime';
                previewCtx.lineWidth = 2;
                previewCtx.strokeRect(drawX, drawY, drawW, drawH);

                const state = (f.state || '').toUpperCase() || '';
                if (state) {
                    previewCtx.fillStyle = 'rgba(0,0,0,0.6)';
                    previewCtx.fillRect(drawX, drawY - 18, previewCtx.measureText(state).width + 8, 18);
                    previewCtx.fillStyle = '#fff';
                    previewCtx.font = '14px Arial';
                    previewCtx.fillText(state, drawX + 4, drawY - 4);
                }
            });
        }
    };
    img.src = 'data:image/jpeg;base64,' + b64;
}

function createStudentCard(tid) {
    const div = document.createElement('div');
    div.className = 'student-card';
    div.id = `student-${tid}`;
    div.innerHTML = `
        <div class="student-img-wrap" style="width:40px;height:40px;background:#eee;border-radius:50%;overflow:hidden">
            <img id="img-${tid}" style="width:100%;height:100%;object-fit:cover;display:none">
        </div>
        <div class="student-info">
            <div>ID: ${tid}</div>
            <div class="student-state" id="state-${tid}">Unknown</div>
        </div>
    `;
    sidebar.appendChild(div);
}

function updateStudentCard(s, b64Img) {
    const elState = document.getElementById(`state-${s.id}`);
    if(elState) {
        elState.innerText = s.state.toUpperCase();
        elState.className = `student-state state-${s.state}`;
    }
    
    if (b64Img) {
        const img = document.getElementById(`img-${s.id}`);
        if(img) {
            img.src = `data:image/jpeg;base64,${b64Img}`;
            img.style.display = 'block';
        }
    }
}

function loop() {
    render();
    requestAnimationFrame(loop);
}

function render() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const W = canvas.width;
    const H = canvas.height;
    const rowHeight = 60;
    const endTime = now;
    const startTime = endTime - windowSize;
    
    // Draw Grid
    ctx.strokeStyle = '#eee';
    ctx.beginPath();
    for(let t = Math.ceil(startTime); t <= endTime; t++) {
        if(t % 5 === 0) {
            const x = ((t - startTime) / windowSize) * W;
            ctx.moveTo(x, 0);
            ctx.lineTo(x, H);
        }
    }
    ctx.stroke();

    // Draw Student Timelines
    let y = 10;
    students.forEach(s => {
        // Draw row background
        ctx.fillStyle = '#fafafa';
        ctx.fillRect(0, y, W, rowHeight - 10);
        
        // Draw history blocks
        if (s.history.length > 0) {
            let blockStart = s.history[0].ts;
            let blockState = s.history[0].state;
            
            for(let i=1; i<s.history.length; i++) {
                const h = s.history[i];
                // If state changed or gap too large (>1s), draw previous block
                if (h.state !== blockState || (h.ts - s.history[i-1].ts) > 1.0) {
                    drawBlock(blockStart, s.history[i-1].ts, blockState, y, rowHeight-10, startTime, windowSize, W);
                    blockStart = h.ts;
                    blockState = h.state;
                }
            }
            // Draw last block
            drawBlock(blockStart, s.history[s.history.length-1].ts, blockState, y, rowHeight-10, startTime, windowSize, W);
        }
        
        // Label
        ctx.fillStyle = '#333';
        ctx.font = '12px Arial';
        ctx.fillText(`Student ${s.id}`, 5, y + 15);
        
        y += rowHeight;
    });
}

function drawBlock(t1, t2, state, y, h, startT, winSize, W) {
    if (t2 < startT) return;
    
    const x1 = ((t1 - startT) / winSize) * W;
    const x2 = ((t2 - startT) / winSize) * W;
    const w = Math.max(x2 - x1, 2); // min width
    
    ctx.fillStyle = COLORS[state] || COLORS['unknown'];
    ctx.fillRect(x1, y + 20, w, h - 20);
}

init();
