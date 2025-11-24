(() => {
  const WS_URL = (location.protocol === 'https:' ? 'wss:' : 'ws:') + '//' + location.host + '/ws';
  const canvas = document.getElementById('timeline');
  const ctx = canvas.getContext('2d');
  const status = document.getElementById('status');
  const windowSel = document.getElementById('windowSel');
  const showText = document.getElementById('showText');

  let dpr = window.devicePixelRatio || 1;
  function resize() {
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.floor(rect.width * dpr);
    canvas.height = Math.floor(rect.height * dpr);
    ctx.scale(dpr, dpr);
  }
  window.addEventListener('resize', resize);
  resize();

  const segments = new Map(); // segment_id -> {start,end,label,text}

  function nowSec() { return Date.now() / 1000; }

  function handleEvent(e) {
    if (!e || !e.type) return;
    if (e.type === 'asr_segment') {
      segments.set(e.id || ('s'+Math.random()), {start: e.start, end: e.end, label: e.label || 'unknown', text: e.text || ''});
    } else if (e.type === 'asr_sentence') {
      // attach sentence text to segment if present
      const seg = segments.get(e.segment_id);
      if (seg) {
        seg.text = (seg.text ? seg.text + ' ' : '') + e.text;
      }
    }
  }

  function render() {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    const win = parseFloat(windowSel.value) || 60;
    const now = nowSec();

    ctx.clearRect(0, 0, w, h);
    // background
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0,0,w,h);

    // draw a baseline
    ctx.fillStyle = '#eee';
    ctx.fillRect(0, h/2 - 30, w, 60);

    // draw segments
    for (const [id, s] of segments) {
      // skip if completely out of window
      if ((s.end || now) < now - win) continue;
      if ((s.start || 0) > now) continue;
      const x1 = Math.max(0, ((s.start - (now - win)) / win) * w);
      const x2 = Math.min(w, ((Math.min(s.end || now, now) - (now - win)) / win) * w);
      const width = Math.max(2, x2 - x1);
      // color by label
      let col = '#999';
      if (s.label === 'teacher') col = '#d9534f';
      else if (s.label === 'student') col = '#428bca';
      ctx.fillStyle = col;
      ctx.fillRect(x1, h/2 - 20, width, 40);

      if (showText.checked && s.text) {
        ctx.fillStyle = '#000';
        ctx.font = '12px Arial';
        const text = s.text.length > 80 ? s.text.slice(0,77) + '...' : s.text;
        ctx.fillText(text, x1 + 4, h/2 - 26);
      }
    }

    // draw time ticks
    ctx.fillStyle = '#666';
    ctx.font = '11px Arial';
    for (let i=0;i<=6;i++){
      const tx = (i/6)*w;
      const t = now - win + (i/6)*win;
      ctx.fillText(new Date(t*1000).toLocaleTimeString(), tx+4, h - 6);
      ctx.fillRect(tx, h/2 - 30, 1, 60);
    }

    requestAnimationFrame(render);
  }

  // websocket connection and event queue
  const eventQueue = [];
  function connect() {
    const ws = new WebSocket(WS_URL);
    ws.onopen = () => { status.textContent = 'connected'; };
    ws.onclose = () => { status.textContent = 'disconnected'; setTimeout(connect, 1000); };
    ws.onerror = (e) => { status.textContent = 'error'; };
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === 'batch' && Array.isArray(msg.events)) {
          for (const e of msg.events) handleEvent(e);
        } else {
          handleEvent(msg);
        }
      } catch (err) {
        console.warn('invalid message', err);
      }
    };
  }

  connect();
  requestAnimationFrame(render);
})();
