(() => {
  const wsDot = document.getElementById("wsDot");
  const wsText = document.getElementById("wsText");
  const recDot = document.getElementById("recDot");
  const recText = document.getElementById("recText");
  const sessionIdEl = document.getElementById("sessionId");

  const btnStart = document.getElementById("btnStart");
  const btnStop = document.getElementById("btnStop");
  const btnReport = document.getElementById("btnReport");

  const faceList = document.getElementById("faceList");
  const faceCount = document.getElementById("faceCount");
  const lastUpdate = document.getElementById("lastUpdate");
  const emptyState = document.getElementById("emptyState");

  const reportMeta = document.getElementById("reportMeta");
  const reportStatus = document.getElementById("reportStatus");
  const reportLinks = document.getElementById("reportLinks");
  const reportPanel = document.getElementById("reportPanel");

  const errorBanner = document.getElementById("errorBanner");

  const STATUS_LABELS = {
    awake: "Awake",
    drowsy: "Drowsy",
    down: "Head down",
    "drowsy+down": "Drowsy + down",
    unknown: "Unknown",
  };

  const STATUS_CLASS = {
    awake: "state-awake",
    drowsy: "state-drowsy",
    down: "state-down",
    "drowsy+down": "state-mix",
    unknown: "state-unknown",
  };

  const STATE_TICK_MS = 5000;
  const FRAME_THROTTLE_MS = 5000;

  const state = {
    ws: null,
    wsConnected: false,
    isRecording: false,
    sessionId: null,
    faces: new Map(),
    cards: new Map(),
    lastUpdateAt: 0,
    lastFrameAt: 0,
    reportJobId: null,
    reportPolling: false,
    reportAuto: false,
    reportSessionId: null,
    autoStartAttempted: false,
    lastStatusCheck: 0,
  };

  const config = {
    autoStart: false,
    autoReport: false,
    statusPollMs: 12000,
    autoUpload: false,
  };

  function normalizeState(value) {
    const v = String(value || "unknown");
    if (v === "awake" || v === "drowsy" || v === "down" || v === "drowsy+down") return v;
    if (v.includes("drowsy") && v.includes("down")) return "drowsy+down";
    if (v.includes("drowsy")) return "drowsy";
    if (v.includes("down")) return "down";
    return "unknown";
  }

  function setWsStatus(connected) {
    state.wsConnected = connected;
    wsText.textContent = connected ? "connected" : "disconnected";
    wsDot.className = `dot ${connected ? "good" : ""}`.trim();
  }

  function setRecording(isRecording, sessionId = null) {
    state.isRecording = Boolean(isRecording);
    if (sessionId != null) state.sessionId = sessionId;

    recText.textContent = state.isRecording ? "recording" : "idle";
    recDot.className = `dot ${state.isRecording ? "bad" : ""}`.trim();
    sessionIdEl.textContent = state.sessionId ? `#${state.sessionId}` : "";

    btnStart.disabled = state.isRecording;
    btnStop.disabled = !state.isRecording;
    btnReport.disabled = state.isRecording || !state.sessionId || state.reportPolling;
  }

  function showError(message) {
    if (!message) return;
    errorBanner.textContent = message;
    errorBanner.classList.remove("is-hidden");
  }

  function clearError() {
    errorBanner.textContent = "";
    errorBanner.classList.add("is-hidden");
  }

  async function fetchJson(url, options = {}, timeoutMs = 8000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const res = await fetch(url, { ...options, signal: controller.signal });
      const data = await res.json();
      return data;
    } finally {
      clearTimeout(timer);
    }
  }

  async function loadUiConfig() {
    try {
      const j = await fetchJson("/api/ui/config", undefined, 4000);
      const lite = j?.lite || {};
      config.autoStart = Boolean(lite.auto_start);
      config.autoReport = Boolean(lite.auto_report);
      config.autoUpload = Boolean(lite.auto_upload);
      if (typeof lite.status_interval_sec === "number") {
        config.statusPollMs = Math.max(4000, Math.round(lite.status_interval_sec * 1000));
      }
      if (config.autoUpload) {
        try {
          const w = await fetchJson("/api/webdav/config", undefined, 3000);
          if (!w?.config?.enabled) {
            config.autoUpload = false;
          }
        } catch (_) {
          config.autoUpload = false;
        }
      }
    } catch (_) {
      // keep defaults
    }
  }

  function connectWs() {
    try {
      if (state.ws) {
        try {
          state.ws.close();
        } catch (_) {
          // ignore
        }
      }
      const proto = location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(`${proto}://${location.host}/ws`);
      state.ws = ws;
      ws.onopen = () => setWsStatus(true);
      ws.onclose = () => {
        setWsStatus(false);
        setTimeout(connectWs, 1500);
      };
      ws.onerror = () => setWsStatus(false);
      ws.onmessage = (msg) => {
        try {
          const payload = JSON.parse(msg.data);
          if (payload && payload.type === "batch" && Array.isArray(payload.events)) {
            for (const ev of payload.events) handleEvent(ev);
          } else {
            handleEvent(payload);
          }
        } catch (_) {
          // ignore
        }
      };
    } catch (_) {
      setWsStatus(false);
      setTimeout(connectWs, 1500);
    }
  }

  function handleEvent(ev) {
    if (!ev || typeof ev !== "object") return;
    if (ev.type === "lite_snapshot") {
      handleLiteSnapshot(ev);
      return;
    }
    if (ev.type === "lite_delta") {
      handleLiteDelta(ev);
      return;
    }
    if (ev.type === "lite_heartbeat") {
      return;
    }
    if (ev.type === "lite_status") {
      handleLiteStatus(ev);
      return;
    }
    if (ev.type === "frame_data") {
      handleFrameData(ev);
      return;
    }
    if (ev.type === "error") {
      showError(ev.error || "Unknown error");
      return;
    }
  }

  function handleLiteStatus(data) {
    const faces = Array.isArray(data.faces) ? data.faces : [];
    applyFaces(faces, true);
  }

  function handleLiteSnapshot(data) {
    const faces = Array.isArray(data.faces) ? data.faces : [];
    applyFaces(faces, true);
  }

  function handleLiteDelta(data) {
    const changed = data?.changed && typeof data.changed === "object" ? data.changed : {};
    const removed = Array.isArray(data?.removed) ? data.removed : [];
    applyDelta(changed, removed);
  }

  function handleFrameData(data) {
    const now = Date.now();
    if (now - state.lastFrameAt < FRAME_THROTTLE_MS) return;
    state.lastFrameAt = now;
    const faces = Array.isArray(data.faces) ? data.faces : [];
    applyFaces(faces);
  }

  function applyFaces(faces, force = false) {
    const next = new Map();
    for (const f of faces) {
      const sid = f.track_id ?? f.student_id;
      if (sid == null) continue;
      next.set(String(sid), normalizeState(f.state));
    }

    let changed = force || next.size !== state.faces.size;
    if (!changed) {
      for (const [sid, st] of next.entries()) {
        const prev = state.faces.get(sid);
        if (!prev || prev.state !== st) {
          changed = true;
          break;
        }
      }
    }

    if (!changed) return;

    state.faces = new Map();
    for (const [sid, st] of next.entries()) {
      const prev = state.cards.get(sid);
      if (!prev) {
        const card = createCard(sid);
        state.cards.set(sid, card);
      }
      const card = state.cards.get(sid);
      card.state = st;
      card.updatedAt = Date.now();
      state.faces.set(sid, { state: st, updatedAt: card.updatedAt });
      updateCard(card);
    }

    for (const sid of Array.from(state.cards.keys())) {
      if (!next.has(sid)) {
        const card = state.cards.get(sid);
        card.el.remove();
        state.cards.delete(sid);
      }
    }

    renderFaceList();
    state.lastUpdateAt = Date.now();
    updateLastUpdateText();
  }

  function applyDelta(changed, removed) {
    let didChange = false;

    for (const [sid, st] of Object.entries(changed || {})) {
      const id = String(sid);
      const norm = normalizeState(st);
      let card = state.cards.get(id);
      if (!card) {
        card = createCard(id);
        state.cards.set(id, card);
      }
      card.state = norm;
      card.updatedAt = Date.now();
      updateCard(card);
      state.faces.set(id, { state: norm, updatedAt: card.updatedAt });
      didChange = true;
    }

    for (const sid of removed || []) {
      const id = String(sid);
      const card = state.cards.get(id);
      if (card) {
        card.el.remove();
        state.cards.delete(id);
      }
      if (state.faces.has(id)) {
        state.faces.delete(id);
        didChange = true;
      }
    }

    if (!didChange) return;
    renderFaceList();
    state.lastUpdateAt = Date.now();
    updateLastUpdateText();
  }

  function createCard(id) {
    const el = document.createElement("div");
    el.className = "face-card";
    el.setAttribute("role", "listitem");

    const row = document.createElement("div");
    row.className = "face-row";

    const label = document.createElement("div");
    label.className = "face-id";
    label.textContent = `ID ${id}`;

    const status = document.createElement("div");
    status.className = "state-pill state-unknown";
    status.textContent = "Unknown";

    row.appendChild(label);
    row.appendChild(status);

    const meta = document.createElement("div");
    meta.className = "face-meta";
    meta.textContent = "Last change: just now";

    el.appendChild(row);
    el.appendChild(meta);

    return { id, el, statusEl: status, metaEl: meta, state: "unknown", updatedAt: 0 };
  }

  function updateCard(card) {
    const st = card.state;
    const label = STATUS_LABELS[st] || STATUS_LABELS.unknown;
    const cls = STATUS_CLASS[st] || STATUS_CLASS.unknown;
    card.statusEl.textContent = label;
    card.statusEl.className = `state-pill ${cls}`;
    card.metaEl.textContent = `Last change: ${formatAge(card.updatedAt)}`;
    card.el.classList.add("is-updated");
    setTimeout(() => card.el.classList.remove("is-updated"), 900);
  }

  function renderFaceList() {
    const ids = Array.from(state.cards.keys());
    ids.sort((a, b) => {
      const na = Number(a);
      const nb = Number(b);
      const fa = Number.isFinite(na);
      const fb = Number.isFinite(nb);
      if (fa && fb) return na - nb;
      if (fa && !fb) return -1;
      if (!fa && fb) return 1;
      return String(a).localeCompare(String(b));
    });

    faceList.innerHTML = "";
    const frag = document.createDocumentFragment();
    for (const id of ids) {
      frag.appendChild(state.cards.get(id).el);
    }
    faceList.appendChild(frag);

    const count = ids.length;
    faceCount.textContent = String(count);
    if (emptyState) {
      emptyState.classList.toggle("is-hidden", count > 0);
      if (count === 0) faceList.appendChild(emptyState);
    }
  }

  function updateLastUpdateText() {
    if (!state.lastUpdateAt) {
      lastUpdate.textContent = "No updates yet";
      return;
    }
    lastUpdate.textContent = `Updated ${formatAge(state.lastUpdateAt)}`;
  }

  function formatAge(ts) {
    if (!ts) return "-";
    const diff = Math.max(0, Math.floor((Date.now() - ts) / 1000));
    if (diff < 2) return "just now";
    if (diff < 60) return `${diff}s ago`;
    const min = Math.floor(diff / 60);
    const sec = diff % 60;
    return `${min}m ${sec}s ago`;
  }

  async function refreshSessionStatus() {
    try {
      const j = await fetchJson("/api/session/status");
      const prevRecording = state.isRecording;
      setRecording(Boolean(j?.is_recording), j?.session_id || null);
      if (!state.isRecording && state.faces.size) {
        clearFaces();
      }
      if (config.autoReport && prevRecording && !state.isRecording && state.sessionId) {
        startReport(true);
      }
      if (config.autoStart && !state.isRecording && !state.autoStartAttempted) {
        state.autoStartAttempted = true;
        startRecording();
      }
    } catch (e) {
      showError(`Status error: ${String(e)}`);
    }
  }

  async function startRecording() {
    clearError();
    try {
      const j = await fetchJson("/api/session/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      if (!j.ok) throw new Error(j.error || "start failed");
      setRecording(true, j.session_id || null);
    } catch (e) {
      showError(`Start failed: ${String(e)}`);
      setRecording(false, state.sessionId);
    }
  }

  async function stopRecording() {
    clearError();
    try {
      const j = await fetchJson("/api/session/stop", { method: "POST" });
      if (!j.ok) throw new Error(j.error || "stop failed");
      setRecording(false, j.session_id || state.sessionId);
    } catch (e) {
      showError(`Stop failed: ${String(e)}`);
      setRecording(false, state.sessionId);
    }
  }

  function setReportStatus(text, meta = null) {
    reportStatus.textContent = text;
    if (meta != null) reportMeta.textContent = meta;
  }

  function setReportVisible(show) {
    if (!reportPanel) return;
    reportPanel.classList.toggle("is-hidden", !show);
  }

  function clearFaces() {
    state.faces = new Map();
    for (const card of state.cards.values()) {
      card.el.remove();
    }
    state.cards = new Map();
    renderFaceList();
    state.lastUpdateAt = 0;
    updateLastUpdateText();
  }

  function renderReportLinks(result) {
    reportLinks.innerHTML = "";
    if (!result) return;
    const links = [
      { label: "Stats", href: result.stats },
      { label: "Transcript", href: result.transcript },
      { label: "Lesson Summary", href: result.lesson_summary },
      { label: "Video", href: result.video },
      { label: "Audio", href: result.audio },
    ];
    for (const item of links) {
      if (!item.href) continue;
      const a = document.createElement("a");
      a.href = item.href;
      a.target = "_blank";
      a.rel = "noreferrer";
      a.textContent = item.label;
      reportLinks.appendChild(a);
    }
  }

  async function startReport(isAuto = false) {
    clearError();
    const sid = state.sessionId;
    if (!sid) {
      showError("No session_id available. Start and stop a session first.");
      return;
    }
    if (state.reportPolling) return;
    state.reportAuto = Boolean(isAuto);
    state.reportSessionId = sid;
    state.reportPolling = true;
    btnReport.disabled = true;
    setReportStatus("Submitting job...", "Running");
    reportLinks.innerHTML = "";
    if (!state.reportAuto) {
      setReportVisible(true);
    }
    try {
      const j = await fetchJson("/api/session/process", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sid }),
      });
      if (!j.ok) throw new Error(j.error || "process start failed");
      state.reportJobId = j.job_id;
      await pollReportJob(j.job_id);
    } catch (e) {
      setReportStatus(`Report failed: ${String(e)}`, "Error");
      setReportVisible(true);
      showError(`Report failed: ${String(e)}`);
      state.reportPolling = false;
      btnReport.disabled = state.isRecording || !state.sessionId;
    }
  }

  async function pollReportJob(jobId) {
    const maxPoll = 240;
    for (let i = 0; i < maxPoll; i++) {
      await new Promise((r) => setTimeout(r, 1000));
      let st = null;
      try {
        st = await fetchJson(`/api/session/process/status?job_id=${encodeURIComponent(jobId)}`);
      } catch (e) {
        setReportStatus(`Polling error: ${String(e)}`, "Error");
        continue;
      }
      const job = st?.job;
      const status = String(job?.status || "unknown");
      if (status === "done") {
        state.reportPolling = false;
        setReportStatus("Report ready", "Done");
        renderReportLinks(job?.result || null);
        btnReport.disabled = state.isRecording || !state.sessionId;
        const shouldShow = !state.reportAuto || !config.autoUpload;
        if (shouldShow) setReportVisible(true);
        if (config.autoUpload) {
          await uploadToWebdav(state.reportSessionId || state.sessionId);
        }
        return;
      }
      if (status === "error") {
        state.reportPolling = false;
        setReportStatus(`Report error: ${String(job?.error || "unknown")}`, "Error");
        setReportVisible(true);
        btnReport.disabled = state.isRecording || !state.sessionId;
        return;
      }
      setReportStatus(`Processing: ${status}`, "Running");
    }
    state.reportPolling = false;
    setReportStatus("Report timeout. Try again later.", "Timeout");
    setReportVisible(true);
    btnReport.disabled = state.isRecording || !state.sessionId;
  }

  async function uploadToWebdav(sessionId) {
    if (!sessionId) return;
    try {
      setReportStatus("Uploading report...", "Running");
      const j = await fetchJson("/api/webdav/upload", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      }, 15000);
      if (!j.ok) throw new Error(j.error || "upload failed");
      setReportStatus("Report ready (uploaded)", "Done");
    } catch (e) {
      setReportStatus(`Upload failed: ${String(e)}`, "Error");
      setReportVisible(true);
      showError(`Upload failed: ${String(e)}`);
    }
  }

  function bindUi() {
    btnStart.addEventListener("click", startRecording);
    btnStop.addEventListener("click", stopRecording);
    btnReport.addEventListener("click", () => startReport(false));
  }

  async function init() {
    await loadUiConfig();
    bindUi();
    connectWs();
    await refreshSessionStatus();
    setInterval(refreshSessionStatus, config.statusPollMs);
    setInterval(() => {
      if (document.hidden) return;
      updateLastUpdateText();
      for (const card of state.cards.values()) {
        card.metaEl.textContent = `Last change: ${formatAge(card.updatedAt)}`;
      }
    }, STATE_TICK_MS);
  }

  init();
})();
