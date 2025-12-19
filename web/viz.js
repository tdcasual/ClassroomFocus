(() => {
  const previewCanvas = document.getElementById("preview");
  const previewCtx = previewCanvas.getContext("2d", { alpha: false });
  const timelineCanvas = document.getElementById("timeline");
  const timelineCtx = timelineCanvas.getContext("2d");

  const studentList = document.getElementById("studentList");
  const studentFilter = document.getElementById("studentFilter");
  const windowSel = document.getElementById("windowSel");
  const sessionSelect = document.getElementById("sessionSelect");
  const btnRefreshSessions = document.getElementById("btnRefreshSessions");

  const wsDot = document.getElementById("wsDot");
  const wsStatus = document.getElementById("wsStatus");

  const recDot = document.getElementById("recDot");
  const recStatusText = document.getElementById("recStatusText");
  const recTimer = document.getElementById("recTimer");
  const sessionIdText = document.getElementById("sessionIdText");
  const toastContainer = document.getElementById("toastContainer");

  const btnModelCenter = document.getElementById("btnModelCenter");
  const modelDot = document.getElementById("modelDot");
  const modelStatusText = document.getElementById("modelStatusText");

  const modelModal = document.getElementById("modelModal");
  const modelDrawer = document.getElementById("modelDrawer");
  const modelDragHandle = document.getElementById("modelDragHandle");
  const btnModelsSave = document.getElementById("btnModelsSave");
  const btnModelsCheck = document.getElementById("btnModelsCheck");
  const btnModelsCheckDeep = document.getElementById("btnModelsCheckDeep");
  const btnModelsResetPos = document.getElementById("btnModelsResetPos");
  const btnModelsClose = document.getElementById("btnModelsClose");
  const modelModeSel = document.getElementById("modelModeSel");
  const asrProviderSel = document.getElementById("asrProviderSel");
  const asrModelSel = document.getElementById("asrModelSel");
  const asrModelInput = document.getElementById("asrModelInput");
  const asrUseIndependentToggle = document.getElementById("asrUseIndependentToggle");
  const asrIndependentSettings = document.getElementById("asrIndependentSettings");
  const asrBaseUrlInput = document.getElementById("asrBaseUrlInput");
  const asrApiKeyInput = document.getElementById("asrApiKeyInput");
  const llmEnabledToggle = document.getElementById("llmEnabledToggle");
  const llmBaseUrlInput = document.getElementById("llmBaseUrlInput");
  const llmApiKeyInput = document.getElementById("llmApiKeyInput");
  const llmModelSel = document.getElementById("llmModelSel");
  const btnLlmPullModels = document.getElementById("btnLlmPullModels");
  const llmModelsStatus = document.getElementById("llmModelsStatus");
  const llmModelInput = document.getElementById("llmModelInput");
  const modelCheckBox = document.getElementById("modelCheckBox");
  const modelEnvBox = document.getElementById("modelEnvBox");
  const modelContent = document.getElementById("modelContent");
  const modelNavButtons = Array.from(document.querySelectorAll(".model-nav-btn"));
  const modelNav = document.querySelector(".model-nav");
  const modelSections = Array.from(document.querySelectorAll(".model-section"));
  const llmSection = document.getElementById("model-section-llm");
  const llmFormGrid = llmSection ? llmSection.querySelector(".form-grid") : null;

  const btnStart = document.getElementById("btnStart");
  const btnStop = document.getElementById("btnStop");
  const btnReport = document.getElementById("btnReport");
  const btnOpenReport = document.getElementById("btnOpenReport");

  const toggleBoxes = document.getElementById("toggleBoxes");
  const toggleLabels = document.getElementById("toggleLabels");
  const toggleAsr = document.getElementById("toggleAsr");
  const toggleTimeline = document.getElementById("toggleTimeline");
  const timelinePanel = document.getElementById("timelinePanel");
  const mainPanel = document.querySelector(".main");

  const fpsDot = document.getElementById("fpsDot");
  const fpsText = document.getElementById("fpsText");

  const selectedStudentText = document.getElementById("selectedStudentText");
  const nowText = document.getElementById("nowText");
  const asrNowText = document.getElementById("asrNowText");

  const btnFollow = document.getElementById("btnFollow");
  const btnClearCursor = document.getElementById("btnClearCursor");
  const cursorInspector = document.getElementById("cursorInspector");
  const cursorTimeText = document.getElementById("cursorTimeText");
  const cursorStateText = document.getElementById("cursorStateText");
  const cursorAsrText = document.getElementById("cursorAsrText");

  const reportModal = document.getElementById("reportModal");
  const reportDrawer = document.getElementById("reportDrawer");
  const reportDragHandle = document.getElementById("reportDragHandle");
  const btnMaximizeReport = document.getElementById("btnMaximizeReport");
  const btnResetReportPos = document.getElementById("btnResetReportPos");
  const maximizeIcon = document.getElementById("maximizeIcon");
  const btnCloseReport = document.getElementById("btnCloseReport");
  const btnReloadReport = document.getElementById("btnReloadReport");
  const reportKpis = document.getElementById("reportKpis");
  const reportLinks = document.getElementById("reportLinks");
  const reportBody = document.getElementById("reportBody");
  const reportSessionId = document.getElementById("reportSessionId");
  const reportSessionName = document.getElementById("reportSessionName");
  const btnRenameSession = document.getElementById("btnRenameSession");

  // Settings Center elements
  const btnSettingsCenter = document.getElementById("btnSettingsCenter");
  const settingsDot = document.getElementById("settingsDot");
  const settingsModal = document.getElementById("settingsModal");
  const settingsDrawer = document.getElementById("settingsDrawer");
  const settingsDragHandle = document.getElementById("settingsDragHandle");
  const btnSettingsSave = document.getElementById("btnSettingsSave");
  const btnSettingsClose = document.getElementById("btnSettingsClose");
  const settingsNavBtns = document.querySelectorAll(".settings-nav-btn");
  const settingsTabs = document.querySelectorAll(".settings-tab");
  
  // WebDAV form elements
  const webdavEnabled = document.getElementById("webdavEnabled");
  const webdavUrl = document.getElementById("webdavUrl");
  const webdavUsername = document.getElementById("webdavUsername");
  const webdavPassword = document.getElementById("webdavPassword");
  const webdavRemotePath = document.getElementById("webdavRemotePath");
  const webdavAutoUpload = document.getElementById("webdavAutoUpload");
  const webdavUploadVideo = document.getElementById("webdavUploadVideo");
  const webdavUploadAudio = document.getElementById("webdavUploadAudio");
  const webdavUploadStats = document.getElementById("webdavUploadStats");
  const webdavUploadTranscript = document.getElementById("webdavUploadTranscript");
  const webdavUploadAll = document.getElementById("webdavUploadAll");
  const webdavFormGrid = document.getElementById("webdavFormGrid");
  const btnWebdavTest = document.getElementById("btnWebdavTest");
  const btnWebdavReset = document.getElementById("btnWebdavReset");
  const btnToggleWebdavPassword = document.getElementById("btnToggleWebdavPassword");
  const webdavTestStatus = document.getElementById("webdavTestStatus");
  const btnUploadWebdav = document.getElementById("btnUploadWebdav");

  const COLORS = {
    awake: "#22c55e",
    drowsy: "#f59e0b",
    down: "#ef4444",
    "drowsy+down": "#8b5cf6",
    unknown: "rgba(255,255,255,0.45)",
    teacher: "rgba(56,189,248,0.30)",
    student: "rgba(255,255,255,0.16)",
  };

  const state = {
    ws: null,
    wsConnected: false,
    isRecording: false,
    sessionId: null,
    recordingStartTime: null,
    recordingTimerId: null,
    sessions: [],  // List of all sessions from /api/sessions

    // WebDAV/Settings state
    webdav: {
      config: null,
      dirty: false,
    },

    windowSec: 60,
    nowSec: 0,

    students: new Map(), // id -> Student
    selectedStudentId: null,

    followLive: true,
    cursorTime: null,

    showBoxes: true,
    showLabels: true,
    showAsr: true,

    lastFrame: {
      image: null,
      ts: 0,
      faces: [],
      b64: null,
    },
    asrSegments: [], // {start,end,text,label}

    fps: {
      lastSec: 0,
      framesThisSec: 0,
      value: 0,
    },

    report: {
      jobId: null,
      lastResult: null,
      lastStatsUrl: null,
      lastSessionId: null,
      polling: false,
    },

    models: {
      config: null,
      env: null,
      providers: null,
      lastCheck: null,
      suggestedMode: null,
      dirty: false,
      checking: false,
      llmModels: [],
      llmModelsLoading: false,
      llmModelsError: null,
      ui: {
        view: "__all__",
        allScrollTop: 0,
      },
    },
  };

  function clamp(v, min, max) {
    return Math.max(min, Math.min(max, v));
  }

  // ===== Toast Notification System =====
  function showToast(message, type = "info", durationMs = 4000) {
    if (!toastContainer) return;
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    
    const textSpan = document.createElement("span");
    textSpan.textContent = message;
    toast.appendChild(textSpan);
    
    const closeBtn = document.createElement("button");
    closeBtn.className = "toast-close";
    closeBtn.innerHTML = "\u00d7";
    closeBtn.title = "关闭";
    closeBtn.addEventListener("click", () => {
      toast.classList.add("fade-out");
      setTimeout(() => toast.remove(), 300);
    });
    toast.appendChild(closeBtn);
    
    toastContainer.appendChild(toast);
    setTimeout(() => {
      if (toast.parentElement) {
        toast.classList.add("fade-out");
        setTimeout(() => toast.remove(), 300);
      }
    }, durationMs);
  }

  // ===== Recording Timer =====
  function formatRecordingTime(ms) {
    const totalSec = Math.floor(ms / 1000);
    const min = Math.floor(totalSec / 60);
    const sec = totalSec % 60;
    return `${String(min).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
  }

  function startRecordingTimer() {
    state.recordingStartTime = Date.now();
    if (recTimer) {
      recTimer.style.display = "inline";
      recTimer.textContent = "00:00";
    }
    state.recordingTimerId = setInterval(() => {
      if (recTimer && state.recordingStartTime) {
        recTimer.textContent = formatRecordingTime(Date.now() - state.recordingStartTime);
      }
    }, 1000);
  }

  function stopRecordingTimer() {
    if (state.recordingTimerId) {
      clearInterval(state.recordingTimerId);
      state.recordingTimerId = null;
    }
    state.recordingStartTime = null;
    if (recTimer) recTimer.style.display = "none";
  }

  // ===== Button Loading State =====
  function setButtonLoading(btn, loading) {
    if (!btn) return;
    if (loading) {
      btn.classList.add("loading");
      btn.disabled = true;
    } else {
      btn.classList.remove("loading");
    }
  }

  // ===== LLM Section Disabled State =====
  function updateLlmSectionDisabledState() {
    if (!llmSection || !llmFormGrid) return;
    const enabled = llmEnabledToggle?.checked ?? true;
    if (enabled) {
      llmSection.classList.remove("disabled");
      llmFormGrid.classList.remove("disabled");
    } else {
      llmSection.classList.add("disabled");
      llmFormGrid.classList.add("disabled");
    }
  }

  const MODEL_POS_KEY = "model_center_pos_v1";
  let modelDragState = null;

  function readModelDrawerPos() {
    try {
      const raw = localStorage.getItem(MODEL_POS_KEY);
      if (!raw) return null;
      const data = JSON.parse(raw);
      if (!data || typeof data !== "object") return null;
      const left = Number(data.left);
      const top = Number(data.top);
      if (!Number.isFinite(left) || !Number.isFinite(top)) return null;
      return { left, top };
    } catch (_) {
      return null;
    }
  }

  function writeModelDrawerPos(left, top) {
    try {
      localStorage.setItem(MODEL_POS_KEY, JSON.stringify({ left, top }));
    } catch (_) {
      // ignore
    }
  }

  function clampModelDrawerPos(left, top, width, height) {
    const margin = 12;
    const maxLeft = Math.max(margin, window.innerWidth - width - margin);
    const maxTop = Math.max(margin, window.innerHeight - height - margin);
    return {
      left: clamp(left, margin, maxLeft),
      top: clamp(top, margin, maxTop),
    };
  }

  function setModelDrawerPos(left, top) {
    if (!modelDrawer) return;
    modelDrawer.style.left = `${Math.round(left)}px`;
    modelDrawer.style.top = `${Math.round(top)}px`;
    modelDrawer.style.right = "auto";
    modelDrawer.style.bottom = "auto";
    modelDrawer.style.transform = "none";
  }

  function initModelDrawerPos({ reset = false } = {}) {
    if (!modelDrawer) return;
    requestAnimationFrame(() => {
      const rect = modelDrawer.getBoundingClientRect();
      const stored = reset ? null : readModelDrawerPos();
      let left = stored?.left;
      let top = stored?.top;
      if (!Number.isFinite(left) || !Number.isFinite(top)) {
        left = Math.max(16, (window.innerWidth - rect.width) / 2);
        top = Math.max(16, (window.innerHeight - rect.height) / 5);
      }
      const clamped = clampModelDrawerPos(left, top, rect.width, rect.height);
      setModelDrawerPos(clamped.left, clamped.top);
      writeModelDrawerPos(clamped.left, clamped.top);
    });
  }

  function ensureModelDrawerInViewport() {
    if (!modelDrawer || !modelModal.classList.contains("open")) return;
    const rect = modelDrawer.getBoundingClientRect();
    const clamped = clampModelDrawerPos(rect.left, rect.top, rect.width, rect.height);
    setModelDrawerPos(clamped.left, clamped.top);
    writeModelDrawerPos(clamped.left, clamped.top);
  }

  function resetModelDrawerPos() {
    try {
      localStorage.removeItem(MODEL_POS_KEY);
    } catch (_) {
      // ignore
    }
    initModelDrawerPos({ reset: true });
  }

  function startModelDrag(e) {
    if (!modelDrawer || !modelDragHandle) return;
    if (e.button !== 0) return;
    const rect = modelDrawer.getBoundingClientRect();
    modelDragState = {
      pointerId: e.pointerId,
      startX: e.clientX,
      startY: e.clientY,
      left: rect.left,
      top: rect.top,
      width: rect.width,
      height: rect.height,
    };
    modelDrawer.classList.add("dragging");
    modelDragHandle.setPointerCapture(e.pointerId);
    e.preventDefault();
  }

  function moveModelDrag(e) {
    if (!modelDragState || e.pointerId !== modelDragState.pointerId) return;
    const dx = e.clientX - modelDragState.startX;
    const dy = e.clientY - modelDragState.startY;
    const nextLeft = modelDragState.left + dx;
    const nextTop = modelDragState.top + dy;
    const clamped = clampModelDrawerPos(nextLeft, nextTop, modelDragState.width, modelDragState.height);
    setModelDrawerPos(clamped.left, clamped.top);
  }

  function endModelDrag(e) {
    if (!modelDragState || e.pointerId !== modelDragState.pointerId) return;
    modelDrawer.classList.remove("dragging");
    modelDragHandle.releasePointerCapture(e.pointerId);
    const rect = modelDrawer.getBoundingClientRect();
    writeModelDrawerPos(rect.left, rect.top);
    modelDragState = null;
  }

  // ===== Report Drawer Drag & Maximize =====
  const REPORT_POS_KEY = "report_drawer_pos_v1";
  let reportDragState = null;
  let reportMaximized = false;

  function readReportDrawerPos() {
    try {
      const raw = localStorage.getItem(REPORT_POS_KEY);
      if (!raw) return null;
      const data = JSON.parse(raw);
      if (!data || typeof data !== "object") return null;
      const left = Number(data.left);
      const top = Number(data.top);
      if (!Number.isFinite(left) || !Number.isFinite(top)) return null;
      return { left, top };
    } catch {
      return null;
    }
  }

  function writeReportDrawerPos(left, top) {
    try {
      localStorage.setItem(REPORT_POS_KEY, JSON.stringify({ left, top }));
    } catch {}
  }

  function clampReportDrawerPos(left, top, width, height) {
    const maxLeft = window.innerWidth - width;
    const maxTop = window.innerHeight - height;
    return {
      left: clamp(left, 0, Math.max(0, maxLeft)),
      top: clamp(top, 0, Math.max(0, maxTop)),
    };
  }

  function setReportDrawerPos(left, top) {
    if (!reportDrawer) return;
    reportDrawer.style.left = `${left}px`;
    reportDrawer.style.top = `${top}px`;
  }

  function initReportDrawerPos({ reset = false } = {}) {
    if (!reportDrawer) return;
    const saved = reset ? null : readReportDrawerPos();
    if (saved) {
      const rect = reportDrawer.getBoundingClientRect();
      const clamped = clampReportDrawerPos(saved.left, saved.top, rect.width, rect.height);
      setReportDrawerPos(clamped.left, clamped.top);
    } else {
      reportDrawer.style.left = "8vw";
      reportDrawer.style.top = "8vh";
    }
  }

  function resetReportDrawerPos() {
    localStorage.removeItem(REPORT_POS_KEY);
    initReportDrawerPos({ reset: true });
  }

  function startReportDrag(e) {
    if (reportMaximized) return; // Don't drag when maximized
    if (!reportDrawer) return;
    const rect = reportDrawer.getBoundingClientRect();
    reportDragState = {
      startX: e.clientX,
      startY: e.clientY,
      startLeft: rect.left,
      startTop: rect.top,
      pointerId: e.pointerId,
    };
    reportDrawer.classList.add("dragging");
    reportDragHandle.setPointerCapture(e.pointerId);
  }

  function moveReportDrag(e) {
    if (!reportDragState || e.pointerId !== reportDragState.pointerId) return;
    const dx = e.clientX - reportDragState.startX;
    const dy = e.clientY - reportDragState.startY;
    const rect = reportDrawer.getBoundingClientRect();
    const clamped = clampReportDrawerPos(reportDragState.startLeft + dx, reportDragState.startTop + dy, rect.width, rect.height);
    setReportDrawerPos(clamped.left, clamped.top);
  }

  function endReportDrag(e) {
    if (!reportDragState || e.pointerId !== reportDragState.pointerId) return;
    reportDrawer.classList.remove("dragging");
    reportDragHandle.releasePointerCapture(e.pointerId);
    const rect = reportDrawer.getBoundingClientRect();
    writeReportDrawerPos(rect.left, rect.top);
    reportDragState = null;
  }

  function toggleReportMaximize() {
    if (!reportDrawer) return;
    reportMaximized = !reportMaximized;
    reportDrawer.classList.toggle("maximized", reportMaximized);
    if (maximizeIcon) {
      maximizeIcon.textContent = reportMaximized ? "\u2922" : "\u2922"; // Use different icons
      maximizeIcon.textContent = reportMaximized ? "\u2198" : "\u2922"; // Shrink vs Expand
    }
  }

  function setActiveModelNav(targetId) {
    for (const btn of modelNavButtons) {
      btn.classList.toggle("active", btn.dataset.target === targetId);
    }
  }

  function setModelSettingsView(nextView, { preserveAllScroll = true } = {}) {
    const ui = state.models?.ui && typeof state.models.ui === "object" ? state.models.ui : null;
    const curView = ui && typeof ui.view === "string" ? ui.view : "__all__";
    let view = String(nextView || "__all__").trim() || "__all__";
    if (view !== "__all__" && !modelSections.some((s) => s && s.id === view)) {
      view = "__all__";
    }

    if (ui) ui.view = view;

    if (modelContent && ui && curView === "__all__" && view !== "__all__") {
      ui.allScrollTop = modelContent.scrollTop;
    }

    for (const sec of modelSections) {
      if (!sec) continue;
      sec.hidden = view !== "__all__" && sec.id !== view;
    }

    if (modelContent) {
      const top = view === "__all__" && ui && preserveAllScroll ? Number(ui.allScrollTop) || 0 : 0;
      try {
        modelContent.scrollTo({ top, behavior: "auto" });
      } catch (_) {
        modelContent.scrollTop = top;
      }
    }

    setActiveModelNav(view);
  }

  let rafId = null;
  let needsPreview = false;
  let needsTimeline = false;
  function scheduleRender({ preview = false, timeline = false } = {}) {
    needsPreview = needsPreview || preview;
    needsTimeline = needsTimeline || timeline;
    if (rafId != null) return;
    rafId = requestAnimationFrame(() => {
      rafId = null;
      if (needsPreview) drawPreview();
      if (needsTimeline) renderTimeline();
      needsPreview = false;
      needsTimeline = false;
    });
  }

  function toNum(v, fallback = 0) {
    const n = Number(v);
    return Number.isFinite(n) ? n : fallback;
  }

  function formatSec(sec) {
    const v = Math.max(0, Number(sec) || 0);
    return `${v.toFixed(2)}s`;
  }

  function formatDuration(sec) {
    const v = Math.max(0, Number(sec) || 0);
    if (v < 60) return `${v.toFixed(1)}s`;
    const m = Math.floor(v / 60);
    const s = v - m * 60;
    return `${m}m ${s.toFixed(0)}s`;
  }

  function escapeHtml(str) {
    if (str == null) return "";
    return String(str).replace(/[&<>"]/g, (ch) => {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch] || ch;
    });
  }

  function normalizeState(s) {
    const v = String(s || "").trim().toLowerCase();
    if (!v) return "unknown";
    if (v === "awake" || v === "drowsy" || v === "down" || v === "drowsy+down") return v;
    if (v.includes("drowsy") && v.includes("down")) return "drowsy+down";
    if (v.includes("drowsy")) return "drowsy";
    if (v.includes("down")) return "down";
    return "unknown";
  }

  function stateLabel(s) {
    switch (s) {
      case "awake":
        return "清醒";
      case "drowsy":
        return "瞌睡";
      case "down":
        return "低头";
      case "drowsy+down":
        return "瞌睡+低头";
      default:
        return "未知";
    }
  }

  function intervalTypeLabel(t) {
    const v = String(t || "").toUpperCase();
    if (v === "INATTENTIVE") return "走神/疑似睡觉";
    if (v === "DROWSY") return "睡觉（闭眼）";
    if (v === "LOOKING_DOWN") return "低头";
    if (v === "NOT_VISIBLE") return "眼睛不可见（趴下/遮挡）";
    return v || "UNKNOWN";
  }

  function dotClassForState(s) {
    switch (s) {
      case "awake":
        return "good";
      case "drowsy":
        return "warn";
      case "down":
        return "bad";
      case "drowsy+down":
        return "severe";
      default:
        return "unknown";
    }
  }

  function setWsStatus(connected) {
    state.wsConnected = connected;
    wsStatus.textContent = connected ? "connected" : "disconnected";
    wsDot.className = `dot ${connected ? "good" : "unknown"}`;
  }

  function setRecording(isRec, sessionId = null) {
    const wasRecording = state.isRecording;
    state.isRecording = Boolean(isRec);
    if (sessionId != null) state.sessionId = sessionId;

    btnStart.disabled = state.isRecording;
    btnStop.disabled = !state.isRecording;
    btnReport.disabled = state.isRecording || !state.sessionId;
    btnOpenReport.style.display = state.report.lastStatsUrl ? "inline-flex" : "none";
    btnModelCenter.disabled = state.isRecording;

    recStatusText.textContent = state.isRecording ? "recording" : "idle";
    recDot.className = `dot ${state.isRecording ? "bad pulse" : "unknown"}`;

    // Handle recording timer
    if (state.isRecording && !wasRecording) {
      startRecordingTimer();
    } else if (!state.isRecording && wasRecording) {
      stopRecordingTimer();
    }

    if (state.sessionId) {
      sessionIdText.textContent = state.sessionId;
      sessionIdText.style.display = "inline";
      sessionIdText.className = "mono";
    } else {
      sessionIdText.style.display = "none";
    }
  }

  function resizeCanvasToElement(canvas) {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(rect.width * dpr));
    const h = Math.max(1, Math.round(rect.height * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
      return true;
    }
    return false;
  }

  function connectWs() {
    try {
      if (state.ws) {
        try {
          state.ws.close();
        } catch (_) {
          // ignore
        }
        state.ws = null;
      }
      const proto = location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(`${proto}://${location.host}/ws`);
      state.ws = ws;
      ws.onopen = () => setWsStatus(true);
      ws.onclose = () => {
        setWsStatus(false);
        setTimeout(connectWs, 1200);
      };
      ws.onerror = () => setWsStatus(false);
      ws.onmessage = (msg) => {
        try {
          const data = JSON.parse(msg.data);
          if (data && data.type === "batch" && Array.isArray(data.events)) {
            for (const ev of data.events) handleEvent(ev);
          } else {
            handleEvent(data);
          }
        } catch (_) {
          // ignore
        }
      };
    } catch (_) {
      setWsStatus(false);
      setTimeout(connectWs, 1200);
    }
  }

  function handleEvent(ev) {
    if (!ev || typeof ev !== "object") return;
    const typ = ev.type;
    if (typ === "frame_data") {
      handleFrameData(ev);
      return;
    }
    if (typ === "asr_segment") {
      handleAsrSegment(ev);
      return;
    }
    if (typ === "ASR_SENTENCE") {
      const t = toNum(ev.ts, state.nowSec);
      const text = typeof ev.text === "string" ? ev.text : JSON.stringify(ev.text || "");
      handleAsrSegment({ start: t, end: t + 2.0, text, label: "teacher" });
      return;
    }
  }

  function ensureStudent(id) {
    const sid = String(id);
    const existing = state.students.get(sid);
    if (existing) return existing;

    const card = document.createElement("button");
    card.type = "button";
    card.className = "student-card";
    card.dataset.sid = sid;

    const avatar = document.createElement("canvas");
    avatar.className = "avatar";
    avatar.width = 46;
    avatar.height = 46;

    const main = document.createElement("div");
    main.className = "student-main";

    const top = document.createElement("div");
    top.className = "student-top";

    const name = document.createElement("div");
    name.className = "student-name";
    name.textContent = `学生 ${sid}`;

    const badge = document.createElement("span");
    badge.className = "badge";
    badge.innerHTML = `<span class="dot unknown"></span><span>未知</span>`;

    top.appendChild(name);
    top.appendChild(badge);

    const meta = document.createElement("div");
    meta.className = "meta-row";

    const metaSeen = document.createElement("span");
    metaSeen.innerHTML = `last <code>—</code>`;

    const metaEar = document.createElement("span");
    metaEar.innerHTML = `EAR <code>—</code>`;

    const metaPitch = document.createElement("span");
    metaPitch.innerHTML = `pitch <code>—</code>`;

    meta.appendChild(metaSeen);
    meta.appendChild(metaEar);
    meta.appendChild(metaPitch);

    main.appendChild(top);
    main.appendChild(meta);

    card.appendChild(avatar);
    card.appendChild(main);

    card.addEventListener("click", () => {
      selectStudent(sid);
    });

    const student = {
      id: sid,
      state: "unknown",
      lastSeen: 0,
      bbox: null,
      ear: null,
      pitch: null,
      blinkCount: 0,
      history: [],
      dom: { card, avatar, name, badge, metaSeen, metaEar, metaPitch },
      lastAvatarTs: -1e9,
    };

    state.students.set(sid, student);
    if (!studentList.querySelector(".student-card")) {
      studentList.innerHTML = "";
    }
    insertStudentCardSorted(card);
    if (!state.selectedStudentId) selectStudent(sid);
    return student;
  }

  function insertStudentCardSorted(card) {
    const sid = card.dataset.sid || "";
    const sidNum = Number(sid);
    const children = Array.from(studentList.querySelectorAll(".student-card"));
    const idx = children.findIndex((el) => {
      const other = el.dataset.sid || "";
      const otherNum = Number(other);
      if (Number.isFinite(sidNum) && Number.isFinite(otherNum)) return sidNum < otherNum;
      return sid.localeCompare(other) < 0;
    });
    if (idx === -1) studentList.appendChild(card);
    else studentList.insertBefore(card, children[idx]);
  }

  function selectStudent(sid) {
    state.selectedStudentId = sid;
    selectedStudentText.textContent = sid ? `学生 ${sid}` : "—";

    for (const s of state.students.values()) {
      s.dom.card.classList.toggle("selected", s.id === sid);
    }
    scheduleRender({ timeline: true });
  }

  function updateStudentDom(s) {
    const st = normalizeState(s.state);
    const dotCls = dotClassForState(st);
    const badgeText = stateLabel(st);
    s.dom.badge.innerHTML = `<span class="dot ${dotCls}"></span><span>${badgeText}</span>`;

    s.dom.metaSeen.innerHTML = `last <code>${formatSec(Math.max(0, state.nowSec - (s.lastSeen || 0)))}</code>`;
    const earTxt = s.ear == null ? "—" : toNum(s.ear).toFixed(3);
    const pitchTxt = s.pitch == null ? "—" : `${toNum(s.pitch).toFixed(1)}°`;
    s.dom.metaEar.innerHTML = `EAR <code>${earTxt}</code>`;
    s.dom.metaPitch.innerHTML = `pitch <code>${pitchTxt}</code>`;
  }

  function maybeUpdateAvatar(s, img) {
    if (!img || !s.bbox || !Array.isArray(s.bbox) || s.bbox.length < 4) return;
    const now = state.nowSec;
    if (now - s.lastAvatarTs < 0.8) return;

    const [nx, ny, nw, nh] = s.bbox.map((v) => toNum(v, 0));
    const iw = img.naturalWidth || img.width || 1;
    const ih = img.naturalHeight || img.height || 1;

    const pad = 0.18;
    const cx = nx + nw / 2;
    const cy = ny + nh / 2;
    const size = Math.max(nw, nh) * (1 + pad * 2);

    const sx0 = clamp(cx - size / 2, 0, 1);
    const sy0 = clamp(cy - size / 2, 0, 1);
    const sx1 = clamp(cx + size / 2, 0, 1);
    const sy1 = clamp(cy + size / 2, 0, 1);
    const sx = sx0 * iw;
    const sy = sy0 * ih;
    const sw = Math.max(1, (sx1 - sx0) * iw);
    const sh = Math.max(1, (sy1 - sy0) * ih);

    const ctx = s.dom.avatar.getContext("2d");
    try {
      ctx.clearRect(0, 0, s.dom.avatar.width, s.dom.avatar.height);
      ctx.drawImage(img, sx, sy, sw, sh, 0, 0, s.dom.avatar.width, s.dom.avatar.height);
    } catch (_) {
      // ignore
    }
    s.lastAvatarTs = now;
  }

  function handleFrameData(data) {
    const ts = toNum(data.ts, state.nowSec);
    state.nowSec = ts;
    state.lastFrame.ts = ts;
    nowText.textContent = formatSec(ts);

    const faces = Array.isArray(data.faces) ? data.faces : [];
    state.lastFrame.faces = faces;

    if (data.image_base64) {
      const b64 = String(data.image_base64);
      if (b64 !== state.lastFrame.b64) {
        state.lastFrame.b64 = b64;
        updatePreviewImageFromB64(b64);
      }
    }

    for (const f of faces) {
      const sid = String(f.track_id ?? f.student_id ?? "");
      if (!sid) continue;
      const s = ensureStudent(sid);
      const newState = normalizeState(f.state);
      if (s.state !== newState) {
        s.history.push({ ts, state: newState });
        if (s.history.length > 8000) s.history.splice(0, s.history.length - 8000);
      }
      s.state = newState;
      s.lastSeen = ts;
      s.bbox = f.bbox || null;
      s.ear = f.ear ?? null;
      s.pitch = f.pitch ?? null;
      s.blinkCount = f.blink_count ?? s.blinkCount;
      updateStudentDom(s);
    }

    // Mark stale students as "not seen"
    for (const s of state.students.values()) {
      if (state.nowSec - (s.lastSeen || 0) > 3.5) {
        s.dom.metaSeen.innerHTML = `last <code class="warn-text">${formatSec(state.nowSec - (s.lastSeen || 0))}</code>`;
      }
    }

    // Update live ASR line if available
    updateAsrNowLine();
    scheduleRender({ preview: true, timeline: true });
  }

  const previewImage = new Image();
  previewImage.decoding = "async";
  previewImage.loading = "eager";
  previewImage.onload = () => {
    state.lastFrame.image = previewImage;
    scheduleRender({ preview: true });
    // update avatars using the latest frame
    for (const s of state.students.values()) {
      maybeUpdateAvatar(s, previewImage);
    }
  };
  previewImage.onerror = () => {
    // ignore
  };

  function updatePreviewImageFromB64(b64) {
    previewImage.src = `data:image/jpeg;base64,${b64}`;
  }

  function drawPreview() {
    resizeCanvasToElement(previewCanvas);

    const img = state.lastFrame.image;
    const cw = previewCanvas.width;
    const ch = previewCanvas.height;
    previewCtx.clearRect(0, 0, cw, ch);

    if (!img) {
      previewCtx.fillStyle = "rgba(0,0,0,0.25)";
      previewCtx.fillRect(0, 0, cw, ch);
      return;
    }

    const iw = img.naturalWidth || img.width || 1;
    const ih = img.naturalHeight || img.height || 1;
    const scale = Math.min(cw / iw, ch / ih);
    const dw = iw * scale;
    const dh = ih * scale;
    const dx = (cw - dw) / 2;
    const dy = (ch - dh) / 2;
    previewCtx.drawImage(img, 0, 0, iw, ih, dx, dy, dw, dh);

    if (!state.showBoxes && !state.showLabels) return;

    const dpr = window.devicePixelRatio || 1;
    previewCtx.lineWidth = Math.max(1, Math.round(2 * dpr));
    previewCtx.font = `${Math.max(12, Math.round(13 * dpr))}px ${getComputedStyle(document.body).fontFamily}`;
    previewCtx.textBaseline = "top";

    const faces = state.lastFrame.faces || [];
    for (const f of faces) {
      const sid = String(f.track_id ?? f.student_id ?? "");
      const bbox = f.bbox;
      if (!Array.isArray(bbox) || bbox.length < 4) continue;
      const st = normalizeState(f.state);
      const color = COLORS[st] || COLORS.unknown;
      const x = dx + toNum(bbox[0]) * dw;
      const y = dy + toNum(bbox[1]) * dh;
      const w = toNum(bbox[2]) * dw;
      const h = toNum(bbox[3]) * dh;

      if (state.showBoxes) {
        previewCtx.strokeStyle = color;
        previewCtx.strokeRect(x, y, w, h);
      }
      if (state.showLabels) {
        const label = sid ? `#${sid} ${stateLabel(st)}` : stateLabel(st);
        const padX = 6 * dpr;
        const padY = 4 * dpr;
        const textW = previewCtx.measureText(label).width;
        const boxW = textW + padX * 2;
        const boxH = Math.max(18 * dpr, 18 * dpr);
        const by = Math.max(0, y - boxH - 4 * dpr);
        previewCtx.fillStyle = "rgba(0,0,0,0.55)";
        previewCtx.fillRect(x, by, boxW, boxH);
        previewCtx.fillStyle = "#fff";
        previewCtx.fillText(label, x + padX, by + padY);
      }
    }

    // FPS update
    const nowMs = performance.now();
    const sec = Math.floor(nowMs / 1000);
    if (sec !== state.fps.lastSec) {
      state.fps.value = state.fps.framesThisSec;
      state.fps.framesThisSec = 0;
      state.fps.lastSec = sec;
      fpsText.textContent = `${state.fps.value} fps`;
      fpsDot.className = `dot ${state.fps.value >= 12 ? "good" : state.fps.value >= 6 ? "warn" : "unknown"}`;
    }
    state.fps.framesThisSec += 1;
  }

  function handleAsrSegment(seg) {
    const start = toNum(seg.start ?? seg.ts, null);
    if (start == null) return;
    const end = toNum(seg.end, start + 2.0);
    const text = String(seg.text || "").trim();
    const label = String(seg.label || "teacher").trim().toLowerCase();
    const rec = {
      start,
      end: Math.max(end, start + 0.2),
      text,
      label: label === "student" ? "student" : "teacher",
    };
    state.asrSegments.push(rec);
    // keep only recent segments
    const cutoff = state.nowSec - Math.max(600, state.windowSec * 6);
    if (state.asrSegments.length > 2000) {
      state.asrSegments = state.asrSegments.filter((s) => s.end >= cutoff);
    } else {
      while (state.asrSegments.length > 0 && state.asrSegments[0].end < cutoff) state.asrSegments.shift();
    }

    updateAsrNowLine();
    scheduleRender({ timeline: true });
  }

  function findAsrAt(t) {
    // most recent segment covering t
    for (let i = state.asrSegments.length - 1; i >= 0; i--) {
      const s = state.asrSegments[i];
      if (t >= s.start && t <= s.end) return s;
    }
    return null;
  }

  function updateAsrNowLine() {
    if (!state.showAsr) {
      asrNowText.textContent = "（ASR 已关闭）";
      return;
    }
    const seg = findAsrAt(state.nowSec);
    if (!seg || !seg.text) {
      asrNowText.textContent = "（暂无 ASR 段）";
      return;
    }
    const prefix = seg.label === "teacher" ? "老师：" : "学生：";
    asrNowText.textContent = `${prefix}${seg.text}`;
  }

  function getStudentStateAt(s, t) {
    if (!s || !Array.isArray(s.history) || s.history.length === 0) return "unknown";
    let last = null;
    for (let i = 0; i < s.history.length; i++) {
      const h = s.history[i];
      if (h.ts <= t) last = h;
      else break;
    }
    return normalizeState(last ? last.state : "unknown");
  }

  function renderTimeline() {
    resizeCanvasToElement(timelineCanvas);
    const W = timelineCanvas.width;
    const H = timelineCanvas.height;
    timelineCtx.clearRect(0, 0, W, H);

    const endT = state.followLive ? state.nowSec : state.cursorTime ?? state.nowSec;
    const startT = endT - state.windowSec;

    // background
    timelineCtx.fillStyle = "rgba(255,255,255,0.03)";
    timelineCtx.fillRect(0, 0, W, H);

    // grid / axis
    timelineCtx.strokeStyle = "rgba(255,255,255,0.10)";
    timelineCtx.lineWidth = 1;
    timelineCtx.beginPath();
    const step = state.windowSec >= 300 ? 30 : state.windowSec >= 120 ? 10 : 5;
    for (let t = Math.ceil(startT / step) * step; t <= endT; t += step) {
      const x = ((t - startT) / state.windowSec) * W;
      timelineCtx.moveTo(x, 0);
      timelineCtx.lineTo(x, H);
    }
    timelineCtx.stroke();

    const dpr = window.devicePixelRatio || 1;
    timelineCtx.font = `${Math.max(11, Math.round(12 * dpr))}px ${getComputedStyle(document.body).fontFamily}`;
    timelineCtx.fillStyle = "rgba(255,255,255,0.65)";
    timelineCtx.textBaseline = "top";
    for (let t = Math.ceil(startT / step) * step; t <= endT; t += step) {
      const x = ((t - startT) / state.windowSec) * W;
      const label = `${Math.max(0, t).toFixed(0)}s`;
      timelineCtx.fillText(label, x + 4 * dpr, 6 * dpr);
    }

    // rows
    const rowStateY = 34 * dpr;
    const rowStateH = 46 * dpr;
    const rowAsrY = rowStateY + rowStateH + 14 * dpr;
    const rowAsrH = 38 * dpr;

    // selected student state segments
    const sid = state.selectedStudentId;
    const s = sid ? state.students.get(sid) : null;
    timelineCtx.fillStyle = "rgba(255,255,255,0.10)";
    timelineCtx.fillRect(0, rowStateY, W, rowStateH);

    if (s && Array.isArray(s.history) && s.history.length > 0) {
      // ensure sorted (history should be chronological)
      const hist = s.history;
      for (let i = 0; i < hist.length; i++) {
        const cur = hist[i];
        const next = hist[i + 1] || { ts: endT };
        const segStart = clamp(cur.ts, startT, endT);
        const segEnd = clamp(next.ts, startT, endT);
        if (segEnd <= segStart) continue;

        const x1 = ((segStart - startT) / state.windowSec) * W;
        const x2 = ((segEnd - startT) / state.windowSec) * W;
        const st = normalizeState(cur.state);
        timelineCtx.fillStyle = COLORS[st] || COLORS.unknown;
        timelineCtx.fillRect(x1, rowStateY, Math.max(2, x2 - x1), rowStateH);
      }

      timelineCtx.fillStyle = "rgba(0,0,0,0.45)";
      timelineCtx.fillRect(10 * dpr, rowStateY + 10 * dpr, 150 * dpr, 22 * dpr);
      timelineCtx.fillStyle = "#fff";
      timelineCtx.fillText(`学生 ${sid} 状态`, 16 * dpr, rowStateY + 14 * dpr);
    } else {
      timelineCtx.fillStyle = "rgba(255,255,255,0.55)";
      timelineCtx.fillText("未选择学生或暂无状态数据", 14 * dpr, rowStateY + 14 * dpr);
    }

    // ASR row
    timelineCtx.fillStyle = "rgba(255,255,255,0.10)";
    timelineCtx.fillRect(0, rowAsrY, W, rowAsrH);

    if (state.showAsr && state.asrSegments.length > 0) {
      const segs = state.asrSegments;
      for (let i = 0; i < segs.length; i++) {
        const seg = segs[i];
        if (seg.end < startT || seg.start > endT) continue;
        const segStart = clamp(seg.start, startT, endT);
        const segEnd = clamp(seg.end, startT, endT);
        const x1 = ((segStart - startT) / state.windowSec) * W;
        const x2 = ((segEnd - startT) / state.windowSec) * W;
        const color = seg.label === "teacher" ? COLORS.teacher : COLORS.student;
        timelineCtx.fillStyle = color;
        timelineCtx.fillRect(x1, rowAsrY, Math.max(2, x2 - x1), rowAsrH);
      }

      const segNow = findAsrAt(endT);
      timelineCtx.fillStyle = "rgba(0,0,0,0.45)";
      timelineCtx.fillRect(10 * dpr, rowAsrY + 8 * dpr, 150 * dpr, 22 * dpr);
      timelineCtx.fillStyle = "#fff";
      timelineCtx.fillText("ASR 讲解", 16 * dpr, rowAsrY + 12 * dpr);

      if (segNow && segNow.text) {
        timelineCtx.fillStyle = "rgba(255,255,255,0.70)";
        const snippet = segNow.text.length > 28 ? `${segNow.text.slice(0, 28)}…` : segNow.text;
        timelineCtx.fillText(snippet, 170 * dpr, rowAsrY + 12 * dpr);
      }
    } else {
      timelineCtx.fillStyle = "rgba(255,255,255,0.55)";
      timelineCtx.fillText(state.showAsr ? "暂无 ASR 段（可通过 /push 注入 asr_segment）" : "ASR 显示已关闭", 14 * dpr, rowAsrY + 12 * dpr);
    }

    // cursor marker
    if (state.cursorTime != null) {
      const x = ((state.cursorTime - startT) / state.windowSec) * W;
      timelineCtx.strokeStyle = "rgba(56,189,248,0.9)";
      timelineCtx.lineWidth = Math.max(1, Math.round(2 * dpr));
      timelineCtx.beginPath();
      timelineCtx.moveTo(x, 0);
      timelineCtx.lineTo(x, H);
      timelineCtx.stroke();
    }

    // update cursor inspector (if any)
    updateCursorInspector();
  }

  function updateCursorInspector() {
    if (state.cursorTime == null) {
      cursorInspector.style.display = "none";
      btnClearCursor.disabled = true;
      btnFollow.textContent = "跟随实时";
      return;
    }

    btnClearCursor.disabled = false;
    btnFollow.textContent = "回到实时";
    cursorInspector.style.display = "flex";

    const t = state.cursorTime;
    cursorTimeText.textContent = formatSec(t);

    const sid = state.selectedStudentId;
    const s = sid ? state.students.get(sid) : null;
    const st = getStudentStateAt(s, t);
    cursorStateText.textContent = stateLabel(st);
    cursorStateText.className = dotClassForState(st) === "good" ? "ok-text" : dotClassForState(st) === "warn" ? "warn-text" : "danger-text";

    const seg = state.showAsr ? findAsrAt(t) : null;
    cursorAsrText.textContent = seg && seg.text ? seg.text : "—";
  }

  function clearAllLiveState() {
    state.students.clear();
    state.selectedStudentId = null;
    state.asrSegments = [];
    state.lastFrame = { image: null, ts: 0, faces: [], b64: null };
    studentList.innerHTML = `<div class="muted" style="padding:8px 4px;">等待人脸轨迹数据…</div>`;
    selectedStudentText.textContent = "—";
    nowText.textContent = "0.00s";
    asrNowText.textContent = "（暂无 ASR 段）";
    state.cursorTime = null;
    state.followLive = true;
    updateCursorInspector();
  }

  async function fetchJson(url, options, timeoutMs = 10000) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const res = await fetch(url, { ...options, signal: controller.signal });
      clearTimeout(timeoutId);
      const text = await res.text();
      let data = null;
      try {
        data = text ? JSON.parse(text) : null;
      } catch (_) {
        data = null;
      }
      if (!res.ok) {
        const err = (data && (data.error || data.detail)) || text || res.statusText;
        throw new Error(err);
      }
      return data;
    } catch (e) {
      clearTimeout(timeoutId);
      if (e.name === "AbortError") {
        throw new Error("Request timeout");
      }
      throw e;
    }
  }

  function isOkOrSkipped(obj) {
    if (!obj || typeof obj !== "object") return false;
    return obj.ok === true || obj.skipped === true;
  }

  function updateModelPill() {
    const cfg = state.models.config;
    if (!cfg) {
      modelDot.className = "dot unknown";
      modelStatusText.textContent = "模型设置";
      return;
    }

    const mode = String(cfg.mode || "offline");
    if (mode === "offline") {
      modelDot.className = "dot warn";
      modelStatusText.textContent = state.models.dirty ? "模型设置*" : "模型设置";
      return;
    }

    const check = state.models.lastCheck;
    if (!check) {
      modelDot.className = "dot unknown";
      modelStatusText.textContent = state.models.dirty ? "模型设置*" : "模型设置";
      return;
    }

    const llmOk = isOkOrSkipped(check.llm);
    const asrOk = isOkOrSkipped(check.asr);
    modelDot.className = `dot ${llmOk && asrOk ? "good" : "bad"}`;
    modelStatusText.textContent = `模型设置${state.models.dirty ? "*" : ""}`;
  }

  function renderModelEnvBox() {
    const env = state.models.env;
    if (!env) {
      modelEnvBox.textContent = "（未加载 env 信息）";
      return;
    }
    const chips = [
      `OpenAI key: ${env.has_openai_key ? "yes" : "no"}`,
      `DashScope key: ${env.has_dashscope_key ? "yes" : "no"}`,
      `Xfyun: ${env.has_xfyun ? "yes" : "no"}`,
    ]
      .map((t) => `<span class="chip">${escapeHtml(t)}</span>`)
      .join("");
    modelEnvBox.innerHTML = `
      <div class="chips">${chips}</div>
      <div class="muted" style="margin-top:10px; line-height:1.55;">
        <div><code>OPENAI_BASE_URL</code>: <span class="mono">${escapeHtml(env.openai_base_url || "")}</span></div>
        <div><code>OPENAI_MODEL</code>: <span class="mono">${escapeHtml(env.openai_model || "")}</span></div>
        <div><code>OPENAI_ASR_MODEL</code>: <span class="mono">${escapeHtml(env.openai_asr_model || "")}</span></div>
      </div>
    `;
  }

  function renderModelCheckBox() {
    if (state.models.checking) {
      modelCheckBox.textContent = "检测中…";
      return;
    }
    const cfg = state.models.config || {};
    const env = state.models.env || {};
    const mode = String(cfg.mode || "offline");
    const llmProvider = String(cfg.llm?.provider || "none");
    const llmEnabled = Boolean(cfg.llm?.enabled) && llmProvider !== "none";
    const llmModel = String(cfg.llm?.model || "").trim();
    const llmBaseUrl = String(cfg.llm?.base_url || "").trim();
    const hasOpenaiKey = Boolean(String(cfg.llm?.api_key || "").trim()) || Boolean(env.has_openai_key);
    const asrProvider = String(cfg.asr?.provider || "none");
    const asrModel = String(cfg.asr?.model || "").trim();
    const asrUseIndep = Boolean(cfg.asr?.use_independent);
    const asrApiKey = String(cfg.asr?.api_key || "").trim();
    const hasAsrKey = asrUseIndep ? (Boolean(asrApiKey) || hasOpenaiKey) : hasOpenaiKey;

    const r = state.models.lastCheck;
    if (!r) {
      const chips = [];
      chips.push(`mode: ${mode}`);
      chips.push(llmEnabled ? `llm: on${llmModel ? ` (${llmModel})` : ""}` : "llm: off");
      if (asrProvider && asrProvider !== "none") {
        chips.push(`asr: ${asrProvider}${asrProvider !== "xfyun_raasr" && asrModel ? ` (${asrModel})` : ""}`);
      } else {
        chips.push("asr: none");
      }

      const hints = [];
      const envDefaultBaseUrl = String(env.openai_base_url || "https://api.openai.com");
      const envDefaultModel = String(env.openai_model || "gpt-4o-mini");
      const envDefaultAsrModel = String(env.openai_asr_model || "whisper-1");
      if (mode === "online") {
        if (llmEnabled) {
          if (!llmModel) hints.push(`LLM：模型名为空会回退为 ${envDefaultModel}`);
          if (!llmBaseUrl) hints.push(`LLM：Base URL 为空会回退为 ${envDefaultBaseUrl}`);
          if (!hasOpenaiKey) hints.push("LLM：缺少 API Key（填写或设置 OPENAI_API_KEY/OPENAI_KEY）");
        }
        if (asrProvider === "openai_compat" && !hasAsrKey) {
          hints.push(asrUseIndep
            ? "ASR：使用独立设置但缺少 API Key（填写 ASR API Key 或 LLM API Key）"
            : "ASR：OpenAI-compatible 需要 API Key（填写 LLM API Key 或设置 OPENAI_API_KEY）");
        }
        if (asrProvider === "openai_compat" && !asrModel) {
          hints.push(`ASR：模型名为空会回退为 ${envDefaultAsrModel}`);
        }
        if (asrProvider === "dashscope" && !env.has_dashscope_key) {
          hints.push("ASR：DashScope 需要 DASH_SCOPE_API_KEY");
        }
        if (asrProvider === "xfyun_raasr" && !env.has_xfyun) {
          hints.push("ASR：讯飞需要 XFYUN_APP_ID/XFYUN_SECRET_KEY");
        }
      }

      modelCheckBox.innerHTML = `
        <div class="chips">${chips.map((c) => `<span class="chip">${escapeHtml(c)}</span>`).join("")}</div>
        <div class="muted" style="margin-top:10px; line-height:1.55;">
          尚未检测。点击“检测”可验证当前配置是否可用。
        </div>
        ${
          hints.length
            ? `<div class="muted" style="margin-top:10px; line-height:1.55;">
                <div><strong>配置提示：</strong></div>
                <div>${hints.slice(0, 8).map((t) => `• ${escapeHtml(t)}`).join("<br/>")}</div>
               </div>`
            : ""
        }
      `;
      return;
    }
    const llm = r.llm || {};
    const asr = r.asr || {};
    const llmOk = isOkOrSkipped(llm);
    const asrOk = isOkOrSkipped(asr);
    const llmHints = Array.isArray(llm.hints) ? llm.hints : [];
    const asrHints = Array.isArray(asr.hints) ? asr.hints : [];

    const lines = [];
    const llmLine = llm.skipped
      ? `LLM: skipped (${llm.reason || "disabled"})`
      : llmOk
        ? `LLM: ok${llm.latency_ms != null ? ` (${llm.latency_ms}ms)` : ""}`
        : `LLM: failed (${llm.error || "unknown"})`;
    const asrLine = asr.skipped
      ? `ASR: skipped (${asr.reason || "disabled"})`
      : asrOk
        ? `ASR: ok${asr.latency_ms != null ? ` (${asr.latency_ms}ms)` : ""}`
        : `ASR: failed (${asr.error || "unknown"})`;
    lines.push(llmLine);
    lines.push(asrLine);

    const suggested = state.models.suggestedMode;
    const dirtyHtml = state.models.dirty
      ? `<div class="warn-text" style="margin-top:8px;">配置已修改，检测结果可能已过期；建议重新“检测”。</div>`
      : "";
    const suggestHtml =
      suggested === "offline"
        ? `<div class="warn-text" style="margin-top:8px;">建议切换为 offline（录制不受影响，但报告将跳过 ASR/LLM）。</div>`
        : "";

    const hintItems = [];
    if (!llmOk && llmHints.length) {
      for (const h of llmHints.slice(0, 6)) hintItems.push(`LLM：${String(h)}`);
    }
    if (!asrOk && asrHints.length) {
      for (const h of asrHints.slice(0, 6)) hintItems.push(`ASR：${String(h)}`);
    }
    const hintHtml =
      hintItems.length > 0
        ? `<div class="muted" style="margin-top:10px; line-height:1.55;">
            <div><strong>可用提示：</strong></div>
            <div>${hintItems.map((t) => `• ${escapeHtml(t)}`).join("<br/>")}</div>
           </div>`
        : "";

    modelCheckBox.innerHTML = `
      <div class="chips">
        <span class="chip">mode: ${escapeHtml(String(r.mode || ""))}</span>
        <span class="chip">${escapeHtml(llmOk ? "LLM ✅" : "LLM ❌")}</span>
        <span class="chip">${escapeHtml(asrOk ? "ASR ✅" : "ASR ❌")}</span>
      </div>
      <div class="codebox" style="margin-top:10px;">${escapeHtml(lines.join("\n"))}</div>
      ${hintHtml}
      ${dirtyHtml}
      ${suggestHtml}
    `;
  }

  function applyModelFormFromState() {
    const cfg = state.models.config;
    if (!cfg) return;
    modelModeSel.value = String(cfg.mode || "offline");
    asrProviderSel.value = String(cfg.asr?.provider || "none");
    asrModelInput.value = String(cfg.asr?.model || "");
    if (asrUseIndependentToggle) asrUseIndependentToggle.checked = Boolean(cfg.asr?.use_independent);
    if (asrBaseUrlInput) asrBaseUrlInput.value = String(cfg.asr?.base_url || "");
    if (asrApiKeyInput) asrApiKeyInput.value = String(cfg.asr?.api_key || "");
    llmEnabledToggle.checked = Boolean(cfg.llm?.enabled);
    llmBaseUrlInput.value = String(cfg.llm?.base_url || "");
    llmApiKeyInput.value = String(cfg.llm?.api_key || "");
    llmModelInput.value = String(cfg.llm?.model || "");
    renderLlmModelsSelect();
    renderAsrModelsSelect();  // Explicitly sync ASR model dropdown
    updateModelFormDisabledState();
    updateLlmSectionDisabledState();
  }

  function updateModelFormDisabledState() {
    const cfg = state.models.config || {};
    const asrProvider = String(asrProviderSel.value || cfg.asr?.provider || "none");
    const asrUseIndependent = asrUseIndependentToggle && asrUseIndependentToggle.checked;
    const busy = Boolean(state.models.checking || state.models.llmModelsLoading);

    modelModeSel.disabled = busy;

    llmEnabledToggle.disabled = busy;
    llmBaseUrlInput.disabled = busy;
    llmApiKeyInput.disabled = busy;
    btnLlmPullModels.disabled = busy;
    llmModelSel.disabled = busy || !(state.models.llmModels && state.models.llmModels.length);
    llmModelInput.disabled = busy;

    asrProviderSel.disabled = busy;
    const asrModelDisabled = busy || asrProvider === "none" || asrProvider === "xfyun_raasr";
    asrModelInput.disabled = asrModelDisabled;
    if (asrModelSel) asrModelSel.disabled = asrModelDisabled || !(state.models.llmModels && state.models.llmModels.length);
    if (asrUseIndependentToggle) asrUseIndependentToggle.disabled = busy || asrProvider !== "openai_compat";
    if (asrBaseUrlInput) asrBaseUrlInput.disabled = busy || !asrUseIndependent;
    if (asrApiKeyInput) asrApiKeyInput.disabled = busy || !asrUseIndependent;
    if (asrIndependentSettings) {
      asrIndependentSettings.style.display = asrUseIndependent && asrProvider === "openai_compat" ? "" : "none";
    }

    btnModelsSave.disabled = busy;
    btnModelsCheck.disabled = busy;
    btnModelsCheckDeep.disabled = busy;
  }

  function renderLlmModelsSelect() {
    const cfg = state.models.config || {};
    const cur = String(cfg.llm?.model || "").trim();
    const models = Array.isArray(state.models.llmModels) ? state.models.llmModels : [];

    llmModelSel.innerHTML = "";
    const opt0 = document.createElement("option");
    opt0.value = "";
    opt0.textContent = models.length ? "选择模型…" : "（点击“拉取模型”）";
    llmModelSel.appendChild(opt0);

    for (const id of models) {
      const mid = String(id || "").trim();
      if (!mid) continue;
      const opt = document.createElement("option");
      opt.value = mid;
      opt.textContent = mid;
      llmModelSel.appendChild(opt);
    }

    // Case-insensitive matching for model selection
    const curLower = cur.toLowerCase();
    const matchIdx = models.findIndex((m) => String(m).toLowerCase() === curLower);
    if (cur && matchIdx >= 0) llmModelSel.value = models[matchIdx];
    else llmModelSel.value = "";

    if (!state.models.llmModelsLoading) {
      if (state.models.llmModelsError) {
        llmModelsStatus.innerHTML = `<span class="danger-text">${escapeHtml(String(state.models.llmModelsError))}</span>`;
      } else if (models.length) {
        llmModelsStatus.innerHTML = `<span class="ok-text">已拉取：${escapeHtml(String(models.length))} 个</span>`;
      } else {
        llmModelsStatus.textContent = "（未拉取）";
      }
    }

    // Also update ASR model select
    renderAsrModelsSelect();
  }

  function renderAsrModelsSelect() {
    if (!asrModelSel) return;
    const cfg = state.models.config || {};
    const cur = String(cfg.asr?.model || "").trim();
    const models = Array.isArray(state.models.llmModels) ? state.models.llmModels : [];

    asrModelSel.innerHTML = "";
    const opt0 = document.createElement("option");
    opt0.value = "";
    opt0.textContent = models.length ? "选择模型…" : "（先在 LLM 拉取模型）";
    asrModelSel.appendChild(opt0);

    for (const id of models) {
      const mid = String(id || "").trim();
      if (!mid) continue;
      const opt = document.createElement("option");
      opt.value = mid;
      opt.textContent = mid;
      asrModelSel.appendChild(opt);
    }

    // Case-insensitive matching for model selection
    const curLower = cur.toLowerCase();
    const matchIdx = models.findIndex((m) => String(m).toLowerCase() === curLower);
    if (cur && matchIdx >= 0) asrModelSel.value = models[matchIdx];
    else asrModelSel.value = "";
  }

  async function fetchJsonSoft(url, options) {
    const res = await fetch(url, options);
    const text = await res.text();
    let data = null;
    try {
      data = text ? JSON.parse(text) : null;
    } catch (_) {
      data = null;
    }
    return { ok: res.ok, status: res.status, data, text };
  }

  async function pullLlmModels() {
    if (!state.models.config) await loadModelsConfig();
    syncModelStateFromForm();

    state.models.llmModelsLoading = true;
    state.models.llmModelsError = null;
    updateModelFormDisabledState();
    llmModelsStatus.textContent = "正在拉取可用模型…";
    try {
      const res = await fetchJsonSoft("/api/models/list", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ config: state.models.config }),
      });
      const j = res.data;
      if (!res.ok || !j || j.ok !== true) {
        const err = (j && (j.error || j.detail)) || res.text || `HTTP ${res.status}`;
        const hints = j && Array.isArray(j.hints) ? j.hints : [];
        state.models.llmModelsError = String(err);
        llmModelsStatus.innerHTML = `<span class="danger-text">拉取失败：${escapeHtml(String(err))}</span>${
          hints.length
            ? `<div class="muted" style="margin-top:6px; line-height:1.55;">${hints
                .slice(0, 6)
                .map((h) => `• ${escapeHtml(String(h))}`)
                .join("<br/>")}</div>`
            : ""
        }`;
        return;
      }
      const models = Array.isArray(j.models) ? j.models : [];
      state.models.llmModels = models.map((m) => String(m || "").trim()).filter(Boolean);
      renderLlmModelsSelect();
      llmModelsStatus.innerHTML = `<span class="ok-text">已拉取：${escapeHtml(String(j.count ?? state.models.llmModels.length))} 个模型</span>`;
    } finally {
      state.models.llmModelsLoading = false;
      updateModelFormDisabledState();
      updateModelPill();
    }
  }

  function syncModelStateFromForm() {
    const existing = state.models.config && typeof state.models.config === "object" ? state.models.config : {};
    const llmEnabled = Boolean(llmEnabledToggle.checked);
    const asrProvider = String(asrProviderSel.value || "none");
    const asrUseIndependent = asrUseIndependentToggle && asrUseIndependentToggle.checked && asrProvider === "openai_compat";
    const cfg = {
      mode: String(modelModeSel.value || existing.mode || "offline"),
      llm: {
        ...(existing.llm && typeof existing.llm === "object" ? existing.llm : {}),
        enabled: llmEnabled,
        provider: llmEnabled ? "openai_compat" : (existing.llm?.provider || "openai_compat"),
        base_url: String(llmBaseUrlInput.value || "").trim(),
        api_key: String(llmApiKeyInput.value || "").trim(),
        model: String(llmModelInput.value || "").trim(),
      },
      asr: {
        ...(existing.asr && typeof existing.asr === "object" ? existing.asr : {}),
        provider: asrProvider,
        model: String(asrModelInput.value || "").trim(),
        use_independent: asrUseIndependent,
        base_url: asrUseIndependent && asrBaseUrlInput ? String(asrBaseUrlInput.value || "").trim() : "",
        api_key: asrUseIndependent && asrApiKeyInput ? String(asrApiKeyInput.value || "").trim() : "",
      },
    };
    state.models.config = cfg;
    state.models.dirty = true;
    updateModelFormDisabledState();
    updateModelPill();
    renderModelCheckBox();
  }

  async function loadModelsConfig() {
    const j = await fetchJson("/api/models/config");
    if (!j || !j.ok) throw new Error(j?.error || "load models config failed");
    state.models.config = j.config || null;
    state.models.env = j.env || null;
    state.models.providers = j.providers || null;
    state.models.dirty = false;
    applyModelFormFromState();
    renderModelEnvBox();
    updateModelPill();
  }

  async function saveModelsConfig() {
    if (!state.models.config) await loadModelsConfig();
    const j = await fetchJson("/api/models/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config: state.models.config }),
    });
    if (!j || !j.ok) throw new Error(j?.error || "save models config failed");
    state.models.config = j.config || state.models.config;
    state.models.env = j.env || state.models.env;
    state.models.dirty = false;
    applyModelFormFromState();
    renderModelEnvBox();
    updateModelPill();
  }

  async function checkModels({ deep = false } = {}) {
    if (!state.models.config) await loadModelsConfig();
    state.models.checking = true;
    renderModelCheckBox();
    updateModelFormDisabledState();
    try {
      const j = await fetchJson("/api/models/check", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ config: state.models.config, deep: Boolean(deep) }),
      });
      if (!j || !j.ok) throw new Error(j?.error || "check models failed");
      state.models.lastCheck = j.result || null;
      state.models.suggestedMode = j.suggested_mode || null;
      renderModelCheckBox();
      updateModelPill();
      return j;
    } finally {
      state.models.checking = false;
      renderModelCheckBox();
      updateModelFormDisabledState();
    }
  }

  function openModelModal() {
    modelModal.classList.add("open");
    applyModelFormFromState();
    renderModelEnvBox();
    renderModelCheckBox();
    initModelDrawerPos();
    setModelSettingsView("__all__", { preserveAllScroll: false });
  }

  function closeModelModal() {
    modelModal.classList.remove("open");
  }

  async function ensureModelsReadyBeforeRecording() {
    if (!state.models.config) {
      try {
        await loadModelsConfig();
      } catch (e) {
        // If the backend is old or the endpoint is missing, don't block recording.
        console.warn(e);
        return null;
      }
    }

    if (state.models.dirty) {
      await saveModelsConfig();
    }

    const cfg = state.models.config || {};
    const mode = String(cfg.mode || "offline");
    if (mode !== "online") return null;

    let chk = null;
    try {
      chk = await checkModels({ deep: false });
    } catch (e) {
      state.models.suggestedMode = "offline";
      renderModelCheckBox();
      updateModelPill();
      const ok = confirm(`模型检测请求失败：${String(e)}\n\n是否切换 offline 继续录制？`);
      if (!ok) throw new Error(`模型检测失败，已取消开始录制：${String(e)}`);

      const cur = state.models.config || {};
      state.models.config = {
        ...cur,
        mode: "offline",
        llm: { ...(cur.llm || {}), enabled: false },
        asr: { ...(cur.asr || {}), provider: "none" },
      };
      state.models.dirty = true;
      applyModelFormFromState();
      await saveModelsConfig();
      return null;
    }

    if (String(chk?.suggested_mode || "") === "offline") {
      const ok = confirm("模型检测未通过，建议切换 offline 继续录制（仍会录制音视频并生成 CV 统计）。\n\n确定切换为 offline 吗？");
      if (!ok) throw new Error("模型不可用，已取消开始录制。");

      state.models.config = {
        ...cfg,
        mode: "offline",
        llm: { ...(cfg.llm || {}), enabled: false },
        asr: { ...(cfg.asr || {}), provider: "none" },
      };
      state.models.dirty = true;
      applyModelFormFromState();
      await saveModelsConfig();
    }
    return chk;
  }

  async function startRecording() {
    setButtonLoading(btnStart, true);
    try {
      await ensureModelsReadyBeforeRecording();
      const j = await fetchJson("/api/session/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_config: state.models.config }),
      });
      if (!j || !j.ok) throw new Error(j?.error || "start failed");
      clearAllLiveState();
      setRecording(true, j.session_id);
      state.report.lastStatsUrl = null;
      state.report.lastSessionId = j.session_id;
      btnOpenReport.style.display = "none";
      showToast("录制已开始", "success");
      if (j.model_config) {
        state.models.config = j.model_config;
        state.models.dirty = false;
        applyModelFormFromState();
        updateModelPill();
      }
    } finally {
      setButtonLoading(btnStart, false);
      btnStart.disabled = state.isRecording;
    }
  }

  async function stopRecording() {
    setButtonLoading(btnStop, true);
    try {
      const j = await fetchJson("/api/session/stop", { method: "POST" });
      if (!j || !j.ok) throw new Error(j?.error || "stop failed");
      if (j.warning) {
        showToast(`录制已停止，但视频存在问题：${j.warning}`, "warning", 6000);
      } else {
        showToast("录制已停止", "success");
      }
      setRecording(false, state.sessionId);
      btnReport.disabled = !state.sessionId;
      await refreshSessions();
    } finally {
      setButtonLoading(btnStop, false);
      btnStop.disabled = !state.isRecording;
    }
  }

  function openReportModal() {
    reportModal.classList.add("open");
    initReportDrawerPos();
  }

  function closeReportModal() {
    reportModal.classList.remove("open");
    // Reset maximize state on close
    if (reportMaximized) {
      reportMaximized = false;
      if (reportDrawer) reportDrawer.classList.remove("maximized");
      if (maximizeIcon) maximizeIcon.textContent = "\u2922";
    }
  }

  function updateReportSessionInfo(sessionId, displayName = null) {
    if (reportSessionId) {
      reportSessionId.textContent = sessionId ? `ID: ${sessionId}` : "";
    }
    if (reportSessionName) {
      reportSessionName.value = displayName || "";
      reportSessionName.placeholder = displayName ? "编辑会话名称" : "输入会话名称";
    }
  }

  async function renameSession(sessionId, newName) {
    if (!sessionId || !newName.trim()) return false;
    try {
      const resp = await fetchJson("/api/session/rename", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, name: newName.trim() }),
      });
      if (resp?.ok) {
        showToast("会话已重命名", "success");
        // Refresh session list to show new name
        await refreshSessions();
        return true;
      } else {
        showToast(`重命名失败：${resp?.error || "unknown"}`, "error");
        return false;
      }
    } catch (e) {
      showToast(`重命名失败：${e}`, "error");
      return false;
    }
  }

  // ===== Settings Center Functions =====
  const SETTINGS_POS_KEY = "settings_drawer_pos_v1";
  let settingsDragState = null;

  function openSettingsModal() {
    settingsModal.classList.add("open");
    initSettingsDrawerPos();
  }

  function closeSettingsModal() {
    settingsModal.classList.remove("open");
    // Reset drag state on close
    settingsDragState = null;
    if (settingsDrawer) settingsDrawer.classList.remove("dragging");
  }

  function initSettingsDrawerPos() {
    if (!settingsDrawer) return;
    try {
      const raw = localStorage.getItem(SETTINGS_POS_KEY);
      if (raw) {
        const data = JSON.parse(raw);
        if (data?.left != null && data?.top != null) {
          settingsDrawer.style.left = `${data.left}px`;
          settingsDrawer.style.top = `${data.top}px`;
          return;
        }
      }
    } catch {}
    settingsDrawer.style.left = "10vw";
    settingsDrawer.style.top = "8vh";
  }

  function startSettingsDrag(e) {
    if (!settingsDrawer) return;
    const rect = settingsDrawer.getBoundingClientRect();
    settingsDragState = {
      startX: e.clientX,
      startY: e.clientY,
      startLeft: rect.left,
      startTop: rect.top,
      pointerId: e.pointerId,
    };
    settingsDrawer.classList.add("dragging");
    settingsDragHandle.setPointerCapture(e.pointerId);
  }

  function moveSettingsDrag(e) {
    if (!settingsDragState || e.pointerId !== settingsDragState.pointerId) return;
    const dx = e.clientX - settingsDragState.startX;
    const dy = e.clientY - settingsDragState.startY;
    const rect = settingsDrawer.getBoundingClientRect();
    const maxLeft = window.innerWidth - rect.width;
    const maxTop = window.innerHeight - rect.height;
    const left = clamp(settingsDragState.startLeft + dx, 0, Math.max(0, maxLeft));
    const top = clamp(settingsDragState.startTop + dy, 0, Math.max(0, maxTop));
    settingsDrawer.style.left = `${left}px`;
    settingsDrawer.style.top = `${top}px`;
  }

  function endSettingsDrag(e) {
    if (!settingsDragState || e.pointerId !== settingsDragState.pointerId) return;
    settingsDrawer.classList.remove("dragging");
    settingsDragHandle.releasePointerCapture(e.pointerId);
    const rect = settingsDrawer.getBoundingClientRect();
    try {
      localStorage.setItem(SETTINGS_POS_KEY, JSON.stringify({ left: rect.left, top: rect.top }));
    } catch {}
    settingsDragState = null;
  }

  function setActiveSettingsTab(tabId) {
    for (const btn of settingsNavBtns) {
      btn.classList.toggle("active", btn.dataset.tab === tabId);
    }
    for (const tab of settingsTabs) {
      tab.classList.toggle("active", tab.id === `settings-tab-${tabId}`);
    }
  }

  async function loadWebdavConfig() {
    try {
      // Use a shorter timeout (3s) since this is not critical
      const resp = await fetchJson("/api/webdav/config", undefined, 3000);
      if (resp?.ok) {
        state.webdav.config = resp.config || {};
        applyWebdavFormFromState();
        updateSettingsDot();
      }
    } catch (e) {
      console.warn("Failed to load WebDAV config:", e);
    }
  }

  function applyWebdavFormFromState() {
    const cfg = state.webdav.config || {};
    if (webdavEnabled) webdavEnabled.checked = Boolean(cfg.enabled);
    if (webdavUrl) webdavUrl.value = cfg.url || "";
    if (webdavUsername) webdavUsername.value = cfg.username || "";
    // Don't show redacted password placeholder - use empty or keep existing
    if (webdavPassword) {
      if (cfg.password && cfg.password !== "***") {
        webdavPassword.value = cfg.password;
      } else if (cfg.password === "***") {
        // Password exists on server but is redacted; keep field empty with placeholder
        webdavPassword.value = "";
        webdavPassword.placeholder = "••••••••（已保存）";
      } else {
        webdavPassword.value = "";
        webdavPassword.placeholder = "密码或应用专用密码";
      }
    }
    if (webdavRemotePath) webdavRemotePath.value = cfg.remote_path || "/classroom_focus";
    if (webdavAutoUpload) webdavAutoUpload.checked = Boolean(cfg.auto_upload);
    if (webdavUploadVideo) webdavUploadVideo.checked = cfg.upload_video !== false;
    if (webdavUploadAudio) webdavUploadAudio.checked = cfg.upload_audio !== false;
    if (webdavUploadStats) webdavUploadStats.checked = cfg.upload_stats !== false;
    if (webdavUploadTranscript) webdavUploadTranscript.checked = cfg.upload_transcript !== false;
    if (webdavUploadAll) webdavUploadAll.checked = Boolean(cfg.upload_all);
    updateWebdavFormDisabledState();
  }

  function syncWebdavStateFromForm() {
    const existingPassword = state.webdav.config?.password;
    const formPassword = webdavPassword?.value || "";
    // If password field is empty but server has a password, preserve it
    const passwordToSave = formPassword || (existingPassword === "***" ? "***" : existingPassword) || "";
    
    state.webdav.config = {
      enabled: webdavEnabled?.checked ?? false,
      url: webdavUrl?.value?.trim() || "",
      username: webdavUsername?.value?.trim() || "",
      password: passwordToSave,
      remote_path: webdavRemotePath?.value?.trim() || "/classroom_focus",
      auto_upload: webdavAutoUpload?.checked ?? false,
      upload_video: webdavUploadVideo?.checked ?? true,
      upload_audio: webdavUploadAudio?.checked ?? true,
      upload_stats: webdavUploadStats?.checked ?? true,
      upload_transcript: webdavUploadTranscript?.checked ?? true,
      upload_all: webdavUploadAll?.checked ?? false,
    };
    state.webdav.dirty = true;
  }

  function updateWebdavFormDisabledState() {
    const enabled = webdavEnabled?.checked ?? false;
    if (webdavFormGrid) {
      webdavFormGrid.classList.toggle("disabled", !enabled);
    }
  }

  async function saveWebdavConfig() {
    syncWebdavStateFromForm();
    setButtonLoading(btnSettingsSave, true);
    try {
      const resp = await fetchJson("/api/webdav/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ config: state.webdav.config }),
      });
      if (resp?.ok) {
        state.webdav.config = resp.config;
        state.webdav.dirty = false;
        showToast("设置已保存", "success");
        updateSettingsDot();
      } else {
        showToast(`保存失败：${resp?.error || "unknown"}`, "error");
      }
    } catch (e) {
      showToast(`保存失败：${e}`, "error");
    } finally {
      setButtonLoading(btnSettingsSave, false);
    }
  }

  async function testWebdavConnection() {
    syncWebdavStateFromForm();
    setButtonLoading(btnWebdavTest, true);
    if (webdavTestStatus) webdavTestStatus.textContent = "测试中…";
    try {
      const resp = await fetchJson("/api/webdav/test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ config: state.webdav.config }),
      });
      if (resp?.ok) {
        if (webdavTestStatus) webdavTestStatus.innerHTML = `<span class="ok-text">✓ ${resp.message || "连接成功"}</span>`;
        showToast("连接成功", "success");
      } else {
        if (webdavTestStatus) webdavTestStatus.innerHTML = `<span class="danger-text">✗ ${resp.error || "连接失败"}</span>`;
        showToast(`连接失败：${resp.error}`, "error");
      }
    } catch (e) {
      if (webdavTestStatus) webdavTestStatus.innerHTML = `<span class="danger-text">✗ ${e}</span>`;
      showToast(`连接失败：${e}`, "error");
    } finally {
      setButtonLoading(btnWebdavTest, false);
    }
  }

  function updateSettingsDot() {
    if (!settingsDot) return;
    const cfg = state.webdav.config || {};
    if (cfg.enabled && cfg.url && cfg.username) {
      settingsDot.className = "dot good";
    } else if (cfg.enabled) {
      settingsDot.className = "dot warn";
    } else {
      settingsDot.className = "dot unknown";
    }
  }

  function resetWebdavToDefaults() {
    state.webdav.config = {
      enabled: false,
      url: "",
      username: "",
      password: "",
      remote_path: "/classroom_focus",
      auto_upload: false,
      upload_video: true,
      upload_audio: true,
      upload_stats: true,
      upload_transcript: true,
      upload_all: false,
    };
    state.webdav.dirty = true;
    applyWebdavFormFromState();
    if (webdavTestStatus) webdavTestStatus.textContent = "";
    showToast("已重置为默认值", "success");
  }

  function toggleWebdavPasswordVisibility() {
    if (!webdavPassword || !btnToggleWebdavPassword) return;
    const isPassword = webdavPassword.type === "password";
    webdavPassword.type = isPassword ? "text" : "password";
    btnToggleWebdavPassword.textContent = isPassword ? "●" : "◉";
    btnToggleWebdavPassword.title = isPassword ? "隐藏密码" : "显示密码";
  }

  function validateWebdavUrl(url) {
    if (!url) return { valid: true, message: "" };
    const trimmed = url.trim();
    if (trimmed && !trimmed.startsWith("http://") && !trimmed.startsWith("https://")) {
      return { valid: false, message: "URL 应以 http:// 或 https:// 开头" };
    }
    return { valid: true, message: "" };
  }

  async function uploadSessionToWebdav(sessionId) {
    if (!sessionId) {
      showToast("没有可上传的会话", "warning");
      return;
    }
    const cfg = state.webdav.config || {};
    if (!cfg.enabled) {
      showToast("WebDAV 未启用，请先在设置中心启用", "warning");
      return;
    }
    setButtonLoading(btnUploadWebdav, true);
    try {
      const resp = await fetchJson("/api/webdav/upload", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      });
      if (resp?.ok) {
        showToast(`上传成功：${resp.uploaded || 0} 个文件`, "success");
      } else {
        showToast(`上传失败：${resp?.error || "unknown"}`, "error");
      }
    } catch (e) {
      showToast(`上传失败：${e}`, "error");
    } finally {
      setButtonLoading(btnUploadWebdav, false);
    }
  }

  function renderReportKpis(stats) {
    reportKpis.innerHTML = "";
    const per = stats?.per_student || {};
    const students = Object.keys(per);
    let totalIntervals = 0;
    let totalDur = 0;
    for (const sid of students) {
      const intervals = per[sid]?.intervals || [];
      totalIntervals += intervals.length;
      for (const it of intervals) {
        totalDur += Math.max(0, toNum(it.end) - toNum(it.start));
      }
    }

    const kpis = [
      { k: "学生数", v: String(students.length), s: "tracks in stats.json" },
      { k: "走神/睡觉区间", v: String(totalIntervals), s: "闭眼 / 低头 / 眼睛不可见" },
      { k: "总走神/睡觉时长", v: formatDuration(totalDur), s: "累计（秒）" },
      { k: "会话", v: String(stats?.session_id || "—"), s: "session_id" },
    ];
    for (const item of kpis) {
      const div = document.createElement("div");
      div.className = "kpi";
      div.innerHTML = `<div class="k">${escapeHtml(item.k)}</div><div class="v">${escapeHtml(item.v)}</div><div class="s">${escapeHtml(item.s)}</div>`;
      reportKpis.appendChild(div);
    }
  }

  function renderReportLinks(result) {
    reportLinks.innerHTML = "";
    const add = (href, label) => {
      if (!href) return;
      const a = document.createElement("a");
      a.href = href;
      a.target = "_blank";
      a.rel = "noreferrer";
      a.textContent = label;
      reportLinks.appendChild(a);
    };
    add(result?.video, "下载视频");
    add(result?.audio, "下载音频");
    add(result?.transcript, "下载转录");
    add(result?.lesson_summary, "下载课程总结");
    add(result?.stats, "下载统计 JSON");
  }

  function renderReportBody(stats) {
    const lesson = stats?.lesson_summary || null;
    const per = stats?.per_student || {};
    const warnings = Array.isArray(stats?.warnings) ? stats.warnings : [];
    const modelCfg = stats?.model_config && typeof stats.model_config === "object" ? stats.model_config : null;
    const studentIds = Object.keys(per).sort((a, b) => {
      const na = Number(a);
      const nb = Number(b);
      const fa = Number.isFinite(na);
      const fb = Number.isFinite(nb);
      if (fa && fb) return na - nb;
      if (fa && !fb) return -1;
      if (!fa && fb) return 1;
      return String(a).localeCompare(String(b), "zh-Hans-CN", { numeric: true });
    });
    const head = document.createElement("div");
    head.style.display = "flex";
    head.style.flexDirection = "column";
    head.style.gap = "10px";

    if (modelCfg || warnings.length) {
      const mode = String(modelCfg?.mode || "");
      const asrProvider = String(modelCfg?.asr?.provider || "");
      const asrModel = String(modelCfg?.asr?.model || "");
      const llmEnabled = Boolean(modelCfg?.llm?.enabled);
      const llmModel = String(modelCfg?.llm?.model || "");
      const llmBase = String(modelCfg?.llm?.base_url || "");

      const chips = [];
      if (mode) chips.push(`mode: ${mode}`);
      if (asrProvider) chips.push(`asr: ${asrProvider}${asrModel ? ` (${asrModel})` : ""}`);
      chips.push(llmEnabled ? `llm: on${llmModel ? ` (${llmModel})` : ""}` : "llm: off");

      const box = document.createElement("div");
      box.className = "interval";
      box.innerHTML = `
        <div class="interval-head">
          <div class="interval-title">模型/告警</div>
          <div class="interval-time">${escapeHtml(stats?.session_id || "")}</div>
        </div>
        <div class="interval-body">
          <div><strong>配置：</strong> <div class="chips">${chips.map((c) => `<span class="chip">${escapeHtml(c)}</span>`).join("")}</div></div>
          ${
            llmEnabled && llmBase
              ? `<div><strong>LLM Base URL：</strong> <span class="mono">${escapeHtml(llmBase)}</span></div>`
              : ""
          }
          ${
            warnings.length
              ? `<div><strong>告警：</strong><div class="muted">${warnings
                  .slice(0, 12)
                  .map((w) => `• ${escapeHtml(String(w))}`)
                  .join("<br/>")}</div></div>`
              : '<div><strong>告警：</strong> <span class="muted">（无）</span></div>'
          }
        </div>
      `;
      head.appendChild(box);
    }

    if (lesson && typeof lesson === "object") {
      const title = String(lesson.title || "").trim();
      const overview = String(lesson.overview || "").trim();
      const keyPoints = Array.isArray(lesson.key_points) ? lesson.key_points : [];
      const outline = Array.isArray(lesson.outline) ? lesson.outline : [];
      const timeline = Array.isArray(lesson.timeline) ? lesson.timeline : [];

      const box = document.createElement("div");
      box.className = "interval";
      box.innerHTML = `
        <div class="interval-head">
          <div class="interval-title">${escapeHtml(title || "本节课总结")}</div>
          <div class="interval-time">${escapeHtml(stats?.session_id || "")}</div>
        </div>
        <div class="interval-body">
          <div><strong>概览：</strong> ${overview ? escapeHtml(overview) : '<span class="muted">（未生成）</span>'}</div>
          <div>
            <strong>关键要点：</strong>
            ${
              keyPoints.length
                ? `<div class="chips">${keyPoints.slice(0, 12).map((k) => `<span class="chip">${escapeHtml(k)}</span>`).join("")}</div>`
                : '<span class="muted">（无）</span>'
            }
          </div>
          <div>
            <strong>大纲：</strong>
            ${
              outline.length
                ? `<div class="muted">${outline.slice(0, 12).map((x) => `• ${escapeHtml(x)}`).join("<br/>")}</div>`
                : '<span class="muted">（无）</span>'
            }
          </div>
          <div>
            <strong>时间线：</strong>
            ${
              timeline.length
                ? `<div class="muted">${timeline
                    .slice(0, 12)
                    .map((t) => `${escapeHtml(formatSec(t.start))}–${escapeHtml(formatSec(t.end))} · ${escapeHtml(t.topic || "（未命名主题）")}`)
                    .join("<br/>")}</div>`
                : '<span class="muted">（无）</span>'
            }
          </div>
        </div>
      `;
      head.appendChild(box);
    }

    if (studentIds.length === 0) {
      reportBody.innerHTML = "";
      reportBody.appendChild(head);
      const info = document.createElement("div");
      info.className = "muted";
      info.textContent = "没有检测到走神/疑似睡觉区间。";
      head.appendChild(info);
      return;
    }

    const wrap = document.createElement("div");
    wrap.style.display = "flex";
    wrap.style.flexDirection = "column";
    wrap.style.gap = "10px";

    for (const sid of studentIds) {
      const info = per[sid] || {};
      const intervals = Array.isArray(info.intervals) ? info.intervals : [];
      const total = intervals.reduce((acc, it) => acc + Math.max(0, toNum(it.end) - toNum(it.start)), 0);

      const det = document.createElement("details");
      det.open = true;
      const sum = document.createElement("summary");
      sum.innerHTML = `
        <div class="summary-left">
          <div class="student-name">学生 ${escapeHtml(sid)}</div>
          <span class="badge"><span class="dot warn"></span><span>${intervals.length} 段</span></span>
        </div>
        <div class="summary-right">累计 ${escapeHtml(formatDuration(total))}</div>
      `;
      det.appendChild(sum);

      const list = document.createElement("div");
      list.className = "intervals";
      if (intervals.length === 0) {
        list.innerHTML = `<div class="muted">没有检测到异常状态。</div>`;
      } else {
        for (const it of intervals) {
          const type = String(it.type || "UNKNOWN");
          const start = toNum(it.start);
          const end = toNum(it.end);
          const dur = Math.max(0, end - start);
          const asr = String(it.asr_text || "").trim();
          const kps = Array.isArray(it.knowledge_points) ? it.knowledge_points : [];
          const topics = Array.isArray(it.lecture_topics) ? it.lecture_topics : [];
          const kinds = Array.isArray(it.kinds) ? it.kinds : [];

          const item = document.createElement("div");
          item.className = "interval";
          item.innerHTML = `
            <div class="interval-head">
              <div class="interval-title">${escapeHtml(intervalTypeLabel(type))}</div>
              <div class="interval-time">${escapeHtml(formatSec(start))} → ${escapeHtml(formatSec(end))} · ${escapeHtml(formatDuration(dur))}</div>
            </div>
            <div class="interval-body">
              <div>
                <strong>判定：</strong>
                ${
                  kinds.length
                    ? `<div class="chips">${kinds
                        .slice(0, 6)
                        .map((k) => `<span class="chip">${escapeHtml(intervalTypeLabel(k))}</span>`)
                        .join("")}</div>`
                    : '<span class="muted">（无）</span>'
                }
              </div>
              <div><strong>当时讲解：</strong> ${asr ? escapeHtml(asr) : '<span class="muted">（无 ASR 文本）</span>'}</div>
              <div>
                <strong>课程主题：</strong>
                ${
                  topics.length
                    ? `<div class="chips">${topics.slice(0, 8).map((k) => `<span class="chip">${escapeHtml(k)}</span>`).join("")}</div>`
                    : '<span class="muted">（未生成/无匹配）</span>'
                }
              </div>
              <div>
                <strong>知识点：</strong>
                ${
                  kps.length
                    ? `<div class="chips">${kps.map((k) => `<span class="chip">${escapeHtml(k)}</span>`).join("")}</div>`
                    : '<span class="muted">（无知识点/未配置大模型）</span>'
                }
              </div>
            </div>
          `;
          list.appendChild(item);
        }
      }
      det.appendChild(list);
      wrap.appendChild(det);
    }

    reportBody.innerHTML = "";
    reportBody.appendChild(head);
    reportBody.appendChild(wrap);
  }

  function setReportStatus(html, kind = "muted") {
    reportBody.innerHTML = `<div class="${kind}">${html}</div>`;
  }

  async function startReportJob({ sessionId } = {}) {
    const sid = sessionId || state.sessionId;
    if (!sid) {
      setReportStatus("没有可用的 session_id（请先开始并停止一次录制）。", "warn-text");
      return;
    }

    state.report.lastSessionId = sid;
    openReportModal();
    
    // Show session info and try to get display name
    const sessions = state.sessions || [];
    const sessionInfo = sessions.find(s => s.session_id === sid);
    updateReportSessionInfo(sid, sessionInfo?.display_name || null);
    
    reportLinks.innerHTML = "";
    reportKpis.innerHTML = "";
    setReportStatus("正在提交统计任务…");

    const j = await fetchJson("/api/session/process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sid }),
    });
    if (!j.ok) throw new Error(j.error || "process start failed");
    state.report.jobId = j.job_id;
    state.report.polling = true;

    setReportStatus(`任务已提交：<span class="mono">${escapeHtml(j.job_id)}</span>，处理中…`);
    await pollReportJob(j.job_id);
  }

  async function pollReportJob(jobId) {
    const maxPoll = 240; // seconds
    for (let i = 0; i < maxPoll; i++) {
      await new Promise((r) => setTimeout(r, 1000));
      let st = null;
      try {
        st = await fetchJson(`/api/session/process/status?job_id=${encodeURIComponent(jobId)}`);
      } catch (e) {
        setReportStatus(`轮询失败：${escapeHtml(String(e))}`, "warn-text");
        continue;
      }
      const job = st?.job;
      const status = String(job?.status || "unknown");
      if (status === "done") {
        state.report.polling = false;
        state.report.lastResult = job.result || null;
        state.report.lastStatsUrl = job.result?.stats || null;
        btnOpenReport.style.display = state.report.lastStatsUrl ? "inline-flex" : "none";
        setReportStatus("统计完成，正在加载结果…");
        showToast("报告生成完成", "success");
        await loadReportFromResult(job.result);
        return;
      }
      if (status === "error") {
        state.report.polling = false;
        setReportStatus(`统计出错：${escapeHtml(String(job?.error || "unknown"))}`, "danger-text");
        showToast("报告生成失败", "error");
        return;
      }
      setReportStatus(`处理中：${escapeHtml(status)}（${i + 1}s）`);
    }
    state.report.polling = false;
    setReportStatus("统计超时：任务可能仍在后台运行，可稍后点击“重载”。", "warn-text");
  }

  async function loadReportFromResult(result) {
    renderReportLinks(result);
    if (!result?.stats) {
      setReportStatus("没有找到 stats.json 输出。", "warn-text");
      return;
    }
    state.report.lastStatsUrl = result.stats;
    btnOpenReport.style.display = "inline-flex";

    const stats = await fetchJson(result.stats);
    renderReportKpis(stats);
    renderReportBody(stats);
  }

  async function loadReportFromSessionId(sessionId) {
    const sid = String(sessionId || "").trim();
    if (!sid) return;
    openReportModal();
    reportLinks.innerHTML = "";
    reportKpis.innerHTML = "";

    // Show session info and try to get display name
    const sessions = state.sessions || [];
    const sessionInfo = sessions.find(s => s.session_id === sid);
    updateReportSessionInfo(sid, sessionInfo?.display_name || null);

    // try direct stats first
    const statsUrl = `/out/${encodeURIComponent(sid)}/stats.json`;
    try {
      const stats = await fetchJson(statsUrl);
      state.report.lastSessionId = sid;
      state.report.lastStatsUrl = statsUrl;
      btnOpenReport.style.display = "inline-flex";
      renderReportLinks({
        stats: statsUrl,
        video: stats.video ? `/out/${sid}/${stats.video}` : null,
        audio: stats.audio ? `/out/${sid}/${stats.audio}` : null,
        transcript: stats.transcript ? `/out/${sid}/${stats.transcript}` : null,
        lesson_summary: stats.lesson_summary ? `/out/${sid}/lesson_summary.json` : null,
      });
      renderReportKpis(stats);
      renderReportBody(stats);
      return;
    } catch (_) {
      // fall through
    }

    setReportStatus(`该会话尚未生成 <span class="mono">stats.json</span>，正在为其生成报告…`);
    await startReportJob({ sessionId: sid });
  }

  async function refreshSessions() {
    sessionSelect.disabled = true;
    btnRefreshSessions.disabled = true;
    try {
      const j = await fetchJson(`/api/sessions?_t=${Date.now()}`);
      console.log("[refreshSessions] API response:", j);
      const items = Array.isArray(j?.sessions) ? j.sessions : [];
      
      // Store sessions in state for later lookup
      state.sessions = items;
      
      sessionSelect.innerHTML = "";
      const opt0 = document.createElement("option");
      opt0.value = "";
      opt0.textContent = "选择历史会话…";
      sessionSelect.appendChild(opt0);

      for (const it of items) {
        const sid = String(it.session_id || it.id || "");
        if (!sid) continue;
        const hasStats = Boolean(it.has_stats);
        const displayName = it.display_name || null;
        console.log(`[refreshSessions] Session ${sid}: display_name = ${displayName}`);
        const opt = document.createElement("option");
        opt.value = sid;
        // Show display_name if available, otherwise just session_id
        const label = displayName ? `${displayName} (${sid})` : sid;
        opt.textContent = `${label}${hasStats ? " · stats✅" : ""}`;
        sessionSelect.appendChild(opt);
      }
      sessionSelect.disabled = false;
    } catch (e) {
      sessionSelect.innerHTML = `<option value="">（无法加载：${String(e)}）</option>`;
      sessionSelect.disabled = true;
    } finally {
      btnRefreshSessions.disabled = false;
    }
  }

  function applyStudentFilter() {
    const q = String(studentFilter.value || "").trim().toLowerCase();
    const cards = studentList.querySelectorAll(".student-card");
    for (const card of cards) {
      const sid = String(card.dataset.sid || "").toLowerCase();
      card.style.display = !q || sid.includes(q) ? "" : "none";
    }
  }

  function bindUi() {
    // Global keyboard shortcuts
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        if (settingsModal && settingsModal.classList.contains("open")) {
          closeSettingsModal();
          e.preventDefault();
        } else if (modelModal.classList.contains("open")) {
          closeModelModal();
          e.preventDefault();
        } else if (reportModal.classList.contains("open")) {
          closeReportModal();
          e.preventDefault();
        }
      }
      // 'T' to toggle timeline (only when no modal is open and not in input)
      if ((e.key === "t" || e.key === "T") && !e.ctrlKey && !e.metaKey && !e.altKey) {
        const isModalOpen = (settingsModal && settingsModal.classList.contains("open")) ||
                            modelModal.classList.contains("open") ||
                            reportModal.classList.contains("open");
        const isInputFocused = document.activeElement?.tagName === "INPUT" ||
                               document.activeElement?.tagName === "TEXTAREA";
        if (!isModalOpen && !isInputFocused) {
          toggleTimeline.checked = !toggleTimeline.checked;
          toggleTimeline.dispatchEvent(new Event("change"));
          e.preventDefault();
        }
      }
    });

    // Mobile: scroll indicator for model nav
    if (modelNav) {
      const checkScrollEnd = () => {
        const isAtEnd = modelNav.scrollLeft + modelNav.clientWidth >= modelNav.scrollWidth - 5;
        modelNav.classList.toggle("scroll-end", isAtEnd);
      };
      modelNav.addEventListener("scroll", checkScrollEnd);
      window.addEventListener("resize", checkScrollEnd);
      setTimeout(checkScrollEnd, 100);  // Initial check after render
    }

    // Settings Center event listeners
    if (btnSettingsCenter) {
      btnSettingsCenter.addEventListener("click", async () => {
        // Open modal immediately for responsiveness
        openSettingsModal();
        // Load config in background
        try {
          await loadWebdavConfig();
        } catch (e) {
          console.warn("Failed to load WebDAV config:", e);
          // Don't show error toast - the modal is already open and user can still use it
        }
      });
    }
    if (settingsModal) {
      settingsModal.addEventListener("click", (e) => {
        if (e.target === settingsModal) closeSettingsModal();
      });
    }
    if (settingsDragHandle) {
      settingsDragHandle.addEventListener("pointerdown", startSettingsDrag);
      settingsDragHandle.addEventListener("pointermove", moveSettingsDrag);
      settingsDragHandle.addEventListener("pointerup", endSettingsDrag);
      settingsDragHandle.addEventListener("pointercancel", endSettingsDrag);
    }
    if (btnSettingsSave) {
      btnSettingsSave.addEventListener("click", saveWebdavConfig);
    }
    if (btnSettingsClose) {
      btnSettingsClose.addEventListener("click", closeSettingsModal);
    }
    if (btnWebdavTest) {
      btnWebdavTest.addEventListener("click", testWebdavConnection);
    }
    if (btnWebdavReset) {
      btnWebdavReset.addEventListener("click", resetWebdavToDefaults);
    }
    if (btnToggleWebdavPassword) {
      btnToggleWebdavPassword.addEventListener("click", toggleWebdavPasswordVisibility);
    }
    if (webdavUrl) {
      webdavUrl.addEventListener("blur", () => {
        const validation = validateWebdavUrl(webdavUrl.value);
        if (!validation.valid) {
          showToast(validation.message, "warning");
        }
      });
    }
    if (webdavEnabled) {
      webdavEnabled.addEventListener("change", () => {
        syncWebdavStateFromForm();
        updateWebdavFormDisabledState();
      });
    }
    for (const btn of settingsNavBtns) {
      btn.addEventListener("click", () => {
        setActiveSettingsTab(btn.dataset.tab);
      });
    }

    btnModelCenter.addEventListener("click", async () => {
      try {
        if (!state.models.config) await loadModelsConfig();
        openModelModal();
      } catch (e) {
        alert(`模型设置不可用：${String(e)}`);
      }
    });
    modelModal.addEventListener("click", (e) => {
      if (e.target === modelModal) closeModelModal();
    });
    reportModal.addEventListener("click", (e) => {
      if (e.target === reportModal) closeReportModal();
    });
    if (modelDragHandle) {
      modelDragHandle.addEventListener("pointerdown", startModelDrag);
      modelDragHandle.addEventListener("pointermove", moveModelDrag);
      modelDragHandle.addEventListener("pointerup", endModelDrag);
      modelDragHandle.addEventListener("pointercancel", endModelDrag);
    }
    if (btnModelsResetPos) {
      btnModelsResetPos.addEventListener("click", resetModelDrawerPos);
    }
    for (const btn of modelNavButtons) {
      btn.addEventListener("click", () => {
        setModelSettingsView(btn.dataset.target || "__all__");
      });
    }
    btnModelsClose.addEventListener("click", closeModelModal);
    btnModelsSave.addEventListener("click", async () => {
      try {
        syncModelStateFromForm();
        await saveModelsConfig();
        modelCheckBox.innerHTML = `<span class="ok-text">已保存并应用。</span>`;
        updateModelPill();
      } catch (e) {
        alert(`保存失败：${String(e)}`);
      }
    });
    btnModelsCheck.addEventListener("click", async () => {
      setButtonLoading(btnModelsCheck, true);
      try {
        syncModelStateFromForm();
        if (state.models.dirty) await saveModelsConfig();
        await checkModels({ deep: false });
        showToast("模型检测完成", "success");
      } catch (e) {
        modelCheckBox.innerHTML = `<span class="danger-text">检测失败：${escapeHtml(String(e))}</span>`;
        showToast(`检测失败：${e}`, "error");
      } finally {
        setButtonLoading(btnModelsCheck, false);
      }
    });
    btnModelsCheckDeep.addEventListener("click", async () => {
      setButtonLoading(btnModelsCheckDeep, true);
      try {
        syncModelStateFromForm();
        if (state.models.dirty) await saveModelsConfig();
        await checkModels({ deep: true });
        showToast("深度检测完成", "success");
      } catch (e) {
        modelCheckBox.innerHTML = `<span class="danger-text">深度检测失败：${escapeHtml(String(e))}</span>`;
        showToast(`深度检测失败：${e}`, "error");
      } finally {
        setButtonLoading(btnModelsCheckDeep, false);
      }
    });

    btnLlmPullModels.addEventListener("click", async () => {
      setButtonLoading(btnLlmPullModels, true);
      try {
        await pullLlmModels();
        showToast("模型列表拉取成功", "success");
      } catch (e) {
        modelCheckBox.innerHTML = `<span class="danger-text">拉取模型失败：${escapeHtml(String(e))}</span>`;
        showToast(`拉取模型失败：${e}`, "error");
      } finally {
        setButtonLoading(btnLlmPullModels, false);
      }
    });

    modelModeSel.addEventListener("change", syncModelStateFromForm);
    asrProviderSel.addEventListener("change", syncModelStateFromForm);
    asrModelInput.addEventListener("input", syncModelStateFromForm);
    llmEnabledToggle.addEventListener("change", () => {
      syncModelStateFromForm();
      updateLlmSectionDisabledState();
    });
    llmBaseUrlInput.addEventListener("input", () => {
      state.models.llmModels = [];
      state.models.llmModelsError = null;
      renderLlmModelsSelect();
      syncModelStateFromForm();
    });
    llmApiKeyInput.addEventListener("input", syncModelStateFromForm);
    llmModelSel.addEventListener("change", () => {
      const v = String(llmModelSel.value || "").trim();
      if (v) llmModelInput.value = v;
      syncModelStateFromForm();
    });
    llmModelInput.addEventListener("input", syncModelStateFromForm);

    // ASR model selection events
    if (asrModelSel) {
      asrModelSel.addEventListener("change", () => {
        const v = String(asrModelSel.value || "").trim();
        if (v) asrModelInput.value = v;
        syncModelStateFromForm();
      });
    }
    if (asrUseIndependentToggle) {
      asrUseIndependentToggle.addEventListener("change", () => {
        syncModelStateFromForm();
        updateModelFormDisabledState();
      });
    }
    if (asrBaseUrlInput) {
      asrBaseUrlInput.addEventListener("input", syncModelStateFromForm);
    }
    if (asrApiKeyInput) {
      asrApiKeyInput.addEventListener("input", syncModelStateFromForm);
    }

    windowSel.addEventListener("change", () => {
      state.windowSec = toNum(windowSel.value, 60);
      scheduleRender({ timeline: true });
    });

    studentFilter.addEventListener("input", applyStudentFilter);

    sessionSelect.addEventListener("change", async () => {
      const sid = String(sessionSelect.value || "").trim();
      if (!sid) return;
      await loadReportFromSessionId(sid);
    });
    btnRefreshSessions.addEventListener("click", refreshSessions);

    btnStart.addEventListener("click", async () => {
      try {
        await startRecording();
      } catch (e) {
        alert(`开始录制失败：${e}`);
        setRecording(false, state.sessionId);
      }
    });
    btnStop.addEventListener("click", async () => {
      try {
        await stopRecording();
      } catch (e) {
        alert(`停止失败：${e}`);
        setRecording(false, state.sessionId);
      }
    });

    btnReport.addEventListener("click", async () => {
      setButtonLoading(btnReport, true);
      try {
        await startReportJob({ sessionId: state.sessionId });
      } catch (e) {
        setReportStatus(`生成报告失败：${escapeHtml(String(e))}`, "danger-text");
        showToast(`生成报告失败：${e}`, "error");
      } finally {
        setButtonLoading(btnReport, false);
        btnReport.disabled = state.isRecording || !state.sessionId;
      }
    });
    btnOpenReport.addEventListener("click", openReportModal);
    btnCloseReport.addEventListener("click", closeReportModal);
    if (btnUploadWebdav) {
      btnUploadWebdav.addEventListener("click", () => {
        uploadSessionToWebdav(state.report.lastSessionId);
      });
    }

    // Report drawer drag & maximize
    if (reportDragHandle) {
      reportDragHandle.addEventListener("pointerdown", startReportDrag);
      reportDragHandle.addEventListener("pointermove", moveReportDrag);
      reportDragHandle.addEventListener("pointerup", endReportDrag);
      reportDragHandle.addEventListener("pointercancel", endReportDrag);
    }
    if (btnMaximizeReport) {
      btnMaximizeReport.addEventListener("click", toggleReportMaximize);
    }
    if (btnResetReportPos) {
      btnResetReportPos.addEventListener("click", resetReportDrawerPos);
    }
    if (btnRenameSession) {
      btnRenameSession.addEventListener("click", async () => {
        const sid = state.report.lastSessionId;
        const newName = reportSessionName?.value?.trim();
        if (!sid) {
          showToast("没有当前会话", "warning");
          return;
        }
        if (!newName) {
          showToast("请输入会话名称", "warning");
          return;
        }
        setButtonLoading(btnRenameSession, true);
        await renameSession(sid, newName);
        setButtonLoading(btnRenameSession, false);
      });
    }
    // Allow pressing Enter to save rename
    if (reportSessionName) {
      reportSessionName.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          btnRenameSession?.click();
        }
      });
    }

    btnReloadReport.addEventListener("click", async () => {
      try {
        if (state.report.lastStatsUrl) {
          const stats = await fetchJson(state.report.lastStatsUrl);
          renderReportKpis(stats);
          renderReportBody(stats);
        } else if (state.report.lastSessionId) {
          await loadReportFromSessionId(state.report.lastSessionId);
        } else {
          setReportStatus("没有可重载的报告。", "warn-text");
        }
      } catch (e) {
        setReportStatus(`重载失败：${escapeHtml(String(e))}`, "danger-text");
      }
    });

    toggleBoxes.addEventListener("change", () => {
      state.showBoxes = toggleBoxes.checked;
      scheduleRender({ preview: true });
    });
    toggleLabels.addEventListener("change", () => {
      state.showLabels = toggleLabels.checked;
      scheduleRender({ preview: true });
    });
    toggleAsr.addEventListener("change", () => {
      state.showAsr = toggleAsr.checked;
      updateAsrNowLine();
      scheduleRender({ timeline: true });
    });

    toggleTimeline.addEventListener("change", () => {
      const show = toggleTimeline.checked;
      timelinePanel.classList.toggle("hidden", !show);
      mainPanel.classList.toggle("timeline-hidden", !show);
      localStorage.setItem("showTimeline", show ? "1" : "0");
      if (show) {
        scheduleRender({ timeline: true });
      }
    });

    btnFollow.addEventListener("click", () => {
      state.followLive = true;
      state.cursorTime = null;
      scheduleRender({ timeline: true });
    });
    btnClearCursor.addEventListener("click", () => {
      state.cursorTime = null;
      state.followLive = true;
      scheduleRender({ timeline: true });
    });

    timelineCanvas.addEventListener("click", (e) => {
      const rect = timelineCanvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const W = rect.width;
      if (W <= 0) return;

      const endT = state.followLive ? state.nowSec : state.cursorTime ?? state.nowSec;
      const startT = endT - state.windowSec;
      const t = startT + (x / W) * state.windowSec;

      state.cursorTime = clamp(t, startT, endT);
      state.followLive = false;
      scheduleRender({ timeline: true });
    });

    window.addEventListener("resize", () => {
      scheduleRender({ preview: true, timeline: true });
      ensureModelDrawerInViewport();
    });
  }

  async function loadInitialStatus() {
    try {
      const j = await fetchJson("/api/session/status");
      const sid = j?.session_id || null;
      setRecording(Boolean(j?.is_recording), sid);
      if (j?.model_config && typeof j.model_config === "object") {
        state.models.config = j.model_config;
        state.models.dirty = false;
        applyModelFormFromState();
        updateModelPill();
      }
    } catch (_) {
      setRecording(false, null);
    }
  }

  async function init() {
    state.windowSec = toNum(windowSel.value, 60);
    state.showBoxes = toggleBoxes.checked;
    state.showLabels = toggleLabels.checked;
    state.showAsr = toggleAsr.checked;

    // Initialize timeline visibility (default: hidden)
    const showTimeline = localStorage.getItem("showTimeline") === "1";
    toggleTimeline.checked = showTimeline;
    timelinePanel.classList.toggle("hidden", !showTimeline);
    mainPanel.classList.toggle("timeline-hidden", !showTimeline);

    bindUi();
    connectWs();
    await loadInitialStatus();
    await refreshSessions();

    // Load configs in parallel for faster initialization
    await Promise.allSettled([
      loadWebdavConfig().catch(e => console.warn("Failed to load WebDAV config:", e)),
      loadModelsConfig().then(() => checkModels({ deep: false })).catch(e => {
        console.warn(e);
        updateModelPill();
      })
    ]);

    scheduleRender({ preview: true, timeline: true });
  }

  init().catch((e) => {
    console.error(e);
  });
})();
