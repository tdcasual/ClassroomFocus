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
  const langSelect = document.getElementById("langSelect");
  const classLangSelect = document.getElementById("classLangSelect");
  const btnThemeToggle = document.getElementById("btnThemeToggle");
  const themeIcon = document.getElementById("themeIcon");
  const themeText = document.getElementById("themeText");

  const I18N_LANG_KEY = "ui_lang";
  const CLASS_LANG_KEY = "class_lang";
  const THEME_KEY = "ui_theme";
  const SUPPORTED_LANGS = ["zh", "en"];
  const DEFAULT_LANG = "zh";
  const DEFAULT_CLASS_LANG = "zh";
  let i18n = { lang: DEFAULT_LANG, dict: {} };

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
  const CLIENT_THUMB_REFRESH_MS = 60000;
  const CLIENT_THUMB_SIZE = 52;
  const CLIENT_THUMB_QUALITY = 0.72;
  const CLIENT_THUMB_PAD = 0.18;

  function statusLabel(state) {
    return t(STATUS_LABELS[state] || STATUS_LABELS.unknown);
  }

  function detectLanguage() {
    const stored = localStorage.getItem(I18N_LANG_KEY);
    if (SUPPORTED_LANGS.includes(stored)) return stored;
    return DEFAULT_LANG;
  }

  async function loadI18n(lang) {
    const safeLang = SUPPORTED_LANGS.includes(lang) ? lang : DEFAULT_LANG;
    try {
      const res = await fetch(`./i18n/${safeLang}.json?_t=${Date.now()}`);
      if (!res.ok) throw new Error(`i18n load failed: ${res.status}`);
      const data = await res.json();
      i18n = { lang: safeLang, dict: data || {} };
    } catch (_) {
      i18n = { lang: safeLang, dict: {} };
    }
    document.documentElement.lang = safeLang === "en" ? "en" : "zh-CN";
  }

  function t(key, params = null) {
    let str = (i18n.dict && Object.prototype.hasOwnProperty.call(i18n.dict, key)) ? i18n.dict[key] : key;
    if (params && typeof params === "object") {
      str = String(str).replace(/\{(\w+)\}/g, (m, k) => {
        if (params[k] == null) return m;
        return String(params[k]);
      });
    }
    return String(str);
  }

  function applyI18n(root = document) {
    const textNodes = root.querySelectorAll("[data-i18n]");
    for (const el of textNodes) {
      el.textContent = t(el.dataset.i18n || "");
    }
    const titles = root.querySelectorAll("[data-i18n-title]");
    for (const el of titles) {
      el.title = t(el.dataset.i18nTitle || "");
    }
    const placeholders = root.querySelectorAll("[data-i18n-placeholder]");
    for (const el of placeholders) {
      el.placeholder = t(el.dataset.i18nPlaceholder || "");
    }
    const ariaLabels = root.querySelectorAll("[data-i18n-aria-label]");
    for (const el of ariaLabels) {
      el.setAttribute("aria-label", t(el.dataset.i18nAriaLabel || ""));
    }
  }

  async function setLanguage(lang) {
    const safeLang = SUPPORTED_LANGS.includes(lang) ? lang : DEFAULT_LANG;
    localStorage.setItem(I18N_LANG_KEY, safeLang);
    await loadI18n(safeLang);
    applyI18n();
    refreshUiTexts();
  }

  function detectClassLanguage() {
    const stored = localStorage.getItem(CLASS_LANG_KEY);
    if (SUPPORTED_LANGS.includes(stored)) return stored;
    return detectLanguage() || DEFAULT_CLASS_LANG;
  }

  function setClassLanguage(lang) {
    const safeLang = SUPPORTED_LANGS.includes(lang) ? lang : DEFAULT_CLASS_LANG;
    localStorage.setItem(CLASS_LANG_KEY, safeLang);
    state.classLang = safeLang;
  }

  function getSystemTheme() {
    if (window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches) {
      return "light";
    }
    return "dark";
  }

  function getStoredTheme() {
    const v = localStorage.getItem(THEME_KEY);
    if (v === "light" || v === "dark") return v;
    return null;
  }

  function updateThemeToggle() {
    if (!btnThemeToggle || !themeText || !themeIcon) return;
    const resolved = getStoredTheme() || getSystemTheme();
    const isDark = resolved === "dark";
    themeIcon.textContent = isDark ? "☾" : "☀";
    themeText.textContent = isDark ? t("夜间") : t("白天");
    btnThemeToggle.setAttribute("aria-pressed", isDark ? "true" : "false");
  }

  function applyTheme(theme) {
    if (theme === "light") {
      document.documentElement.setAttribute("data-theme", "light");
    } else if (theme === "dark") {
      document.documentElement.setAttribute("data-theme", "dark");
    } else {
      document.documentElement.removeAttribute("data-theme");
    }
    updateThemeToggle();
  }

  function toggleTheme() {
    const current = getStoredTheme() || getSystemTheme();
    const next = current === "dark" ? "light" : "dark";
    localStorage.setItem(THEME_KEY, next);
    applyTheme(next);
  }

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
    reportResult: null,
    thumbs: new Map(),
    modelConfig: null,
    classLang: DEFAULT_CLASS_LANG,
  };

  const thumbCanvas = document.createElement("canvas");
  thumbCanvas.width = CLIENT_THUMB_SIZE;
  thumbCanvas.height = CLIENT_THUMB_SIZE;
  const thumbCtx = thumbCanvas.getContext("2d");
  const thumbImage = new Image();
  thumbImage.decoding = "async";
  let pendingThumbFaces = [];

  thumbImage.onload = () => {
    const targets = pendingThumbFaces;
    pendingThumbFaces = [];
    renderThumbsFromImage(thumbImage, targets);
  };
  thumbImage.onerror = () => {
    pendingThumbFaces = [];
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
    wsText.textContent = connected ? t("connected") : t("disconnected");
    wsDot.className = `dot ${connected ? "good" : ""}`.trim();
  }

  function setRecording(isRecording, sessionId = null) {
    state.isRecording = Boolean(isRecording);
    if (sessionId != null) state.sessionId = sessionId;

    recText.textContent = state.isRecording ? t("recording") : t("idle");
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

  async function loadModelConfig() {
    try {
      const j = await fetchJson("/api/models/config", undefined, 4000);
      if (j && j.ok) {
        state.modelConfig = j.config || null;
        const cfgLang = String(j?.config?.llm?.language || "").toLowerCase();
        if (cfgLang === "zh" || cfgLang === "en") {
          state.classLang = cfgLang;
          if (classLangSelect) classLangSelect.value = cfgLang;
          setClassLanguage(cfgLang);
        }
      }
    } catch (_) {
      state.modelConfig = null;
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
      showError(ev.error || t("Unknown error"));
      return;
    }
  }

  function handleLiteStatus(data) {
    const faces = Array.isArray(data.faces) ? data.faces : [];
    applyFaces(faces, true);
    applyThumbs(data?.thumbs);
  }

  function handleLiteSnapshot(data) {
    const faces = Array.isArray(data.faces) ? data.faces : [];
    applyFaces(faces, true);
    applyThumbs(data?.thumbs);
  }

  function handleLiteDelta(data) {
    const changed = data?.changed && typeof data.changed === "object" ? data.changed : {};
    const removed = Array.isArray(data?.removed) ? data.removed : [];
    applyDelta(changed, removed);
    applyThumbs(data?.thumbs);
  }

  function handleFrameData(data) {
    const now = Date.now();
    if (now - state.lastFrameAt < FRAME_THROTTLE_MS) return;
    state.lastFrameAt = now;
    const faces = Array.isArray(data.faces) ? data.faces : [];
    applyFaces(faces);
    if (data && data.thumbs) {
      applyThumbs(data.thumbs);
    }
    if (data && data.image_base64) {
      updateThumbsFromFrame(String(data.image_base64 || ""), faces);
    }
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

  function shouldRefreshThumb(card, nowMs) {
    if (!card) return false;
    if (!card.thumbSet) return true;
    const last = card.thumbUpdatedAt || 0;
    return (nowMs - last) >= CLIENT_THUMB_REFRESH_MS;
  }

  function updateThumbsFromFrame(b64, faces) {
    if (!b64 || !faces || !faces.length) return;
    const nowMs = Date.now();
    const targets = [];
    for (const f of faces) {
      if (!f || typeof f !== "object") continue;
      const sid = String(f.track_id ?? f.student_id ?? "");
      if (!sid) continue;
      const bbox = f.bbox;
      if (!Array.isArray(bbox) || bbox.length < 4) continue;
      const card = state.cards.get(sid);
      if (!card || !shouldRefreshThumb(card, nowMs)) continue;
      targets.push({ sid, bbox, card });
    }
    if (!targets.length) return;
    pendingThumbFaces = targets;
    const src = `data:image/jpeg;base64,${b64}`;
    if (thumbImage.src !== src) {
      thumbImage.src = src;
    } else if (thumbImage.complete) {
      renderThumbsFromImage(thumbImage, pendingThumbFaces);
    }
  }

  function renderThumbsFromImage(img, targets) {
    if (!img || !targets || !targets.length || !thumbCtx) return;
    const iw = img.naturalWidth || img.width || 1;
    const ih = img.naturalHeight || img.height || 1;
    const nowMs = Date.now();
    for (const t of targets) {
      const bbox = t.bbox;
      if (!bbox) continue;
      const nx = Number(bbox[0]) || 0;
      const ny = Number(bbox[1]) || 0;
      const nw = Number(bbox[2]) || 0;
      const nh = Number(bbox[3]) || 0;
      if (nw <= 0 || nh <= 0) continue;

      const pad = CLIENT_THUMB_PAD;
      const cx = nx + nw / 2;
      const cy = ny + nh / 2;
      const size = Math.max(nw, nh) * (1 + pad * 2);

      const sx0 = Math.max(0, cx - size / 2);
      const sy0 = Math.max(0, cy - size / 2);
      const sx1 = Math.min(1, cx + size / 2);
      const sy1 = Math.min(1, cy + size / 2);

      const sx = sx0 * iw;
      const sy = sy0 * ih;
      const sw = Math.max(1, (sx1 - sx0) * iw);
      const sh = Math.max(1, (sy1 - sy0) * ih);

      thumbCtx.clearRect(0, 0, CLIENT_THUMB_SIZE, CLIENT_THUMB_SIZE);
      try {
        thumbCtx.drawImage(img, sx, sy, sw, sh, 0, 0, CLIENT_THUMB_SIZE, CLIENT_THUMB_SIZE);
      } catch (_) {
        continue;
      }
      const dataUrl = thumbCanvas.toDataURL("image/jpeg", CLIENT_THUMB_QUALITY);
      const b64 = dataUrl.split(",")[1] || "";
      if (!b64) continue;
      state.thumbs.set(t.sid, b64);
      setCardThumb(t.card, b64, nowMs);
    }
  }

  function storeThumbs(thumbs) {
    if (!thumbs || typeof thumbs !== "object") return false;
    let changed = false;
    for (const [sid, b64] of Object.entries(thumbs)) {
      if (!sid || !b64) continue;
      const prev = state.thumbs.get(sid);
      if (prev !== b64) {
        state.thumbs.set(sid, b64);
        changed = true;
      }
    }
    return changed;
  }

  function applyThumbs(thumbs) {
    if (!thumbs || typeof thumbs !== "object") return;
    storeThumbs(thumbs);
    for (const [sid, b64] of Object.entries(thumbs)) {
      const card = state.cards.get(String(sid));
      if (!card) continue;
      setCardThumb(card, b64);
    }
  }

  function setCardThumb(card, b64, updatedAt = null) {
    if (!card || !card.thumbEl || !b64) return;
    if (card.thumbB64 === b64) {
      card.thumbSet = true;
      card.thumbUpdatedAt = updatedAt || Date.now();
      return;
    }
    card.thumbB64 = b64;
    card.thumbEl.style.backgroundImage = `url(data:image/jpeg;base64,${b64})`;
    card.thumbEl.classList.add("has-thumb");
    card.thumbSet = true;
    card.thumbUpdatedAt = updatedAt || Date.now();
  }

  function createCard(id) {
    const el = document.createElement("div");
    el.className = "face-card";
    el.setAttribute("role", "listitem");

    const head = document.createElement("div");
    head.className = "face-head";

    const thumb = document.createElement("div");
    thumb.className = "face-thumb";
    thumb.setAttribute("aria-hidden", "true");

    const main = document.createElement("div");
    main.className = "face-main";

    const row = document.createElement("div");
    row.className = "face-row";

    const label = document.createElement("div");
    label.className = "face-id";
    label.textContent = t("ID {id}", { id });

    const status = document.createElement("div");
    status.className = "state-pill state-unknown";
    status.textContent = t("Unknown");

    row.appendChild(label);
    row.appendChild(status);

    const meta = document.createElement("div");
    meta.className = "face-meta";
    meta.textContent = t("Last change: {age}", { age: t("just now") });

    main.appendChild(row);
    main.appendChild(meta);

    head.appendChild(thumb);
    head.appendChild(main);

    el.appendChild(head);

    const card = { id, el, idEl: label, statusEl: status, metaEl: meta, thumbEl: thumb, thumbSet: false, thumbB64: null, thumbUpdatedAt: 0, state: "unknown", updatedAt: 0 };
    const stored = state.thumbs.get(String(id));
    if (stored) setCardThumb(card, stored);
    return card;
  }

  function updateCard(card) {
    const st = card.state;
    if (card.idEl) {
      card.idEl.textContent = t("ID {id}", { id: card.id });
    }
    const label = statusLabel(st);
    const cls = STATUS_CLASS[st] || STATUS_CLASS.unknown;
    card.statusEl.textContent = label;
    card.statusEl.className = `state-pill ${cls}`;
    card.metaEl.textContent = t("Last change: {age}", { age: formatAge(card.updatedAt) });
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
      lastUpdate.textContent = t("No updates yet");
      return;
    }
    lastUpdate.textContent = t("Updated {age}", { age: formatAge(state.lastUpdateAt) });
  }

  function formatAge(ts) {
    if (!ts) return "-";
    const diff = Math.max(0, Math.floor((Date.now() - ts) / 1000));
    if (diff < 2) return t("just now");
    if (diff < 60) return t("{sec}s ago", { sec: diff });
    const min = Math.floor(diff / 60);
    const sec = diff % 60;
    return t("{min}m {sec}s ago", { min, sec });
  }

  function refreshUiTexts() {
    setWsStatus(state.wsConnected);
    setRecording(state.isRecording, state.sessionId);
    updateLastUpdateText();
    updateThemeToggle();
    for (const card of state.cards.values()) {
      updateCard(card);
    }
    if (state.reportResult) {
      renderReportLinks(state.reportResult);
    }
    if (reportMeta && !state.reportPolling && !state.reportJobId) {
      reportMeta.textContent = t("Idle");
    }
    if (reportStatus && !state.reportPolling && reportLinks && reportLinks.children.length === 0) {
      reportStatus.textContent = t("No report running.");
    }
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
      showError(t("Status error: {error}", { error: String(e) }));
    }
  }

  function buildModelConfigForStart() {
    const cfg = state.modelConfig && typeof state.modelConfig === "object"
      ? JSON.parse(JSON.stringify(state.modelConfig))
      : {};
    if (!cfg.llm || typeof cfg.llm !== "object") cfg.llm = {};
    if (!cfg.asr || typeof cfg.asr !== "object") cfg.asr = {};
    if (state.classLang) {
      cfg.llm.language = state.classLang;
      cfg.asr.language = state.classLang;
    }
    return cfg;
  }

  async function startRecording() {
    clearError();
    try {
      if (!state.modelConfig) {
        await loadModelConfig();
      }
      const j = await fetchJson("/api/session/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_config: buildModelConfigForStart() }),
      });
      if (!j.ok) throw new Error(j.error || t("start failed"));
      setRecording(true, j.session_id || null);
    } catch (e) {
      showError(t("Start failed: {error}", { error: String(e) }));
      setRecording(false, state.sessionId);
    }
  }

  async function stopRecording() {
    clearError();
    try {
      const j = await fetchJson("/api/session/stop", { method: "POST" });
      if (!j.ok) throw new Error(j.error || t("stop failed"));
      setRecording(false, j.session_id || state.sessionId);
    } catch (e) {
      showError(t("Stop failed: {error}", { error: String(e) }));
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
    state.thumbs = new Map();
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
    state.reportResult = result;
    const links = [
      { label: t("Stats"), href: result.stats },
      { label: t("Transcript"), href: result.transcript },
      { label: t("Lesson Summary"), href: result.lesson_summary },
      { label: t("Video"), href: result.video },
      { label: t("Audio"), href: result.audio },
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
      showError(t("No session_id available. Start and stop a session first."));
      return;
    }
    if (state.reportPolling) return;
    state.reportAuto = Boolean(isAuto);
    state.reportSessionId = sid;
    state.reportPolling = true;
    btnReport.disabled = true;
    setReportStatus(t("Submitting job..."), t("Running"));
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
      if (!j.ok) throw new Error(j.error || t("process start failed"));
      state.reportJobId = j.job_id;
      await pollReportJob(j.job_id);
    } catch (e) {
      setReportStatus(t("Report failed: {error}", { error: String(e) }), t("Error"));
      setReportVisible(true);
      showError(t("Report failed: {error}", { error: String(e) }));
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
        setReportStatus(t("Polling error: {error}", { error: String(e) }), t("Error"));
        continue;
      }
      const job = st?.job;
      const status = String(job?.status || "unknown");
      if (status === "done") {
        state.reportPolling = false;
        setReportStatus(t("Report ready"), t("Done"));
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
        setReportStatus(
          t("Report error: {error}", { error: String(job?.error || t("unknown")) }),
          t("Error"),
        );
        setReportVisible(true);
        btnReport.disabled = state.isRecording || !state.sessionId;
        return;
      }
      setReportStatus(t("Processing: {status}", { status }), t("Running"));
    }
    state.reportPolling = false;
    setReportStatus(t("Report timeout. Try again later."), t("Timeout"));
    setReportVisible(true);
    btnReport.disabled = state.isRecording || !state.sessionId;
  }

  async function uploadToWebdav(sessionId) {
    if (!sessionId) return;
    try {
      setReportStatus(t("Uploading report..."), t("Running"));
      const j = await fetchJson("/api/webdav/upload", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      }, 15000);
      if (!j.ok) throw new Error(j.error || t("upload failed"));
      setReportStatus(t("Report ready (uploaded)"), t("Done"));
    } catch (e) {
      setReportStatus(t("Upload failed: {error}", { error: String(e) }), t("Error"));
      setReportVisible(true);
      showError(t("Upload failed: {error}", { error: String(e) }));
    }
  }

  function bindUi() {
    btnStart.addEventListener("click", startRecording);
    btnStop.addEventListener("click", stopRecording);
    btnReport.addEventListener("click", () => startReport(false));
    if (langSelect) {
      langSelect.addEventListener("change", () => {
        setLanguage(langSelect.value);
      });
    }
    if (classLangSelect) {
      classLangSelect.addEventListener("change", () => {
        setClassLanguage(classLangSelect.value);
      });
    }
    if (btnThemeToggle) {
      btnThemeToggle.addEventListener("click", toggleTheme);
    }
  }

  async function init() {
    await loadI18n(detectLanguage());
    applyI18n();
    if (langSelect) langSelect.value = i18n.lang;
    state.classLang = detectClassLanguage();
    setClassLanguage(state.classLang);
    if (classLangSelect) classLangSelect.value = state.classLang;
    applyTheme(getStoredTheme() || getSystemTheme());
    await loadUiConfig();
    await loadModelConfig();
    bindUi();
    connectWs();
    await refreshSessionStatus();
    refreshUiTexts();
    setInterval(refreshSessionStatus, config.statusPollMs);
    setInterval(() => {
      if (document.hidden) return;
      updateLastUpdateText();
      for (const card of state.cards.values()) {
        card.metaEl.textContent = t("Last change: {age}", { age: formatAge(card.updatedAt) });
      }
    }, STATE_TICK_MS);
  }

  init();
})();
