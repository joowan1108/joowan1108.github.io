// preview-viewer.js — Canvas-based HWP/HWPX viewer for the Claude Code
// preview pane. Vanilla-JS canvas renderer driven by rhwp WASM.
//
// Key points to preserve when editing:
//   - Set globalThis.measureTextWidth via a hidden <canvas> 2d context
//     BEFORE initialising rhwp WASM. rhwp calls it during text layout.
//   - Cap canvas pixel area at MAX_CANVAS_PIXELS so very large pages don't
//     blow GPU memory.
//   - In the render loop, run the full geometry sweep first
//     (getPageInfo for every page), THEN the render loop. Doing them
//     interleaved on the same doc handle leaks rhwp's internal borrow.
//   - Each page's <canvas> is position:absolute inside a position:relative
//     wrap sized in NATIVE page pixels. fit() updates only CSS width/height
//     to scale the canvas pixel buffer down to the container — no
//     re-rasterisation on resize.
//   - Call doc.getPageTextLayout(i) AFTER each renderPageToCanvas. It's
//     the side-effect that releases rhwp's per-page borrow; without it the
//     next render panics "attempted to take ownership of Rust value while
//     it was borrowed". We discard the payload — v0.10 doesn't surface a
//     selectable text layer.

// ── 1. measureTextWidth (must register BEFORE rhwp init) ──────────────────
{
  let ctx = null;
  let lastFont = "";
  globalThis.measureTextWidth = (font, text) => {
    if (!ctx) ctx = document.createElement("canvas").getContext("2d");
    if (font !== lastFont) { ctx.font = font; lastFont = font; }
    return ctx.measureText(text).width;
  };
}

// ── 2. rhwp WASM ──────────────────────────────────────────────────────────
// Resolve vendor paths relative to this module so the same bundle works
// under both the local preview-server (root host, served from /) and a
// GitHub Pages sub-path (e.g. /claw-hwp/). Bare `/vendor/...` would break
// under any sub-path host.
const rhwpJsUrl = new URL("vendor/rhwp/rhwp.js", import.meta.url).href;
const rhwpWasmUrl = new URL("vendor/rhwp/rhwp_bg.wasm", import.meta.url).href;
const rhwp = await import(rhwpJsUrl);
await rhwp.default({ module_or_path: rhwpWasmUrl });

// ── 3. DOM lookups ────────────────────────────────────────────────────────
const els = {
  filename: document.getElementById("filename"),
  autofix: document.getElementById("autofix"),
  download: document.getElementById("download"),
  pagenav: document.getElementById("pagenav"),
  pagePrev: document.getElementById("page-prev"),
  pageNext: document.getElementById("page-next"),
  pageInput: document.getElementById("page-input"),
  pageTotal: document.getElementById("page-total"),
  container: document.getElementById("pages-container"),
  status: document.getElementById("status"),
  zoomControls: document.getElementById("zoom-controls"),
  zoomSlider: document.getElementById("zoom-slider"),
  zoomLabel: document.getElementById("zoom-label"),
  liveStatus: document.getElementById("live-status"),
  refresh: document.getElementById("refresh"),
};

// ── 4. State ──────────────────────────────────────────────────────────────
const state = {
  fileBytes: null,
  filename: "",
  filePath: "",
  fileHash: "",
  checking: false,
  // Auto-correction defaults to ON — for the vast majority of agent-emitted
  // and Hancom-saved hwp files, reflowLinesegs() produces the visually
  // expected output. The toolbar toggle still flips it for raw inspection.
  autoFix: true,
  pageCount: 0,
  currentPage: 1,
  // Zoom multiplier on the fit-to-container baseline. 1.0 = fits the pane.
  // CSS-scale only — canvas pixel buffer is set once in render(). Past ~1.5x
  // upscaling becomes visible on standard-DPI screens; we accept that for
  // now to keep slider drags instant. Re-rasterise on demand is a future
  // refinement.
  zoom: 1,
};
const ZOOM_MIN = 0.2;

const DEFAULT_DOCUMENT_URL =
  "https://raw.githubusercontent.com/joowan1108/DrGrading/learnable_dist/graduation.hwpx";
const REFRESH_INTERVAL_MS = 30_000;
const ZOOM_MAX = 2;
const ZOOM_STEP = 0.05;

// ── 5. UI helpers ─────────────────────────────────────────────────────────
function syncAutofixButton() {
  const on = state.autoFix === true;
  els.autofix.textContent = `자동 보정 ${on ? "ON" : "OFF"}`;
  els.autofix.classList.toggle("on", on);
}
syncAutofixButton();

function setStatus(text, kind = "info") {
  els.status.textContent = text;
  els.status.className = `status${kind === "error" ? " error" : ""}`;
  els.status.style.display = "block";
}
function clearStatus() { els.status.style.display = "none"; }

function syncPageNav() {
  if (state.pageCount <= 0) {
    els.pagenav.hidden = true;
    els.zoomControls.hidden = true;
    return;
  }
  els.pagenav.hidden = false;
  els.zoomControls.hidden = false;
  els.pageInput.max = String(state.pageCount);
  els.pageInput.value = String(state.currentPage);
  els.pageTotal.textContent = String(state.pageCount);
  els.pagePrev.disabled = state.currentPage <= 1;
  els.pageNext.disabled = state.currentPage >= state.pageCount;
}

function scrollToPage(n) {
  const idx = Math.max(1, Math.min(state.pageCount, n));
  const wrap = els.container.querySelector(`.hwp-page[data-page-num="${idx}"]`);
  if (wrap) wrap.scrollIntoView({ behavior: "smooth", block: "start" });
  state.currentPage = idx;
  syncPageNav();
}

// ── 6. Layout helpers ─────────────────────────────────────────────────────
const MAX_CANVAS_PIXELS = 67_108_864; // ≈ 8192 × 8192
function pickEffectiveDpr(pageW, pageH, zoom, rawDpr) {
  const phys = pageW * zoom * rawDpr * pageH * zoom * rawDpr;
  if (phys <= MAX_CANVAS_PIXELS) return rawDpr;
  const limited = Math.sqrt(MAX_CANVAS_PIXELS / (pageW * zoom * pageH * zoom));
  return Math.max(1, Math.floor(limited));
}

function fit() {
  const wraps = els.container.querySelectorAll(".hwp-page");
  if (wraps.length === 0) return;
  // At zoom=1 we want the page rendered at its NATURAL width (or fit-to-pane,
  // whichever is smaller). The earlier "always fill 1100px" rule made A4
  // (~595px native) look bloated on wide Code panes — 100% felt too big.
  // Now 100% = 1:1, zoom in to enlarge, zoom out to shrink. 1100px is kept
  // as a final ceiling for pathological wide pages.
  let maxPageW = 0;
  wraps.forEach((wrap) => {
    const canvas = wrap.querySelector("canvas");
    if (!canvas) return;
    const pw = parseFloat(canvas.dataset.pageWidth || "0");
    if (pw > maxPageW) maxPageW = pw;
  });
  const containerW = Math.max(0, els.container.clientWidth - 32);
  const naturalCap = maxPageW > 0 ? maxPageW : 1100;
  const baseAvail = Math.max(
    280,
    Math.min(naturalCap, containerW || naturalCap, 1100),
  );
  const avail = baseAvail * state.zoom;
  wraps.forEach((wrap) => {
    const canvas = wrap.querySelector("canvas");
    if (!canvas) return;
    const pageW = parseFloat(canvas.dataset.pageWidth || "0");
    const pageH = parseFloat(canvas.dataset.pageHeight || "0");
    if (pageW <= 0 || pageH <= 0) return;
    const ratio = avail / pageW;
    const scaledH = pageH * ratio;
    wrap.style.width = `${avail}px`;
    wrap.style.height = `${scaledH}px`;
    canvas.style.width = `${avail}px`;
    canvas.style.height = `${scaledH}px`;
  });
}

function syncZoomUi() {
  els.zoomSlider.value = String(state.zoom);
  els.zoomLabel.textContent = `${Math.round(state.zoom * 100)}%`;
}

function setZoom(z) {
  const clamped = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, z));
  if (clamped === state.zoom) return;
  state.zoom = clamped;
  syncZoomUi();
  fit();
}

const ro = new ResizeObserver(() => fit());
ro.observe(els.container);

const pageObserver = new IntersectionObserver(
  (entries) => {
    let bestIdx = state.currentPage;
    let bestRatio = 0;
    entries.forEach((e) => {
      if (e.intersectionRatio > bestRatio) {
        bestRatio = e.intersectionRatio;
        const num = Number(e.target.dataset.pageNum);
        if (Number.isFinite(num)) bestIdx = num;
      }
    });
    if (bestRatio > 0 && bestIdx !== state.currentPage) {
      state.currentPage = bestIdx;
      syncPageNav();
    }
  },
  { root: els.container, threshold: [0.1, 0.5, 0.9] },
);

// ── 7. Render pipeline ────────────────────────────────────────────────────
function setLiveStatus(text, kind = "") {
  els.liveStatus.textContent = text;
  els.liveStatus.className = `live-status${kind ? ` ${kind}` : ""}`;
}

async function digest(bytes) {
  const hash = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(hash), (b) => b.toString(16).padStart(2, "0")).join("");
}

function documentSource() {
  const params = new URLSearchParams(window.location.search);
  return params.get("file") || DEFAULT_DOCUMENT_URL;
}

async function fetchLatest({ silent = false } = {}) {
  if (state.checking) return;
  state.checking = true;
  setLiveStatus("최신 버전 확인 중", "checking");
  if (!silent) setStatus("문서 불러오는 중…");

  const source = documentSource();
  try {
    const requestUrl = new URL(source, window.location.href);
    requestUrl.searchParams.set("_updated", String(Date.now()));
    const res = await fetch(requestUrl, { cache: "no-store" });
    if (!res.ok) throw new Error(`서버 응답 ${res.status}`);

    const bytes = new Uint8Array(await res.arrayBuffer());
    const hash = await digest(bytes);
    const checkedAt = new Date().toLocaleTimeString("ko-KR", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });

    if (state.fileHash === hash) {
      setLiveStatus(`최신 상태 · ${checkedAt}`);
      return;
    }

    const wasLoaded = Boolean(state.fileBytes);
    state.fileBytes = bytes;
    state.fileHash = hash;
    state.filePath = source;
    state.filename = decodeURIComponent(new URL(source, window.location.href).pathname.split("/").pop()) || "graduation.hwpx";
    els.filename.textContent = state.filename;
    els.download.href = source;
    els.download.setAttribute("download", state.filename);
    if (!wasLoaded) state.currentPage = 1;
    await render();
    setLiveStatus(`${wasLoaded ? "새 버전 반영" : "연결됨"} · ${checkedAt}`);
  } catch (err) {
    setLiveStatus("연결 지연 · 다시 시도 중", "error");
    if (!state.fileBytes) setStatus(`불러오기 실패: ${err.message}`, "error");
  } finally {
    state.checking = false;
  }
}

async function render() {
  if (!state.fileBytes) return;

  // Tear down previous canvases + observer attachments.
  els.container.querySelectorAll(".hwp-page").forEach((n) => {
    pageObserver.unobserve(n);
    n.remove();
  });

  let doc;
  try {
    doc = new rhwp.HwpDocument(state.fileBytes);
  } catch (err) {
    setStatus(`문서 파싱 실패: ${err.message}`, "error");
    return;
  }

  try {
    if (state.autoFix === true) {
      try { doc.reflowLinesegs(); }
      catch (err) { console.warn("[claw-hwp] reflowLinesegs failed:", err); }
    }

    state.pageCount = doc.pageCount();
    state.currentPage = Math.min(state.currentPage || 1, state.pageCount);
    syncPageNav();

    // Geometry first — every getPageInfo call before any renderPageToCanvas.
    const geoms = [];
    for (let i = 0; i < state.pageCount; i++) {
      try {
        const info = JSON.parse(doc.getPageInfo(i));
        geoms.push({ width: Number(info.width) || 0, height: Number(info.height) || 0 });
      } catch (err) {
        console.error(`[claw-hwp] getPageInfo(${i}) failed:`, err);
        geoms.push({ width: 0, height: 0 });
      }
    }

    const rawDpr = window.devicePixelRatio || 1;
    const zoom = 1;
    const frag = document.createDocumentFragment();
    const wraps = [];

    for (let i = 0; i < state.pageCount; i++) {
      const { width: pageW, height: pageH } = geoms[i];
      if (pageW <= 0 || pageH <= 0) continue;
      const dpr = pickEffectiveDpr(pageW, pageH, zoom, rawDpr);
      const renderScale = zoom * dpr;

      const canvas = document.createElement("canvas");
      canvas.width = Math.round(pageW * renderScale);
      canvas.height = Math.round(pageH * renderScale);
      canvas.dataset.pageWidth = String(pageW);
      canvas.dataset.pageHeight = String(pageH);
      canvas.style.display = "block";
      canvas.style.position = "absolute";
      canvas.style.top = "0";
      canvas.style.left = "0";
      canvas.style.width = `${pageW}px`;
      canvas.style.height = `${pageH}px`;
      canvas.style.background = "#fff";

      try {
        doc.renderPageToCanvas(i, canvas, renderScale);
      } catch (err) {
        console.error(`[claw-hwp] renderPageToCanvas(${i}) failed:`, err);
        continue;
      }

      // Always call getPageTextLayout (and discard) — without it rhwp keeps
      // its internal borrow on the doc and the next render panics.
      try { JSON.parse(doc.getPageTextLayout(i)); } catch {}

      const wrap = document.createElement("div");
      wrap.className = "hwp-page";
      wrap.dataset.pageNum = String(i + 1);
      wrap.style.position = "relative";
      wrap.style.background = "#fff";
      wrap.style.boxShadow = "0 1px 4px rgba(0, 0, 0, 0.45)";
      wrap.style.width = `${pageW}px`;
      wrap.style.height = `${pageH}px`;
      wrap.appendChild(canvas);
      frag.appendChild(wrap);
      wraps.push(wrap);
    }

    els.container.appendChild(frag);
    wraps.forEach((wrap) => pageObserver.observe(wrap));
    requestAnimationFrame(() => fit());
    clearStatus();
  } finally {
    if (typeof doc.free === "function") doc.free();
  }
}

// ── 8. Event wiring ───────────────────────────────────────────────────────
els.autofix.addEventListener("click", () => {
  state.autoFix = !state.autoFix;
  syncAutofixButton();
  render();
});

els.pagePrev.addEventListener("click", () => scrollToPage(state.currentPage - 1));
els.pageNext.addEventListener("click", () => scrollToPage(state.currentPage + 1));
els.pageInput.addEventListener("change", () => {
  const v = Number(els.pageInput.value);
  if (Number.isFinite(v)) scrollToPage(v);
});
els.refresh.addEventListener("click", () => fetchLatest());
els.pageInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    const v = Number(els.pageInput.value);
    if (Number.isFinite(v)) scrollToPage(v);
  }
});

// Zoom — slider drag, label click for reset, Ctrl/Cmd+wheel.
els.zoomSlider.addEventListener("input", (e) => {
  const v = Number(e.target.value);
  if (Number.isFinite(v)) setZoom(v);
});
els.zoomLabel.addEventListener("click", () => setZoom(1));
els.container.addEventListener("wheel", (e) => {
  // Ctrl on Win/Linux, Cmd on macOS. Trackpad pinch-zoom also surfaces as
  // wheel + ctrlKey on macOS, so this picks up both gestures.
  if (!(e.ctrlKey || e.metaKey)) return;
  e.preventDefault();
  const delta = e.deltaY < 0 ? ZOOM_STEP : -ZOOM_STEP;
  setZoom(state.zoom + delta);
}, { passive: false });
syncZoomUi();

// ── 9. Heartbeat ──────────────────────────────────────────────────────────
// While this tab is open, ping the server so it knows we're alive. When the
// tab closes the pings stop, the server's idle timer fires, and the process
// self-kills — sparing the user from having to remember to stop it. Only
// applies when running under preview-server (port 3737); under static
// hosting (GitHub Pages) there's nothing on the other end of the heartbeat.
if (location.port === "3737") {
  setInterval(() => {
    fetch("/__heartbeat", { method: "GET", cache: "no-store" }).catch(() => {});
  }, 30_000);
}

setInterval(() => fetchLatest({ silent: true }), REFRESH_INTERVAL_MS);
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") fetchLatest({ silent: true });
});

fetchLatest();
