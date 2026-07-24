const sampleSelect = document.getElementById("sampleSelect");
const analyzeBtn = document.getElementById("analyzeBtn");
const cancelBtn = document.getElementById("cancelBtn");
const transcriptView = document.getElementById("transcriptView");
const reportView = document.getElementById("reportView");
const ollamaBadge = document.getElementById("ollamaBadge");

let samples = [];

// Cache of completed analyses, keyed by transcript_id, so re-picking a
// sample already analyzed this session doesn't re-run the multi-minute
// CPU inference. Cleared on page reload (in-memory only).
const analysisCache = new Map();

// Lets the Cancel button actually abort the in-flight fetch, and lets us
// tell an aborted request apart from a real failure.
let activeController = null;
let elapsedTimerId = null;

function scoreClass(pct) {
  if (pct >= 75) return "good";
  if (pct >= 50) return "mid";
  return "bad";
}

async function loadOllamaHealth() {
  try {
    const res = await fetch("/api/qa/health");
    const data = await res.json();
    if (data.status === "ok") {
      ollamaBadge.textContent = `${data.model} ●`;
      ollamaBadge.className = "provider-pill on";
    } else {
      ollamaBadge.textContent = `${data.model} — ${data.detail || "unavailable"}`;
      ollamaBadge.className = "provider-pill off";
    }
  } catch {
    ollamaBadge.textContent = "ollama unreachable";
    ollamaBadge.className = "provider-pill off";
  }
}

async function loadSamples() {
  const res = await fetch("/api/qa/samples");
  samples = await res.json();
  sampleSelect.innerHTML = samples
    .map((t, i) => `<option value="${i}">${t.transcript_id}</option>`)
    .join("");
  if (samples.length) renderTranscript(samples[0]);
}

function renderTranscript(transcript) {
  transcriptView.innerHTML = transcript.messages
    .map((m) => {
      const time = m.elapsed_seconds !== null && m.elapsed_seconds !== undefined ? fmtClock(m.elapsed_seconds) : null;
      const timeEl = time ? `<span class="msg-time">${time}</span>` : "";
      return `<div class="msg msg-${m.speaker}"><span class="who">${m.speaker}${timeEl}</span>${escapeHtml(m.text)}</div>`;
    })
    .join("");
}

// mm:ss timestamp for the transcript panel, from elapsed_seconds (seconds
// since chat start) - distinct from fmtSecs below, which formats a duration
// like "3m 20s" for the QA report's timing blocks.
function fmtClock(s) {
  const total = Math.max(0, Math.round(s));
  const m = Math.floor(total / 60);
  const sec = total % 60;
  return `${m}:${String(sec).padStart(2, "0")}`;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function renderReport(data, fromCache) {
  const a = data.analysis;
  const cachedNote = fromCache
    ? `<p class="qa-cached-note">Showing a cached result from earlier this session — click Analyze to re-run.</p>`
    : "";
  const pct = a.overall_score;
  const cls = scoreClass(pct);

  const bars = Object.entries(a.scores)
    .map(([label, val]) => {
      const barPct = (val / 5) * 100;
      const barCls = scoreClass(barPct);
      const niceLabel = label.replace(/_/g, " ");
      return `<div class="qa-score-bar">
        <span class="label">${niceLabel}</span>
        <span class="track"><span class="fill ${barCls}" style="width:${barPct}%"></span></span>
        <span>${val}/5</span>
      </div>`;
    })
    .join("");

  const flags = a.flags.length
    ? `<div class="qa-flags">${a.flags.map((f) => `<span class="qa-flag-pill">${f.replace(/_/g, " ")}</span>`).join("")}</div>`
    : "";

  const strengths = a.strengths.length
    ? `<div class="qa-list-block"><h3>Strengths</h3><ul>${a.strengths.map((s) => `<li>${escapeHtml(s)}</li>`).join("")}</ul></div>`
    : "";

  const improvements = a.improvements.length
    ? `<div class="qa-list-block"><h3>Improvements</h3><ul>${a.improvements.map((s) => `<li>${escapeHtml(s)}</li>`).join("")}</ul></div>`
    : "";

  reportView.innerHTML = `
    ${cachedNote}
    <div class="qa-score-hero">
      <span class="num ${cls}">${pct}</span>
      <span class="cat">${a.category.replace(/_/g, " ")} · ${a.resolved ? "resolved" : "unresolved"}${a.escalation_needed ? " · needs escalation" : ""}</span>
    </div>
    <div class="qa-score-bars">${bars}</div>
    ${flags}
    <p>${escapeHtml(a.summary)}</p>
    ${renderChatFlow(a.chat_flow)}
    ${strengths}
    ${improvements}
    ${renderHoldTimeCompliance(a.hold_time_compliance)}
    ${renderIdleProtocolCompliance(a.idle_protocol_compliance)}
    <p class="qa-meta">${data.model} · thinking ${data.thinking_disabled ? "off" : "on"} · ${data.latency_ms}ms</p>
  `;
}

// The 7-stage chat flow (see prompts.py) - agent greeting through close.
// The LLM reports followed/not-followed + a short note per stage; this just
// renders that list, it doesn't compute anything.
function renderChatFlow(chatFlow) {
  if (!chatFlow || !chatFlow.length) return "";
  const rows = chatFlow
    .map((s) => {
      const pillCls = s.followed ? "ok" : "warn";
      const pillText = s.followed ? "followed" : "missed";
      const niceStage = s.stage.replace(/_/g, " ");
      return `<div class="qa-flow-row">
        <span class="qa-flow-name">${escapeHtml(niceStage)}</span>
        <span class="qa-timing-pill qa-flow-pill ${pillCls}">${pillText}</span>
        <span class="qa-flow-note">${escapeHtml(s.note || "")}</span>
      </div>`;
    })
    .join("");
  return `<div class="qa-flow-block"><h3>Chat flow</h3>${rows}</div>`;
}

function startElapsedTimer() {
  const startedAt = Date.now();
  analyzeBtn.textContent = "Analyzing… 0s elapsed";
  elapsedTimerId = setInterval(() => {
    const secs = Math.floor((Date.now() - startedAt) / 1000);
    analyzeBtn.textContent = `Analyzing… ${secs}s elapsed`;
  }, 1000);
}

function stopElapsedTimer() {
  if (elapsedTimerId !== null) {
    clearInterval(elapsedTimerId);
    elapsedTimerId = null;
  }
}

function fmtSecs(s) {
  if (s === null || s === undefined) return "—";
  const m = Math.floor(s / 60);
  const rem = Math.round(s % 60);
  return m > 0 ? `${m}m ${rem}s` : `${rem}s`;
}

// Both of these are computed in plain Python (timing_checks.py), not by the
// LLM - see the note in that module for why.
function renderHoldTimeCompliance(hold) {
  if (!hold.evaluated) {
    return `<div class="qa-timing-block"><h3>Hold-time compliance</h3><p class="qa-timing-note">${escapeHtml(hold.note || "Not evaluated.")}</p></div>`;
  }
  if (!hold.holds.length) {
    return `<div class="qa-timing-block"><h3>Hold-time compliance</h3><p class="qa-timing-note">${escapeHtml(hold.note || "No hold/wait duration was stated.")}</p></div>`;
  }
  const rows = hold.holds
    .map((h) => {
      const pillCls = h.exceeded ? "warn" : "ok";
      const pillText = h.exceeded ? `+${fmtSecs(h.overage_seconds)} over policy` : "returned within 5 min";
      const statedCls = h.stated_duration_compliant ? "ok" : "warn";
      const statedText = h.stated_duration_compliant ? "stated 5 min" : `stated ${fmtSecs(h.stated_seconds)} (must be 5 min)`;
      return `<div class="qa-timing-row">
        <span class="qa-timing-pill ${statedCls}">${statedText}</span>
        <span class="qa-timing-pill ${pillCls}">${pillText}</span>
        <span>actual ${fmtSecs(h.actual_seconds)} vs ${fmtSecs(h.policy_seconds)} policy</span>
      </div>`;
    })
    .join("");
  return `<div class="qa-timing-block"><h3>Hold-time compliance <span class="qa-soft-flag">5 min policy</span></h3>${rows}</div>`;
}

function renderIdleProtocolCompliance(idle) {
  if (!idle.evaluated) {
    return `<div class="qa-timing-block"><h3>Idle-protocol adherence</h3><p class="qa-timing-note">${escapeHtml(idle.note || "Not evaluated.")}</p></div>`;
  }
  if (!idle.windows.length) {
    return `<div class="qa-timing-block"><h3>Idle-protocol adherence</h3><p class="qa-timing-note">${escapeHtml(idle.note || "No idle window long enough to check.")}</p></div>`;
  }
  const rows = idle.windows
    .map((w) => {
      const pillCls = w.violations.length ? "warn" : "ok";
      const pillText = w.violations.length ? w.violations.join(", ").replace(/_/g, " ") : "on protocol";
      const checkinPart = w.first_checkin_seconds !== null && w.first_checkin_seconds !== undefined
        ? `check-in at ${fmtSecs(w.first_checkin_seconds)}`
        : "no check-in";
      const finalPart = w.final_notice_sent ? `, closed at ${fmtSecs(w.final_notice_seconds)}` : "";
      return `<div class="qa-timing-row">
        <span class="qa-timing-pill ${pillCls}">${pillText}</span>
        <span>idle ${fmtSecs(w.idle_duration_seconds)} - ${checkinPart}${finalPart}</span>
      </div>`;
    })
    .join("");
  return `<div class="qa-timing-block"><h3>Idle-protocol adherence</h3>${rows}</div>`;
}

analyzeBtn.addEventListener("click", async () => {
  const idx = Number(sampleSelect.value);
  const transcript = samples[idx];
  if (!transcript) return;

  renderTranscript(transcript);

  activeController = new AbortController();
  analyzeBtn.disabled = true;
  cancelBtn.hidden = false;
  startElapsedTimer();
  reportView.innerHTML = `<p class="empty-state">Running qwen3:4b locally, please wait…</p>`;

  try {
    const res = await fetch("/api/qa/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(transcript),
      signal: activeController.signal,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Analysis failed");
    analysisCache.set(transcript.transcript_id, data);
    renderReport(data, false);
  } catch (err) {
    if (err.name === "AbortError") {
      reportView.innerHTML = `<p class="empty-state">Analysis cancelled.</p>`;
    } else {
      reportView.innerHTML = `<p class="error-state">${escapeHtml(err.message)}</p>`;
    }
  } finally {
    stopElapsedTimer();
    activeController = null;
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze chat";
    cancelBtn.hidden = true;
  }
});

cancelBtn.addEventListener("click", () => {
  if (activeController) activeController.abort();
});

sampleSelect.addEventListener("change", () => {
  const idx = Number(sampleSelect.value);
  const transcript = samples[idx];
  if (!transcript) return;

  renderTranscript(transcript);
  const cached = analysisCache.get(transcript.transcript_id);
  if (cached) {
    renderReport(cached, true);
  } else {
    reportView.innerHTML = `<p class="empty-state">No analysis yet.</p>`;
  }
});

loadOllamaHealth();
loadSamples();
