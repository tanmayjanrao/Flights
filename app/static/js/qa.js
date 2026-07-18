const sampleSelect = document.getElementById("sampleSelect");
const analyzeBtn = document.getElementById("analyzeBtn");
const transcriptView = document.getElementById("transcriptView");
const reportView = document.getElementById("reportView");
const ollamaBadge = document.getElementById("ollamaBadge");

let samples = [];

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
    .map(
      (m) => `<div class="msg msg-${m.speaker}"><span class="who">${m.speaker}</span>${escapeHtml(m.text)}</div>`
    )
    .join("");
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function renderReport(data) {
  const a = data.analysis;
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
    <div class="qa-score-hero">
      <span class="num ${cls}">${pct}</span>
      <span class="cat">${a.category.replace(/_/g, " ")} · ${a.resolved ? "resolved" : "unresolved"}${a.escalation_needed ? " · needs escalation" : ""}</span>
    </div>
    <div class="qa-score-bars">${bars}</div>
    ${flags}
    <p>${escapeHtml(a.summary)}</p>
    ${strengths}
    ${improvements}
    ${renderHoldTimeCompliance(a.hold_time_compliance)}
    ${renderIdleProtocolCompliance(a.idle_protocol_compliance)}
    <p class="qa-meta">${data.model} · thinking ${data.thinking_disabled ? "off" : "on"} · ${data.latency_ms}ms</p>
  `;
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
      const pillText = h.exceeded ? `+${fmtSecs(h.overage_seconds)} over` : "within stated time";
      return `<div class="qa-timing-row">
        <span class="qa-timing-pill ${pillCls}">${pillText}</span>
        <span>stated ${fmtSecs(h.stated_seconds)}, actual ${fmtSecs(h.actual_seconds)}</span>
      </div>`;
    })
    .join("");
  return `<div class="qa-timing-block"><h3>Hold-time compliance <span class="qa-soft-flag">soft flag</span></h3>${rows}</div>`;
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
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing… (CPU inference, may take a bit)";
  reportView.innerHTML = `<p class="empty-state">Running qwen3:4b locally, please wait…</p>`;

  try {
    const res = await fetch("/api/qa/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(transcript),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Analysis failed");
    renderReport(data);
  } catch (err) {
    reportView.innerHTML = `<p class="error-state">${escapeHtml(err.message)}</p>`;
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze chat";
  }
});

sampleSelect.addEventListener("change", () => {
  const idx = Number(sampleSelect.value);
  if (samples[idx]) {
    renderTranscript(samples[idx]);
    reportView.innerHTML = `<p class="empty-state">No analysis yet.</p>`;
  }
});

loadOllamaHealth();
loadSamples();
