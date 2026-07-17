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
    <p class="qa-meta">${data.model} · thinking ${data.thinking_disabled ? "off" : "on"} · ${data.latency_ms}ms</p>
  `;
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
