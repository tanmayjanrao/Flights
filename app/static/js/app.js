const form = document.getElementById("searchForm");
const boardBody = document.getElementById("boardBody");
const tabs = document.querySelectorAll(".tab");
const submitBtn = form.querySelector(".btn-search");
let mode = "number";

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    tabs.forEach((t) => { t.classList.remove("active"); t.setAttribute("aria-selected", "false"); });
    tab.classList.add("active");
    tab.setAttribute("aria-selected", "true");
    mode = tab.dataset.mode;
    document.querySelectorAll('[data-group="number"]').forEach((el) => el.classList.toggle("hidden", mode !== "number"));
    document.querySelectorAll('[data-group="route"]').forEach((el) => el.classList.toggle("hidden", mode !== "route"));
  });
});

async function loadHealth() {
  const badgeWrap = document.getElementById("providerBadges");
  try {
    const res = await fetch("/api/flights/health");
    const data = await res.json();
    badgeWrap.innerHTML = Object.entries(data.providers)
      .map(([name, ok]) => `<span class="provider-pill ${ok ? "on" : "off"}">${name} ${ok ? "●" : "○"}</span>`)
      .join("");
  } catch {
    badgeWrap.innerHTML = "";
  }
}
loadHealth();

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  submitBtn.disabled = true;
  submitBtn.textContent = "Tracking…";
  boardBody.innerHTML = `<p class="empty-state">Pulling live data…</p>`;

  try {
    if (mode === "number") {
      const flightIata = document.getElementById("flightNumber").value.trim().toUpperCase();
      if (!flightIata) throw new Error("Enter a flight number, e.g. AA1004");
      const res = await fetch(`/api/flights/status/${encodeURIComponent(flightIata)}`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Flight not found");
      renderBoard([data.flight], data.source, data.fallback_used);
    } else {
      const dep = document.getElementById("depIata").value.trim().toUpperCase();
      const arr = document.getElementById("arrIata").value.trim().toUpperCase();
      const status = document.getElementById("statusFilter").value;
      const qs = new URLSearchParams();
      if (dep) qs.set("dep_iata", dep);
      if (arr) qs.set("arr_iata", arr);
      if (status) qs.set("flight_status", status);
      if (![...qs.keys()].length) throw new Error("Enter at least a departure or arrival airport");
      const res = await fetch(`/api/flights/search?${qs.toString()}`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Search failed");
      renderBoard(data.results, data.source, data.fallback_used);
    }
  } catch (err) {
    boardBody.innerHTML = `<p class="error-state">${escapeHtml(err.message)}</p>`;
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Track";
  }
});

function renderBoard(flights, source, fallbackUsed) {
  if (!flights || !flights.length) {
    boardBody.innerHTML = `<p class="empty-state">No flights matched that search.</p>`;
    return;
  }

  boardBody.innerHTML = flights.map((f, idx) => rowTemplate(f, idx, source, fallbackUsed)).join("");

  boardBody.querySelectorAll(".board-row").forEach((row) => {
    row.addEventListener("click", () => {
      const detail = row.nextElementSibling;
      if (detail && detail.classList.contains("detail-wrap")) {
        detail.classList.toggle("open");
        detail.style.display = detail.classList.contains("open") ? "block" : "none";
      }
    });
  });
}

function rowTemplate(f, idx, source, fallbackUsed) {
  const code = f.flight_iata || f.flight_number || "—";
  const letters = code.split("").map((c, i) => `<span style="animation-delay:${i * 30}ms">${escapeHtml(c)}</span>`).join("");
  const statusClass = `status-${f.status || "unknown"}`;

  return `
    <div class="board-row" style="animation-delay:${idx * 60}ms">
      <span class="flight-code">${letters}</span>
      <span class="route-text"><b>${escapeHtml(f.departure?.iata || "?")}</b> → <b>${escapeHtml(f.arrival?.iata || "?")}</b></span>
      ${timeCell(f.departure)}
      ${timeCell(f.arrival)}
      <span class="status-pill ${statusClass}">${escapeHtml(f.status || "unknown")}</span>
    </div>
    <div class="detail-wrap" style="display:none;">
      <div class="detail-panel">
        ${detailItem("Airline", f.airline?.iata || f.airline?.name || "—")}
        ${detailItem("Aircraft", f.aircraft?.iata_type || "—")}
        ${detailItem("Dep. terminal / gate", `${f.departure?.terminal || "—"} / ${f.departure?.gate || "—"}`)}
        ${detailItem("Arr. terminal / gate", `${f.arrival?.terminal || "—"} / ${f.arrival?.gate || "—"}`)}
        ${detailItem("Dep. delay", f.departure?.delay_minutes != null ? `${f.departure.delay_minutes} min` : "—")}
        ${detailItem("Arr. delay", f.arrival?.delay_minutes != null ? `${f.arrival.delay_minutes} min` : "—")}
        <p class="source-tag ${fallbackUsed ? "fallback" : ""}">
          Data via <b>${escapeHtml(source)}</b>${fallbackUsed ? " (fallback — primary provider was unavailable)" : ""}
        </p>
      </div>
    </div>
  `;
}

function timeCell(leg) {
  if (!leg) return `<span class="time-cell">—</span>`;
  const time = leg.actual || leg.estimated || leg.scheduled;
  // Always render in THIS leg's own airport timezone, never the viewer's browser
  // timezone - a DEL departure shows IST, an LHR arrival shows BST, in the same response.
  const label = formatTime(time, leg.timezone);
  const delay = leg.delay_minutes ? `<span class="delay">+${leg.delay_minutes}m</span>` : "";
  return `<span class="time-cell">${label}${delay}</span>`;
}

function formatTime(value, tz) {
  if (!value) return "—";
  const d = new Date(value);
  if (isNaN(d.getTime())) return escapeHtml(String(value));
  const opts = { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" };
  if (tz) {
    opts.timeZone = tz;
    opts.timeZoneName = "short"; // adds the "IST" / "BST" style abbreviation
  }
  return d.toLocaleString(undefined, opts);
}

function detailItem(label, value) {
  return `<div class="detail-item"><span class="label">${escapeHtml(label)}</span><span class="value">${escapeHtml(String(value))}</span></div>`;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
