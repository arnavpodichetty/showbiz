const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (p) => `${Math.round(p * 100)}%`;
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");

let current = null;    // { mbid, name, disambiguation }
let overview = null;
let lastResult = null;

// confidence tiers: how firmly each song is written on the sheet
const tier = (p) => (p >= 0.85 ? "t-lock" : p >= 0.55 ? "t-likely" : "t-gamble");

async function api(path, opts = {}) {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch {}
    const err = new Error(msg);
    err.status = r.status;
    throw err;
  }
  return r.json();
}

function setStatus(text, isError = false) {
  const el = $("#status");
  el.textContent = text;
  el.classList.toggle("is-error", isError);
}

function fmtDate(iso, opts = { weekday: "short", day: "numeric", month: "short" }) {
  const [y, m, d] = iso.split("-").map(Number);
  return new Date(y, m - 1, d).toLocaleDateString("en-GB", opts);
}

// stable little wobble per line so the sheet looks hand-written, not typeset
function wobble(str) {
  let h = 2166136261;
  for (const c of str) h = Math.imul(h ^ c.charCodeAt(0), 16777619);
  const a = ((h >>> 0) % 1000) / 1000, b = ((h >>> 10) % 1000) / 1000;
  return { rot: (a - 0.5) * 1.4, dx: b * 6 };
}

// ---------- views ----------
document.querySelectorAll(".nav-link").forEach((btn) =>
  btn.addEventListener("click", () => {
    const view = btn.dataset.view;
    document.querySelectorAll(".nav-link").forEach((b) => b.removeAttribute("aria-current"));
    btn.setAttribute("aria-current", "page");
    $("#view-predict").hidden = view !== "predict";
    $("#view-past").hidden = view !== "past";
    if (view === "past") loadPast(false);
  })
);

// ---------- artist search ----------
$("#search-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("#search-q").value.trim();
  if (!q) return;
  setStatus("Searching setlist.fm…");
  $("#search-results").innerHTML = "";
  try {
    const res = await api(`/api/artists/search?q=${encodeURIComponent(q)}`);
    if (!res.length) return setStatus(`No artists match "${q}". Check the spelling or try their full name.`, true);
    setStatus("");
    $("#search-results").innerHTML = res
      .map((a, i) => `<li><button class="pick" data-i="${i}">${esc(a.name)}${a.disambiguation ? `<small>${esc(a.disambiguation)}</small>` : ""}</button></li>`)
      .join("");
    $("#search-results").querySelectorAll(".pick").forEach((el) => el.addEventListener("click", () => chooseArtist(res[+el.dataset.i])));
    $("#search-results .pick")?.focus();
  } catch {
    setStatus("Couldn't reach setlist.fm. Check your connection and search again.", true);
  }
});

$("#change-artist").addEventListener("click", () => {
  $("#chosen").hidden = true;
  $("#search-form").hidden = false;
  loadRecentArtists();
  $("#search-q").select();
  $("#search-q").focus();
});

async function loadRecentArtists() {
  const list = await api("/api/artists").catch(() => []);
  const el = $("#recent-artists");
  if (!list.length || !$("#chosen").hidden) { el.innerHTML = ""; return; }
  el.innerHTML = `Checked before: ` + list.map((a, i) => `<button class="link-button" data-i="${i}">${esc(a.name)}</button>`).join("");
  el.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => chooseArtist(list[+b.dataset.i])));
}

async function chooseArtist(a) {
  current = { mbid: a.mbid, name: a.name, disambiguation: a.disambiguation || "" };
  $("#search-results").innerHTML = "";
  $("#recent-artists").innerHTML = "";
  $("#search-form").hidden = true;
  $("#chosen").hidden = false;
  $("#chosen-name").textContent = a.name;
  ["#show-form", "#upcoming", "#based-on", "#all-songs"].forEach((s) => ($(s).hidden = true));
  sheetMessage(a.name, "Reading their old setlists…");
  try {
    await api("/api/sync", { method: "POST", body: JSON.stringify(current) });
    pollSync();
  } catch {
    setStatus("Couldn't start downloading setlists. Wait a minute and pick the artist again.", true);
  }
}

// Shows the prediction as soon as the newest setlists are in ("ready"), keeps tracking the
// download of older ones, then quietly re-predicts with everything once it's done.
async function pollSync(mbid = current.mbid, shown = false) {
  if (!current || current.mbid !== mbid) return;          // switched artist: stop tracking this one
  let s;
  try { s = await api(`/api/sync/${mbid}`); } catch { return setTimeout(() => pollSync(mbid, shown), 1500); }
  if (!current || current.mbid !== mbid) return;
  const total = s.total ? Math.min(s.total, 1000) : null;
  if (s.status === "error") {
    if (shown) return setStatus("");                    // keep the early prediction; older shows just didn't arrive
    setStatus("Couldn't download setlists from setlist.fm. Wait a minute and pick the artist again.", true);
    sheetMessage(current.name, "Couldn't get their setlists this time.");
    return;
  }
  if (s.status === "syncing" || s.status === "training") {
    if (s.ready && !shown) {
      shown = true;
      await loadOverview();
    }
    if (shown) {
      if (s.first_time && total && s.fetched < total) setStatus(`Still reading older setlists (${s.fetched} of ${total}). The prediction will update when it's done.`);
    } else if (s.status === "training") {
      setStatus("Learning how they build a set…");
    } else {
      setStatus(total ? `Downloading setlists: ${s.fetched || 0} of ${total}` : "Downloading setlists…");
      sheetMessage(current.name, total ? `Reading ${s.fetched || 0} of ${total} old setlists…` : "Reading their old setlists…");
    }
    return setTimeout(() => pollSync(mbid, shown), 1000);
  }
  // done
  if (shown) {
    setStatus("");
    if (s.new > 0) predict();                             // new setlists arrived: update (keeps the form as is)
  } else {
    setStatus("");
    loadOverview();
  }
}

// ---------- overview + show form ----------
async function loadOverview() {
  try {
    overview = await api(`/api/artist/${current.mbid}/overview`);
  } catch (err) {
    const n = /found (\d+)/.exec(err.message)?.[1];
    const msg = n
      ? `Only ${n} setlists found for ${current.name}. We need at least 10 to predict a set. Try an artist who tours more often.`
      : `No recent shows found for ${current.name}. Try a different artist.`;
    setStatus(msg, true);
    sheetMessage(current.name, "Not enough shows to go on yet.");
    return;
  }
  const o = overview;

  const tours = o.tours.length ? o.tours : [];
  $("#f-tour").innerHTML =
    tours.map((t) => `<option value="${esc(t)}" ${t === o.current_tour ? "selected" : ""}>${esc(t)}</option>`).join("") +
    `<option value="__new__">A new tour</option><option value="__none__">Not part of a tour</option>`;
  if (!tours.length) $("#f-tour").value = "__none__";

  const next = o.upcoming[0];
  $("#f-date").value = next ? next.date : "";
  $("#f-city").value = next ? next.city : "";
  $("#f-country").value = next ? next.country : "";
  $("#f-fest").checked = next ? next.festival : false;
  $("#f-save").checked = false;
  $("#show-form").hidden = false;

  $("#upcoming").hidden = !o.upcoming.length;
  $("#upcoming-list").innerHTML = o.upcoming
    .map((s, i) => `<li><button class="pick" data-i="${i}" aria-pressed="${i === 0}">${esc(fmtDate(s.date, { day: "numeric", month: "short", year: "numeric" }))}, ${esc(s.city)}<small>${esc(s.venue)}${s.festival ? ", festival" : ""}</small></button></li>`)
    .join("");
  $("#upcoming-list").querySelectorAll(".pick").forEach((b) =>
    b.addEventListener("click", () => {
      const s = o.upcoming[+b.dataset.i];
      $("#upcoming-list").querySelectorAll(".pick").forEach((x) => x.setAttribute("aria-pressed", x === b));
      $("#f-date").value = s.date; $("#f-city").value = s.city; $("#f-country").value = s.country;
      $("#f-fest").checked = s.festival;
      if (s.tour && tours.includes(s.tour)) $("#f-tour").value = s.tour;
      predict();
    })
  );

  $("#recent-shows").innerHTML = o.recent.slice(0, 6)
    .map((s) => `<li><a href="${esc(s.url)}" target="_blank" rel="noopener">${esc(fmtDate(s.date, { day: "numeric", month: "short", year: "numeric" }))}, ${esc(s.city)}</a> <span>${s.n_songs} songs${s.festival ? ", festival" : ""}</span></li>`)
    .join("");
  $("#based-on").hidden = false;

  predict();
}

function isoInDays(n) {
  const d = new Date(Date.now() + n * 864e5);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

$("#show-form").addEventListener("submit", (e) => { e.preventDefault(); predict(); });

async function predict() {
  if ($("#f-save").checked && !$("#f-date").value) {
    return setStatus("Pick a date to save a prediction, so we know which show to grade it against.", true);
  }
  const btn = $("#show-form .button-primary");
  btn.disabled = true;
  btn.textContent = "Predicting…";
  try {
    const out = await api("/api/predict", {
      method: "POST",
      body: JSON.stringify({
        mbid: current.mbid, date: $("#f-date").value || null,
        city: $("#f-city").value, country: $("#f-country").value,
        tour: $("#f-tour").value, is_festival: $("#f-fest").checked, log: $("#f-save").checked,
      }),
    });
    lastResult = out;
    setStatus(out.prediction_id ? "Saved. We'll grade it in Past predictions once the real setlist is posted." : "");
    $("#f-save").checked = false;
    writeSheet(out);
  } catch {
    setStatus("Couldn't predict this show. Check the date and try again.", true);
  } finally {
    btn.disabled = false;
    btn.textContent = "Predict setlist";
  }
}

// ---------- the sheet ----------
function sheetMessage(title, note) {
  const sheet = $("#sheet");
  sheet.classList.add("is-empty");
  sheet.classList.remove("is-landing");
  $("#sheet-body").innerHTML = `<p class="hand hand-title">${esc(title)}</p><p class="hand hand-note">${esc(note)}</p>`;
}

function writeSheet(o) {
  const t = o.target;
  const byName = Object.fromEntries(o.songs.map((s) => [s.song, s]));
  const onSheet = new Set(o.setlist.map((s) => s.song));

  // rotation alternates: songs from the same swap group that didn't make the sheet.
  // Each sheet song gets at most MAX_ALTS, and no alternate is listed under two songs.
  // If a group covers several sheet songs, the alternates are shared out round-robin,
  // starting with the least certain sheet song (the one most likely to be swapped out).
  const MAX_ALTS = 3;
  const altsFor = {};
  for (const g of o.rotations) {
    const members = g.songs.filter((s) => onSheet.has(s.song)).sort((a, b) => a.p - b.p);
    if (!members.length) continue;
    const alts = g.songs.filter((s) => !onSheet.has(s.song) && s.p >= 0.05);   // already sorted, most likely first
    members.forEach((m) => (altsFor[m.song] = []));
    alts.forEach((a, k) => {
      const list = altsFor[members[k % members.length].song];
      if (list.length < MAX_ALTS) list.push(a);
    });
  }

  const kind = t.tour_mode === "none" ? "one-off show" : t.tour_mode === "new" ? "new tour" : "";
  const where = [t.city, t.is_festival ? "festival set" : kind].filter(Boolean).join(", ");
  const when = t.general ? "a typical show" : fmtDate(t.date);
  const sub = where ? `${where}, ${when}` : when.charAt(0).toUpperCase() + when.slice(1);

  let delay = 520, html = "", n = 0, encoreDrawn = false;
  const step = 58;
  for (const s of o.setlist) {
    if (s.encore && !encoreDrawn) {
      encoreDrawn = true;
      html += `<li aria-hidden="true"><svg class="encore-rule write" viewBox="0 0 400 22" preserveAspectRatio="none" style="--delay:${delay}ms"><path pathLength="1" d="M4 12 C 60 6, 120 16, 190 10 S 320 8, 396 13"/></svg></li>`;
      delay += 260;
    }
    n += 1;
    const info = byName[s.song] || {};
    const alts = altsFor[s.song] || [];
    const w = wobble(s.song);
    const dur = Math.min(520, 220 + s.song.length * 9);
    const altText = alts.length ? `<span class="alts">${alts.map((a) => `or ${esc(a.song)}`).join(", ")}</span>` : "";
    html += `<li class="song ${tier(s.p)}">
      <button class="song-btn write" aria-expanded="false" aria-controls="d${n}"
        style="--rot:${w.rot.toFixed(2)}deg;--dx:${w.dx.toFixed(1)}px;--delay:${delay}ms;--dur:${dur}ms">
        <span class="n">${n}</span>
        <span class="name">${esc(s.song)}${altText}</span>
        <span class="pct">${pct(s.p)}</span>
      </button>
      <div class="detail" id="d${n}" hidden>${detailText(s, info, alts, t)}</div>
    </li>`;
    delay += step;
  }

  const rare = o.rare_alert;
  const rareSong = rare.top[0];
  const deep = rare.p_any >= 0.03
    ? `${pct(rare.p_any)} chance of a deep cut${rareSong ? `. Maybe ${esc(rareSong.song)}?` : "."}`
    : "Don't expect any deep cuts tonight.";

  // a long break means the guess is really "their last tour's set": say so on the sheet
  const last = o.based_on.last_show;
  const from = t.general ? new Date() : new Date(t.date);
  const gapDays = (from - new Date(last.date)) / 864e5;
  const lastStr = esc(fmtDate(last.date, { day: "numeric", month: "short", year: "numeric" }));
  const stale = gapDays > 120
    ? `<p class="hand hand-aside">No shows since ${lastStr}, so this is based on their most recent shows. A new tour could change a lot.</p>`
    : "";

  const sheet = $("#sheet");
  sheet.classList.remove("is-empty", "is-landing");
  $("#sheet-body").innerHTML = `
    <h1 class="hand hand-title">${esc(current.name)}</h1>
    <p class="hand hand-sub">${esc(sub)}</p>${stale}
    <ol class="setlist" aria-label="Predicted setlist, ${o.setlist.length} songs">${html}</ol>
    <p class="hand deep-cut write" style="--delay:${delay + 120}ms;--dur:500ms">${deep}</p>`;
  void sheet.offsetWidth; // restart the animation
  if (!reduceMotion.matches) sheet.classList.add("is-landing");

  sheet.querySelectorAll(".song-btn").forEach((b) => b.addEventListener("click", () => toggleDetail(b)));

  renderAllSongs();
  $("#all-songs").hidden = false;
  if (matchMedia("(max-width: 860px)").matches) sheet.scrollIntoView({ behavior: reduceMotion.matches ? "auto" : "smooth", block: "start" });
}

function detailText(s, info, alts, t) {
  const lines = [`<strong>${pct(s.p)} chance</strong> they play it.`];
  if (info.rate_tour !== undefined) {
    lines.push(info.rate_tour >= 0.995 ? "Played at every show this tour."
      : info.rate_tour <= 0.005 ? "Not played yet this tour."
      : `Played at ${pct(info.rate_tour)} of shows this tour.`);
  }
  if (info.shows_since !== undefined) {
    lines.push(info.shows_since === 0 ? "Played at their last show." : `Last played ${info.shows_since} show${info.shows_since === 1 ? "" : "s"} ago.`);
  }
  if (info.played_in_run) lines.push(`Already played earlier in this run${t.city ? ` in ${esc(t.city)}` : ""}, so a repeat is less likely.`);
  if (s.opener_rate >= 0.5) lines.push("Usually opens the show.");
  else if (s.closer_rate >= 0.5) lines.push("Usually closes the show.");
  else if (s.encore) lines.push("Usually in the encore.");
  if (alts.length) lines.push(`Takes turns with ${alts.map((a) => `${esc(a.song)} (${pct(a.p)})`).join(" and ")}.`);
  return `<p>${lines.join(" ")}</p>`;
}

function toggleDetail(btn) {
  const d = document.getElementById(btn.getAttribute("aria-controls"));
  const open = btn.getAttribute("aria-expanded") !== "true";
  btn.setAttribute("aria-expanded", open);
  d.hidden = !open;
  if (open) {
    d.classList.remove("is-opening");
    void d.offsetWidth;
    d.classList.add("is-opening");
  }
}

function renderAllSongs() {
  $("#songs-table").innerHTML =
    `<thead><tr><th scope="col">Song</th><th scope="col" class="r">Chance</th><th scope="col" class="r">Plain play count</th><th scope="col" class="r">Shows since played</th></tr></thead><tbody>` +
    lastResult.songs.map((s) => `<tr><td>${esc(s.song)}</td><td class="r">${pct(s.p)}</td><td class="r">${pct(s.p_baseline)}</td><td class="r">${s.shows_since}</td></tr>`).join("") +
    `</tbody>`;
}

// ---------- past predictions ----------
$("#btn-refresh").addEventListener("click", () => loadPast(true));

async function loadPast(refresh) {
  const btn = $("#btn-refresh");
  btn.disabled = true;
  btn.textContent = refresh ? "Checking…" : btn.textContent;
  try {
    const sb = await api(`/api/scoreboard?refresh=${refresh}`);
    const s = sb.summary;
    $("#past-summary").innerHTML = s
      ? `Across ${s.n} graded show${s.n === 1 ? "" : "s"}, ${pct(s.model_hits)} of our picks were played. A plain play count got ${pct(s.baseline_hits)}. ` +
        `Our chances were ${s.model_brier < s.baseline_brier ? "closer to what happened" : "further from what happened"} than the play count's (Brier score ${s.model_brier.toFixed(3)} vs ${s.baseline_brier.toFixed(3)}, lower is better).`
      : "";
    $("#past-list").innerHTML = sb.predictions.length
      ? sb.predictions.map(pastItem).join("")
      : `<li class="past-item"><p>Nothing saved yet. Predict a show and tick "Save this prediction" before the band plays.</p></li>`;
  } catch {
    $("#past-list").innerHTML = `<li class="past-item"><p>Couldn't check setlist.fm for new setlists. Wait a minute and try again.</p></li>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Check for new setlists";
  }
}

function pastItem(p) {
  const r = p.result;
  const title = `${esc(p.artist)}${p.city ? ` in ${esc(p.city)}` : ""}, ${esc(fmtDate(p.show_date, { day: "numeric", month: "short", year: "numeric" }))}`;
  const actual = r ? new Set(r.actual) : null;
  const line = r
    ? `${r.setlist_hits} of ${r.setlist_size} predicted songs were played. <a href="${esc(r.url)}" target="_blank" rel="noopener">See the real setlist</a>.` +
      (r.surprises.length ? ` Surprises: ${r.surprises.map(esc).join(", ")}.` : "")
    : p.show_date > isoInDays(0) ? "Waiting for the show." : "Waiting for the setlist to be posted.";
  return `<li class="past-item"><h2>${title}</h2><p>${line}</p>
    <ul class="past-songs">${p.predicted.map((x) => `<li class="${actual ? (actual.has(x) ? "hit" : "miss") : ""}">${esc(x)}</li>`).join("")}</ul></li>`;
}

loadRecentArtists();
