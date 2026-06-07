// radar dashboard — interactive UI logic.
//
// Charts register into `charts` so hover/click hit-testing and the
// modal "enlarge" button share one implementation. `state` carries
// cross-render UI bits like the selected experiment id and pause flag.

const fmt = (x, d=4) => x === null || x === undefined ? '—'
  : (typeof x === 'number' ? x.toFixed(d) : x);
const fmtInt = x => x === null || x === undefined ? '—'
  : Number(x).toLocaleString();
const obj = (e, k) => (e.objectives && e.objectives[k] !== undefined) ? e.objectives[k] : null;
const esc = s => String(s == null ? '' : s).replace(/[&<>]/g,
  c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

async function get(p) { const r = await fetch(p); return r.json(); }

const TAB_KEY = 'radar.activeTab';
const VALID_TABS = ['architecture', 'data_pipeline', 'service_log', 'checkpoints'];

const state = {
  selectedId: null,
  detailOpen: false,
  lossLogScale: false,
  lastData: null,  // last refresh payload, used by modal/redraw
  activeTab: (() => {
    try {
      const stored = localStorage.getItem(TAB_KEY);
      return VALID_TABS.includes(stored) ? stored : 'architecture';
    } catch (_e) { return 'architecture'; }
  })(),
};

const charts = {};  // id -> { canvas, points, draw, lastArgs }

// Sort/filter state per table. Tables share data with the charts (raw
// rows are stashed here so re-sorting doesn't need a fetch).
const tableMeta = {};   // id -> { rowFn, rank }
const tableState = {};  // id -> { sortKey, sortDir, filter }
const tableRaw = {};    // id -> rows[]

// ── Tooltip ────────────────────────────────────────────────
const tip = () => document.getElementById('tooltip');
function showTip(html, ev) {
  const t = tip();
  t.innerHTML = html;
  t.style.display = 'block';
  const pad = 12;
  // Position near cursor but keep inside viewport.
  const w = t.offsetWidth, h = t.offsetHeight;
  let x = ev.clientX + pad, y = ev.clientY + pad;
  if (x + w > window.innerWidth) x = ev.clientX - w - pad;
  if (y + h > window.innerHeight) y = ev.clientY - h - pad;
  t.style.left = (x + window.scrollX) + 'px';
  t.style.top = (y + window.scrollY) + 'px';
}
function hideTip() { tip().style.display = 'none'; }

function expTooltip(e) {
  const rows = [
    ['id', e.id], ['round', e.round_id], ['miner', esc(e.miner_id)],
    ['name', esc(e.name)], ['metric', fmt(e.metric)],
    ['score', fmt(e.score, 3)],
  ];
  if (e.is_continuation) {
    rows.push(['kind', `<b style="color:#f0a040">continuation</b>`]);
    if (e.parent_index !== null && e.parent_index !== undefined)
      rows.push(['parent', e.parent_index]);
    if (e.n_rounds)              rows.push(['rounds', e.n_rounds]);
    if (e.cumulative_compute)    rows.push(['Σ compute', fmt(e.cumulative_compute, 2)]);
    if (e.delta !== undefined)   rows.push(['Δ', fmt(e.delta)]);
  } else {
    rows.push(['kind', `<b style="color:#7ec97e">novel</b>`]);
  }
  if (obj(e, 'aulc') !== null) rows.push(['aulc', fmt(obj(e, 'aulc'))]);
  if (obj(e, 'gift_metric') !== null)
    rows.push(['gift', fmt(obj(e, 'gift_metric'))]);
  if (obj(e, 'frozen_arch_version') !== null)
    rows.push(['arch v', `v${obj(e, 'frozen_arch_version')}`]);
  if (obj(e, 'crps') !== null) rows.push(['crps', fmt(obj(e, 'crps'))]);
  if (obj(e, 'mase') !== null) rows.push(['mase', fmt(obj(e, 'mase'))]);
  if (obj(e, 'flops_equivalent_size') !== null)
    rows.push(['flops', fmtInt(obj(e, 'flops_equivalent_size'))]);
  return rows.map(([k, v]) => `<span class="k">${k}</span> <b>${v}</b>`).join('<br>');
}

function kindBadge(e) {
  return e.is_continuation
    ? `<span class="badge badge-cont" title="warm-started from a parent checkpoint">cont</span>`
    : `<span class="badge badge-novel" title="trained from scratch">novel</span>`;
}

// Short task label so the leaderboard/recent tables show whether a row
// came from the real torch+GIFT-Eval forecasting challenge, the
// synthetic-data pipeline challenge, or the numpy regression task.
const TASK_LABELS = {
  ts_forecasting: { short: 'forecast', cls: 'badge-task-fc' },
  ts_data_pipeline: { short: 'pipeline', cls: 'badge-task-dp' },
  synth_regression: { short: 'synth', cls: 'badge-task-synth' },
};
function taskBadge(e) {
  const t = e.task || '';
  const meta = TASK_LABELS[t] || { short: t || '?', cls: 'badge-task-other' };
  return `<span class="badge ${meta.cls}" title="${esc(t || 'unknown task')}">${esc(meta.short)}</span>`;
}

// ── Tables ─────────────────────────────────────────────────
function row(e, rank) {
  const tr = document.createElement('tr');
  tr.className = 'row' + (e.id === state.selectedId ? ' selected' : '');
  tr.dataset.expId = e.id;
  tr.onclick = () => showDetail(e.id);
  tr.innerHTML = (rank !== undefined ? `<td class="rank">${rank}</td>` : '')
    + `<td>${e.id}</td><td>${e.round_id}</td>`
    + `<td>${esc(e.miner_id)}</td><td>${esc(e.name)}</td>`
    + `<td>${taskBadge(e)}</td>`
    + `<td>${kindBadge(e)}</td>`
    + `<td class="metric">${fmt(e.metric)}</td>`
    + `<td class="num">${fmt(obj(e, 'crps'))}</td>`
    + `<td class="num">${fmt(obj(e, 'mase'))}</td>`
    + `<td class="num">${fmtInt(obj(e, 'flops_equivalent_size'))}</td>`
    + `<td class="score">${fmt(e.score, 3)}</td>`;
  return tr;
}

function recentRow(e) {
  const tr = document.createElement('tr');
  tr.className = 'row' + (e.id === state.selectedId ? ' selected' : '');
  tr.dataset.expId = e.id;
  tr.onclick = () => showDetail(e.id);
  tr.innerHTML = `<td>${e.id}</td><td>${e.round_id}</td>`
    + `<td>${esc(e.miner_id)}</td><td>${esc(e.name)}</td>`
    + `<td>${taskBadge(e)}</td>`
    + `<td>${kindBadge(e)}</td>`
    + `<td class="metric">${fmt(e.metric)}</td>`
    + `<td class="score">${fmt(e.score, 3)}</td>`
    + (e.success ? `<td class="ok">ok</td>` : `<td class="fail">fail</td>`);
  return tr;
}

function contRow(e) {
  const tr = document.createElement('tr');
  tr.className = 'row' + (e.id === state.selectedId ? ' selected' : '');
  tr.dataset.expId = e.id;
  tr.onclick = () => showDetail(e.id);
  tr.innerHTML = `<td>${e.id}</td><td>${e.round_id}</td>`
    + `<td>${esc(e.miner_id)}</td>`
    + `<td>${e.parent_index ?? '—'}</td>`
    + `<td class="num">${fmtInt(e.n_rounds)}</td>`
    + `<td class="num">${fmt(e.cumulative_compute, 2)}</td>`
    + `<td class="num">${fmt(e.delta)}</td>`
    + `<td class="num">${fmt(e.metric)}</td>`
    + `<td class="score">${fmt(e.score, 3)}</td>`;
  return tr;
}

function frontRow(e) {
  const tr = document.createElement('tr');
  tr.className = 'row' + (e.id === state.selectedId ? ' selected' : '');
  tr.dataset.expId = e.id;
  tr.onclick = () => showDetail(e.id);
  tr.innerHTML = `<td>${e.id}</td><td>${esc(e.miner_id)}</td>`
    + `<td class="num">${fmtInt(obj(e, 'flops_equivalent_size'))}</td>`
    + `<td class="num">${fmt(e.metric)}</td>`
    + `<td class="num">${fmt(obj(e, 'crps'))}</td>`
    + `<td class="num">${fmt(obj(e, 'mase'))}</td>`;
  return tr;
}

function frontRowCM(e) {
  const tr = document.createElement('tr');
  tr.className = 'row' + (e.id === state.selectedId ? ' selected' : '');
  tr.dataset.expId = e.id;
  tr.onclick = () => showDetail(e.id);
  tr.innerHTML = `<td>${e.id}</td><td>${esc(e.miner_id)}</td><td>${esc(e.name)}</td>`
    + `<td class="num">${fmt(obj(e, 'crps'))}</td>`
    + `<td class="num">${fmt(obj(e, 'mase'))}</td>`
    + `<td class="num">${fmtInt(obj(e, 'flops_equivalent_size'))}</td>`;
  return tr;
}

function dpRow(e) {
  const tr = document.createElement('tr');
  tr.className = 'row' + (e.id === state.selectedId ? ' selected' : '');
  tr.dataset.expId = e.id;
  tr.onclick = () => showDetail(e.id);
  tr.innerHTML = `<td>${e.id}</td><td>${e.round_id}</td>`
    + `<td>${esc(e.miner_id)}</td><td>${esc(e.name)}</td>`
    + `<td class="num">${fmtInt(obj(e, 'frozen_arch_version'))}</td>`
    + `<td class="num">${fmt(obj(e, 'aulc'))}</td>`
    + `<td class="num">${fmt(obj(e, 'gift_metric'))}</td>`
    + `<td class="num">${fmt(obj(e, 'crps'))}</td>`
    + `<td class="num">${fmt(obj(e, 'mase'))}</td>`
    + `<td class="metric">${fmt(e.metric)}</td>`
    + `<td class="score">${fmt(e.score, 3)}</td>`;
  return tr;
}

function recentDPRow(e) {
  const tr = document.createElement('tr');
  tr.className = 'row' + (e.id === state.selectedId ? ' selected' : '');
  tr.dataset.expId = e.id;
  tr.onclick = () => showDetail(e.id);
  tr.innerHTML = `<td>${e.id}</td><td>${e.round_id}</td>`
    + `<td>${esc(e.miner_id)}</td><td>${esc(e.name)}</td>`
    + `<td class="num">${fmtInt(obj(e, 'frozen_arch_version'))}</td>`
    + `<td class="num">${fmt(obj(e, 'aulc'))}</td>`
    + `<td class="num">${fmt(obj(e, 'gift_metric'))}</td>`
    + `<td class="metric">${fmt(e.metric)}</td>`
    + (e.success ? `<td class="ok">ok</td>` : `<td class="fail">fail</td>`);
  return tr;
}

function frozenArchRow(e) {
  const tr = document.createElement('tr');
  tr.className = 'row';
  const ts = e.created_at
    ? new Date(e.created_at * 1000).toLocaleString() : '—';
  tr.innerHTML = `<td>v${e.version}</td>`
    + `<td>${fmtInt(e.source_experiment_id)}</td>`
    + `<td>${esc(e.source_name)}</td>`
    + `<td class="num">${fmt(e.source_metric)}</td>`
    + `<td class="num">${fmt(e.source_crps)}</td>`
    + `<td class="num">${fmt(e.source_mase)}</td>`
    + `<td class="num">${fmtInt(e.source_flops)}</td>`
    + `<td>${esc(ts)}</td>`;
  // Make source experiment clickable so it opens its detail panel.
  if (e.source_experiment_id) {
    tr.style.cursor = 'pointer';
    tr.onclick = () => showDetail(e.source_experiment_id);
  }
  return tr;
}

// ── Service-log + checkpoints ──────────────────────────────
function fmtBytes(n) {
  if (n === null || n === undefined) return '—';
  n = Number(n);
  if (!Number.isFinite(n)) return '—';
  if (n < 1024) return n + ' B';
  if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' KB';
  if (n < 1024 ** 3) return (n / 1024 / 1024).toFixed(1) + ' MB';
  return (n / 1024 ** 3).toFixed(2) + ' GB';
}
function fmtTs(ts) {
  if (!ts) return '—';
  const d = new Date(Number(ts) * 1000);
  if (Number.isNaN(d.getTime())) return '—';
  return d.toLocaleString();
}
function fmtRelTs(ts) {
  if (!ts) return '—';
  const dt = Date.now() / 1000 - Number(ts);
  if (dt < 0) return fmtTs(ts);
  if (dt < 60) return Math.round(dt) + 's ago';
  if (dt < 3600) return Math.round(dt / 60) + 'm ago';
  if (dt < 86400) return (dt / 3600).toFixed(1) + 'h ago';
  return (dt / 86400).toFixed(1) + 'd ago';
}

// Map event kinds to a small accent palette. Falls back to neutral grey.
const KIND_COLORS = {
  llm: '#6ba4ff', desearch: '#7ecbe8', wiki: '#c9c97e',
  arxiv: '#e87e9d', frontier: '#7ec97e', experiments: '#a07cc7',
  artifacts: '#f0a040', logs: '#9aa0aa',
};
function kindChipHtml(kind, n, active) {
  const c = KIND_COLORS[kind] || '#6c7280';
  return `<button class="chip" data-kind="${esc(kind)}"`
    + (active ? ` data-active="true"` : '')
    + ` style="--chip-accent:${c}">`
    + `<span class="chip-dot"></span>`
    + `<span class="chip-label">${esc(kind || '(blank)')}</span>`
    + `<span class="chip-count">${fmtInt(n)}</span></button>`;
}

const evState = {
  kind: '',
  round_id: '',
  miner_id: '',
  endpoint: '',
  errors: false,
  limit: 100,
  stats: null,
};
const ckState = { dir: '', items: [] };

function eventRow(e) {
  const tr = document.createElement('tr');
  tr.className = 'row';
  tr.dataset.eventId = e.id;
  tr.onclick = () => showEventDetail(e.id);
  const isErr = e.error || (e.status !== null && Number(e.status) >= 400);
  const statusCell = e.status === null || e.status === undefined
    ? `<td class="num muted">—</td>`
    : `<td class="num ${isErr ? 'fail' : 'ok'}">${e.status}</td>`;
  const kindColor = KIND_COLORS[e.kind] || '#6c7280';
  const preview = e.error
    ? `<span style="color:var(--red)">${esc(e.error)}</span>`
    : `<span class="muted">${esc(e.response_preview || e.request_preview || '')}</span>`;
  tr.innerHTML = `<td>${e.id}</td>`
    + `<td title="${esc(fmtTs(e.ts))}">${esc(fmtRelTs(e.ts))}</td>`
    + `<td>${e.round_id ?? '—'}</td>`
    + `<td>${esc(e.miner_id || '—')}</td>`
    + `<td><span class="kind-pill" style="--chip-accent:${kindColor}">`
    + `${esc(e.kind || '?')}</span></td>`
    + `<td class="ep" title="${esc(e.endpoint)}">${esc(e.endpoint)}</td>`
    + statusCell
    + `<td class="num">${fmt(e.latency_ms, 1)}</td>`
    + `<td class="num">${fmtBytes(e.response_bytes)}</td>`
    + `<td class="ep">${preview}</td>`;
  return tr;
}

function ckRow(c) {
  const tr = document.createElement('tr');
  tr.className = 'row';
  tr.dataset.expId = c.exp_id;
  tr.onclick = () => showCheckpointDetail(c.exp_id);
  const m = c.meta || {};
  tr.innerHTML = `<td>${c.exp_id}</td>`
    + `<td>${m.round_id ?? '—'}</td>`
    + `<td>${esc(m.miner_id || '—')}</td>`
    + `<td>${esc(m.name || '—')}</td>`
    + `<td>${esc(m.task || '—')}</td>`
    + `<td class="num">${fmtBytes(c.size_bytes)}</td>`
    + `<td class="num">${fmt(m.metric)}</td>`
    + `<td class="num">${fmtInt(m.n_rounds)}</td>`
    + `<td class="num">${m.parent_index ?? '—'}</td>`
    + `<td class="num">${fmt(m.cumulative_compute, 2)}</td>`
    + `<td title="${esc(fmtTs(c.mtime))}">${esc(fmtRelTs(c.mtime))}</td>`;
  return tr;
}

function getCkField(c, key) {
  if (!key) return null;
  if (key.startsWith('meta.')) {
    return c.meta ? (c.meta[key.slice(5)] ?? null) : null;
  }
  return c[key];
}

async function loadEvents() {
  const params = new URLSearchParams();
  if (evState.kind) params.set('kind', evState.kind);
  if (evState.round_id !== '') params.set('round_id', evState.round_id);
  if (evState.miner_id) params.set('miner_id', evState.miner_id);
  if (evState.endpoint) params.set('endpoint', evState.endpoint);
  if (evState.errors) params.set('errors', '1');
  params.set('limit', String(evState.limit));
  const events = await get('/api/events?' + params.toString());
  tableRaw.events = events;
  renderTable('events');
  document.getElementById('ev-shown').textContent =
    ` · showing ${events.length}`;
}

async function loadEventStats() {
  const stats = await get('/api/event_stats');
  evState.stats = stats;
  const cells = [
    ['total events', fmtInt(stats.total)],
    ['errors', fmtInt(stats.errors)],
    ['first', fmtRelTs(stats.first_ts)],
    ['last', fmtRelTs(stats.last_ts)],
  ];
  document.getElementById('evStats').innerHTML = cells.map(([k, v]) =>
    `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`
  ).join('');
  // Kind chips with counts. The empty-kind row is rare but possible.
  const chipsHost = document.getElementById('evKindChips');
  const chips = [
    `<button class="chip" data-kind=""`
      + (evState.kind === '' ? ` data-active="true"` : '')
      + ` style="--chip-accent:#9aa0aa">`
      + `<span class="chip-dot"></span>`
      + `<span class="chip-label">all</span>`
      + `<span class="chip-count">${fmtInt(stats.total)}</span></button>`,
  ];
  for (const k of (stats.by_kind || [])) {
    chips.push(kindChipHtml(k.kind, k.n, evState.kind === k.kind));
  }
  chipsHost.innerHTML = chips.join('');
  chipsHost.querySelectorAll('.chip').forEach(btn => {
    btn.onclick = () => {
      evState.kind = btn.dataset.kind || '';
      loadEventStats();
      loadEvents();
    };
  });
  // Miner datalist for the input.
  const ml = document.getElementById('evMinerList');
  ml.innerHTML = (stats.by_miner || []).map(m =>
    `<option value="${esc(m.miner_id || '')}">`).join('');
  document.getElementById('ev-count').textContent =
    ` (${fmtInt(stats.total)})`;
}

async function showEventDetail(id) {
  const e = await get(`/api/event/${id}`);
  if (!e || e.error === 'not found') return;
  const isErr = e.error || (e.status !== null && Number(e.status) >= 400);
  const reqStr = e.request === null || e.request === undefined ? ''
    : typeof e.request === 'string' ? e.request
    : JSON.stringify(e.request, null, 2);
  const resStr = e.response === null || e.response === undefined ? ''
    : typeof e.response === 'string' ? e.response
    : JSON.stringify(e.response, null, 2);
  openModal(`event ${e.id} · ${e.kind || '?'} ${e.endpoint || ''}`, body => {
    body.innerHTML = `<div class="event-meta">`
      + `<span><b>when</b> ${esc(fmtTs(e.ts))}</span>`
      + `<span><b>round</b> ${e.round_id ?? '—'}</span>`
      + `<span><b>miner</b> ${esc(e.miner_id || '—')}</span>`
      + `<span><b>status</b> <span class="${isErr ? 'fail' : 'ok'}">`
      + `${e.status ?? '—'}</span></span>`
      + `<span><b>latency</b> ${fmt(e.latency_ms, 1)} ms</span>`
      + `<span><b>req</b> ${fmtBytes(e.request_bytes)}</span>`
      + `<span><b>res</b> ${fmtBytes(e.response_bytes)}</span>`
      + `</div>`
      + (e.error
          ? `<h3>error</h3><pre class="err">${esc(e.error)}</pre>` : '')
      + `<div class="json-grid">`
      + `<div><h3>request <button class="copy-btn" data-copy="req">copy</button></h3>`
      + `<pre class="json-pre" id="evReq">${esc(reqStr) || '<span class="muted">(empty)</span>'}</pre></div>`
      + `<div><h3>response <button class="copy-btn" data-copy="res">copy</button></h3>`
      + `<pre class="json-pre" id="evRes">${esc(resStr) || '<span class="muted">(empty)</span>'}</pre></div>`
      + `</div>`;
    body.querySelectorAll('button[data-copy]').forEach(btn => {
      btn.onclick = () => {
        const s = btn.dataset.copy === 'req' ? reqStr : resStr;
        navigator.clipboard.writeText(s).then(() => {
          btn.textContent = 'copied';
          setTimeout(() => btn.textContent = 'copy', 1200);
        });
      };
    });
  });
}

async function loadCheckpoints() {
  const data = await get('/api/checkpoints');
  ckState.dir = data.dir || '';
  ckState.items = data.items || [];
  const dirLine = data.exists
    ? `${esc(data.dir)} · ${ckState.items.length} file(s)`
    : `<span class="fail">checkpoint dir missing</span>: ${esc(data.dir)}`;
  document.getElementById('ckDir').innerHTML = dirLine;
  document.getElementById('ck-count').textContent =
    ` (${ckState.items.length})`;
  tableRaw.checkpoints = ckState.items;
  renderTable('checkpoints');
}

async function showCheckpointDetail(expId) {
  const sig = await get(`/api/checkpoint/${expId}/signature`);
  if (!sig || sig.error) return;
  const top = sig.tensors.slice(0, 200);
  openModal(`checkpoint ${sig.exp_id} · signature`, body => {
    body.innerHTML = `<div class="event-meta">`
      + `<span><b>path</b> <code>${esc(sig.path)}</code></span>`
      + `<span><b>size</b> ${fmtBytes(sig.size_bytes)}</span>`
      + `<span><b>tensors</b> ${fmtInt(sig.tensors.length)}</span>`
      + `<span><b>parameters</b> ${fmtInt(sig.total_params)}</span>`
      + `<span><a href="#" id="ckOpenExp">→ open experiment ${sig.exp_id}</a></span>`
      + `</div>`
      + `<h3>tensors <span class="muted">(top by parameter count)</span></h3>`
      + `<div class="table-wrap" style="max-height:70vh">`
      + `<table><thead><tr>`
      + `<th>name</th><th class="num">shape</th><th class="num">params</th>`
      + `</tr></thead><tbody>`
      + top.map(t =>
          `<tr><td><code>${esc(t.name)}</code></td>`
          + `<td class="num">${esc(t.shape.join(' × '))}</td>`
          + `<td class="num">${fmtInt(t.n_params)}</td></tr>`
        ).join('')
      + `</tbody></table></div>`
      + (sig.tensors.length > top.length
          ? `<div class="muted">… ${sig.tensors.length - top.length} more not shown</div>`
          : '');
    const link = body.querySelector('#ckOpenExp');
    if (link) link.onclick = (ev) => {
      ev.preventDefault();
      closeModal();
      showDetail(sig.exp_id);
    };
  });
}

// ── Sort + filter ──────────────────────────────────────────
function getField(e, key) {
  if (!key) return null;
  if (key.startsWith('objectives.')) return obj(e, key.slice(11));
  if (key.startsWith('meta.')) {
    return e.meta ? (e.meta[key.slice(5)] ?? null) : null;
  }
  return e[key];
}
function cmpVals(a, b, type) {
  const aNull = a === null || a === undefined || (typeof a === 'number' && !Number.isFinite(a));
  const bNull = b === null || b === undefined || (typeof b === 'number' && !Number.isFinite(b));
  if (aNull && bNull) return 0;
  if (aNull) return 1;   // nulls sink to the bottom regardless of dir
  if (bNull) return -1;
  if (type === 'num') return Number(a) - Number(b);
  return String(a).localeCompare(String(b));
}
function matchesFilter(e, q) {
  if (!q) return true;
  q = q.toLowerCase();
  const hay = [e.id, e.round_id, e.miner_id, e.name, e.task, e.metric, e.score];
  if (hay.some(v => v !== null && v !== undefined
      && String(v).toLowerCase().includes(q))) return true;
  if (e.objectives) {
    for (const v of Object.values(e.objectives)) {
      if (v !== null && v !== undefined
          && String(v).toLowerCase().includes(q)) return true;
    }
  }
  return false;
}
function renderTable(tableId) {
  const meta = tableMeta[tableId];
  const st = tableState[tableId];
  const data = tableRaw[tableId] || [];
  let rows = data.filter(e => matchesFilter(e, st.filter));
  if (st.sortKey) {
    const th = document.querySelector(
      `table[data-table="${tableId}"] th[data-key="${st.sortKey}"]`);
    const type = th ? (th.dataset.type || '') : '';
    const sign = st.sortDir === 'asc' ? 1 : -1;
    rows = rows.slice().sort((a, b) => sign *
      cmpVals(getField(a, st.sortKey), getField(b, st.sortKey), type));
  }
  const el = document.getElementById(tableId);
  el.innerHTML = '';
  rows.forEach((e, i) =>
    el.appendChild(meta.rank ? meta.rowFn(e, i + 1) : meta.rowFn(e)));
  document.querySelectorAll(
      `table[data-table="${tableId}"] th[data-key]`).forEach(th => {
    th.classList.remove('sort-asc', 'sort-desc');
    if (th.dataset.key === st.sortKey)
      th.classList.add(st.sortDir === 'asc' ? 'sort-asc' : 'sort-desc');
  });
}
function setupTables() {
  document.querySelectorAll('table[data-table]').forEach(t => {
    const tableId = t.dataset.table;
    t.querySelectorAll('th[data-key]').forEach(th => {
      th.classList.add('sortable');
      th.addEventListener('click', () => {
        const st = tableState[tableId];
        if (st.sortKey === th.dataset.key) {
          st.sortDir = st.sortDir === 'asc' ? 'desc' : 'asc';
        } else {
          st.sortKey = th.dataset.key;
          st.sortDir = 'asc';
        }
        renderTable(tableId);
      });
    });
  });
  document.querySelectorAll('.tbl-search').forEach(inp => {
    const tableId = inp.dataset.table;
    inp.addEventListener('input', () => {
      tableState[tableId].filter = inp.value;
      renderTable(tableId);
    });
  });
}

// Marker helpers — keep continuation runs visually separate from
// novel ones across every chart. Circle = novel, triangle = continuation.
function drawMarker(ctx, x, y, r, isCont) {
  ctx.beginPath();
  if (isCont) {
    ctx.moveTo(x, y - r);
    ctx.lineTo(x + r * 0.95, y + r * 0.75);
    ctx.lineTo(x - r * 0.95, y + r * 0.75);
    ctx.closePath();
  } else {
    ctx.arc(x, y, r, 0, 2*Math.PI);
  }
  ctx.fill();
}

// ── Pareto: metric × flops (log x) ─────────────────────────
function drawPareto(canvas, front, all) {
  const c = canvas, ctx = c.getContext('2d');
  ctx.fillStyle = '#161820'; ctx.fillRect(0, 0, c.width, c.height);
  const points = all.filter(e => e.success && e.metric !== null);
  const hits = [];
  if (points.length === 0) {
    drawEmpty(ctx, c, 'no successful experiments yet');
    return { points: hits };
  }
  const xs = points.map(e => Math.max(1, obj(e, 'flops_equivalent_size') || 1));
  const ys = points.map(e => e.metric);
  const xMin = Math.log10(Math.min(...xs));
  const xMax = Math.log10(Math.max(...xs) + 1);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const pad = PAD_L;
  const xRange = Math.max(1e-6, xMax - xMin), yRange = Math.max(1e-6, yMax - yMin);
  const px = x => PAD_L + (Math.log10(Math.max(1, x)) - xMin) / xRange * (c.width - PAD_L - PAD_R);
  const py = y => c.height - PAD_B - (y - yMin) / yRange * (c.height - PAD_T - PAD_B);
  drawGrid(ctx, c, pad);
  // all points
  for (const e of points) {
    const x = px(obj(e, 'flops_equivalent_size') || 1), y = py(e.metric);
    const selected = e.id === state.selectedId;
    const cont = !!e.is_continuation;
    ctx.fillStyle = selected ? '#f0a040' : (cont ? '#a07cc7' : '#3a4050');
    drawMarker(ctx, x, y, selected ? 5 : (cont ? 4 : 3), cont);
    hits.push({ sx: x, sy: y, exp: e });
  }
  // frontier line + points
  ctx.strokeStyle = '#7ec97e'; ctx.fillStyle = '#7ec97e'; ctx.lineWidth = 1.5;
  ctx.beginPath();
  front.forEach((e, i) => {
    const x = px(obj(e, 'flops_equivalent_size') || 1), y = py(e.metric);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  for (const e of front) {
    const x = px(obj(e, 'flops_equivalent_size') || 1), y = py(e.metric);
    drawMarker(ctx, x, y, 4, !!e.is_continuation);
  }
  drawAxesLabels(ctx, c, 'log10(flops) →', 'metric (lower=better) →');
  drawYTicks(ctx, c, pad, yMin, yMax, 4);
  return { points: hits };
}

// ── Pareto: crps × mase ────────────────────────────────────
function drawParetoCM(canvas, front, all) {
  const c = canvas, ctx = c.getContext('2d');
  ctx.fillStyle = '#161820'; ctx.fillRect(0, 0, c.width, c.height);
  const points = all.filter(e =>
    e.success && obj(e, 'crps') !== null && obj(e, 'mase') !== null);
  const hits = [];
  if (points.length === 0) {
    drawEmpty(ctx, c, 'no experiments with both crps and mase yet');
    return { points: hits };
  }
  const xs = points.map(e => obj(e, 'crps'));
  const ys = points.map(e => obj(e, 'mase'));
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const pad = PAD_L;
  const xRange = Math.max(1e-6, xMax - xMin), yRange = Math.max(1e-6, yMax - yMin);
  const px = x => PAD_L + (x - xMin) / xRange * (c.width - PAD_L - PAD_R);
  const py = y => c.height - PAD_B - (y - yMin) / yRange * (c.height - PAD_T - PAD_B);
  drawGrid(ctx, c, pad);
  for (const e of points) {
    const x = px(obj(e, 'crps')), y = py(obj(e, 'mase'));
    const selected = e.id === state.selectedId;
    const cont = !!e.is_continuation;
    ctx.fillStyle = selected ? '#f0a040' : (cont ? '#a07cc7' : '#3a4050');
    drawMarker(ctx, x, y, selected ? 5 : (cont ? 4 : 3), cont);
    hits.push({ sx: x, sy: y, exp: e });
  }
  ctx.strokeStyle = '#7ec97e'; ctx.fillStyle = '#7ec97e'; ctx.lineWidth = 1.5;
  ctx.beginPath();
  front.forEach((e, i) => {
    const x = px(obj(e, 'crps')), y = py(obj(e, 'mase'));
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  for (const e of front) {
    const x = px(obj(e, 'crps')), y = py(obj(e, 'mase'));
    drawMarker(ctx, x, y, 4, !!e.is_continuation);
  }
  drawAxesLabels(ctx, c, 'crps (lower=better) →', 'mase (lower=better) →');
  drawYTicks(ctx, c, pad, yMin, yMax, 4);
  return { points: hits };
}

// ── Pareto: aulc × gift (data-pipeline only) ───────────────
// Both axes lower=better. Points color-coded by frozen_arch_version so
// cross-version comparisons are visible at a glance.
const ARCH_PALETTE = [
  '#6ba4ff', '#f0a040', '#7ec97e', '#a07cc7',
  '#e87e9d', '#7ecbe8', '#c97e7e', '#c9c97e',
];
function archColor(v) {
  if (!v) return '#3a4050';
  return ARCH_PALETTE[(Number(v) - 1) % ARCH_PALETTE.length];
}
function drawParetoDP(canvas, front, all) {
  const c = canvas, ctx = c.getContext('2d');
  ctx.fillStyle = '#161820'; ctx.fillRect(0, 0, c.width, c.height);
  const points = (all || []).filter(e =>
    e.success && obj(e, 'aulc') !== null && obj(e, 'gift_metric') !== null);
  const hits = [];
  if (points.length === 0) {
    drawEmpty(ctx, c, 'no ts_data_pipeline runs with aulc + gift_metric yet');
    return { points: hits };
  }
  const xs = points.map(e => obj(e, 'aulc'));
  const ys = points.map(e => obj(e, 'gift_metric'));
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const pad = PAD_L;
  const xRange = Math.max(1e-9, xMax - xMin);
  const yRange = Math.max(1e-9, yMax - yMin);
  const px = x => PAD_L + (x - xMin) / xRange * (c.width - PAD_L - PAD_R);
  const py = y => c.height - PAD_B - (y - yMin) / yRange * (c.height - PAD_T - PAD_B);
  drawGrid(ctx, c, pad);
  // Off-frontier points, colored by frozen_arch_version.
  for (const e of points) {
    const x = px(obj(e, 'aulc')), y = py(obj(e, 'gift_metric'));
    const selected = e.id === state.selectedId;
    ctx.fillStyle = selected ? '#f0a040'
      : archColor(obj(e, 'frozen_arch_version'));
    drawMarker(ctx, x, y, selected ? 5 : 3, false);
    hits.push({ sx: x, sy: y, exp: e });
  }
  // Frontier line + markers.
  ctx.strokeStyle = '#7ec97e'; ctx.fillStyle = '#7ec97e'; ctx.lineWidth = 1.5;
  ctx.beginPath();
  (front || []).forEach((e, i) => {
    const x = px(obj(e, 'aulc')), y = py(obj(e, 'gift_metric'));
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  for (const e of (front || [])) {
    const x = px(obj(e, 'aulc')), y = py(obj(e, 'gift_metric'));
    drawMarker(ctx, x, y, 4, false);
  }
  // Per-version legend so the colour code is readable.
  const versions = [...new Set(points
    .map(e => obj(e, 'frozen_arch_version'))
    .filter(v => v))].sort((a, b) => Number(a) - Number(b));
  ctx.font = '11px ui-monospace, monospace';
  versions.slice(0, 6).forEach((v, i) => {
    ctx.fillStyle = archColor(v);
    ctx.fillText(`■ v${v}`, c.width - 70 - i * 60, 16);
  });
  drawAxesLabels(
    ctx, c,
    'aulc (lower=better) →',
    'sqrt(crps·mase) (lower=better) →',
  );
  drawYTicks(ctx, c, pad, yMin, yMax, 4);
  return { points: hits };
}

// ── Pareto: cumulative_compute × Δ (continuation only) ────
// X is cumulative compute (lower=better → left), Y is Δ (higher=better →
// up), so the frontier hugs the top-left and the line slopes downward.
function drawParetoCont(canvas, front, all) {
  const c = canvas, ctx = c.getContext('2d');
  ctx.fillStyle = '#161820'; ctx.fillRect(0, 0, c.width, c.height);
  const hits = [];
  if (!all || all.length === 0) {
    drawEmpty(ctx, c, 'no continuation runs yet');
    return { points: hits };
  }
  const xs = all.map(e => Math.max(1e-9, e.cumulative_compute || 1e-9));
  const ys = all.map(e => e.delta);
  const useLogX = Math.max(...xs) / Math.min(...xs) > 50;
  const xT = v => useLogX ? Math.log10(Math.max(1e-9, v)) : v;
  const xMin = Math.min(...xs.map(xT));
  const xMax = Math.max(...xs.map(xT));
  const yMin = Math.min(...ys, 0);  // anchor at 0 so the "no progress" line shows
  const yMax = Math.max(...ys, 0);
  const pad = PAD_L;
  const xRange = Math.max(1e-6, xMax - xMin), yRange = Math.max(1e-6, yMax - yMin);
  const px = x => PAD_L + (xT(x) - xMin) / xRange * (c.width - PAD_L - PAD_R);
  const py = y => c.height - PAD_B - (y - yMin) / yRange * (c.height - PAD_T - PAD_B);
  drawGrid(ctx, c, pad);
  // zero-Δ reference line: above it = improved on parent.
  if (yMin < 0 && yMax > 0) {
    ctx.strokeStyle = '#3a4050'; ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
    ctx.beginPath();
    const yz = py(0);
    ctx.moveTo(PAD_L, yz); ctx.lineTo(c.width - PAD_R, yz);
    ctx.stroke(); ctx.setLineDash([]);
  }
  // all continuation points — color-code by Δ sign so regressions read red.
  for (const e of all) {
    const x = px(e.cumulative_compute), y = py(e.delta);
    const selected = e.id === state.selectedId;
    ctx.fillStyle = selected
      ? '#f0a040'
      : (e.delta > 0 ? '#a07cc7' : '#6b4040');
    drawMarker(ctx, x, y, selected ? 5 : 4, true);
    hits.push({ sx: x, sy: y, exp: e });
  }
  // frontier line + markers
  ctx.strokeStyle = '#7ec97e'; ctx.fillStyle = '#7ec97e'; ctx.lineWidth = 1.5;
  ctx.beginPath();
  front.forEach((e, i) => {
    const x = px(e.cumulative_compute), y = py(e.delta);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  for (const e of front) {
    const x = px(e.cumulative_compute), y = py(e.delta);
    drawMarker(ctx, x, y, 5, true);
  }
  drawAxesLabels(
    ctx, c,
    (useLogX ? 'log10(Σ compute) →' : 'Σ compute →'),
    'Δ = parent.metric − this.metric (↑ = better) →',
  );
  drawYTicks(ctx, c, pad, yMin, yMax, 4);
  return { points: hits };
}

// ── Shared chart helpers ───────────────────────────────────
// All charts share the same plot-region geometry: `pad` is the inset
// from the canvas edges to the plot area. Bottom/left get extra room
// for axis labels; top/right stay snug. Keep in sync with drawGrid,
// drawAxesLabels and per-chart px/py.
const PAD_L = 58, PAD_R = 24, PAD_T = 18, PAD_B = 46;
function drawGrid(ctx, c, pad) {
  // `pad` arg kept for backwards compat — left/bottom use the shared
  // PAD_L/PAD_B so axis-label spacing is consistent across charts.
  ctx.strokeStyle = '#2a2e3a'; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(PAD_L, PAD_T); ctx.lineTo(PAD_L, c.height - PAD_B);
  ctx.lineTo(c.width - PAD_R, c.height - PAD_B);
  ctx.stroke();
}
function drawYTicks(ctx, c, pad, yMin, yMax, n) {
  ctx.fillStyle = '#9aa0aa'; ctx.font = '11px ui-monospace, monospace';
  ctx.textBaseline = 'middle';
  const range = Math.max(1e-9, yMax - yMin);
  for (let i = 0; i <= n; i++) {
    const yv = yMin + (range * i / n);
    const yy = c.height - PAD_B - (yv - yMin) / range * (c.height - PAD_T - PAD_B);
    ctx.strokeStyle = '#202430';
    ctx.beginPath(); ctx.moveTo(PAD_L, yy); ctx.lineTo(c.width - PAD_R, yy); ctx.stroke();
    ctx.fillStyle = '#9aa0aa';
    const txt = yv.toFixed(yv >= 100 ? 0 : yv >= 1 ? 2 : 3);
    const tw = ctx.measureText(txt).width;
    ctx.fillText(txt, PAD_L - 8 - tw, yy);
  }
  ctx.textBaseline = 'alphabetic';
}
function drawAxesLabels(ctx, c, xLabel, yLabel) {
  ctx.fillStyle = '#d5d7dc';
  ctx.font = '600 12px -apple-system, BlinkMacSystemFont, "Inter", system-ui, sans-serif';
  // X-axis label: centered along the bottom of the plot region.
  const plotW = c.width - PAD_L - PAD_R;
  const xw = ctx.measureText(xLabel).width;
  ctx.fillText(xLabel, PAD_L + (plotW - xw) / 2, c.height - 10);
  // Y-axis label: rotated, centered vertically along the left edge.
  ctx.save();
  ctx.translate(16, PAD_T + (c.height - PAD_T - PAD_B) / 2);
  ctx.rotate(-Math.PI / 2);
  const yw = ctx.measureText(yLabel).width;
  ctx.fillText(yLabel, -yw / 2, 0);
  ctx.restore();
}
function drawEmpty(ctx, c, msg) {
  ctx.fillStyle = '#9aa0aa'; ctx.font = '12px ui-monospace, monospace';
  ctx.textAlign = 'center';
  ctx.fillText(msg, c.width / 2, c.height / 2);
  ctx.textAlign = 'left';
}

// ── Chart hover/click registration ─────────────────────────
function registerChart(id, drawFn, getArgs) {
  const canvas = document.getElementById(id);
  if (!canvas) return;
  charts[id] = { canvas, draw: drawFn, getArgs, points: [] };
  canvas.style.cursor = 'default';
  canvas.addEventListener('mousemove', ev => {
    const ch = charts[id];
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const mx = (ev.clientX - rect.left) * scaleX;
    const my = (ev.clientY - rect.top) * scaleY;
    const hit = nearest(ch.points, mx, my, 14);
    if (hit) {
      showTip(expTooltip(hit.exp), ev);
      canvas.style.cursor = 'pointer';
    } else {
      hideTip();
      canvas.style.cursor = 'default';
    }
  });
  canvas.addEventListener('mouseleave', hideTip);
  canvas.addEventListener('click', ev => {
    const ch = charts[id];
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const mx = (ev.clientX - rect.left) * scaleX;
    const my = (ev.clientY - rect.top) * scaleY;
    const hit = nearest(ch.points, mx, my, 14);
    if (hit) showDetail(hit.exp.id);
  });
}
function nearest(points, x, y, maxDist) {
  let best = null, bestD = maxDist * maxDist;
  for (const p of points) {
    const dx = p.sx - x, dy = p.sy - y;
    const d = dx*dx + dy*dy;
    if (d < bestD) { bestD = d; best = p; }
  }
  return best;
}
function renderChart(id) {
  const ch = charts[id];
  if (!ch) return;
  const args = ch.getArgs();
  if (!args) return;
  const r = ch.draw(ch.canvas, ...args);
  ch.points = (r && r.points) || [];
  ch.lastArgs = args;
}

// ── Detail panel ───────────────────────────────────────────
async function showDetail(id) {
  const e = await get(`/api/experiment/${id}`);
  state.selectedId = id;
  state.detailOpen = true;
  state.lossLogScale = false;
  const d = document.getElementById('detail');
  d.className = 'open';
  const objJson = esc(JSON.stringify(e.objectives, null, 2));
  const ts = e.timestamp ? new Date(e.timestamp * 1000).toLocaleString() : '—';
  const hasLoss = Array.isArray(e.loss_curve) && e.loss_curve.length > 0;
  const spikes = obj(e, 'num_spikes_skipped');
  const contLine = e.is_continuation
    ? `<p><b>kind</b> <span style="color:#a07cc7">continuation</span>`
      + (e.parent_index != null ? ` <b>parent</b> `
          + `<a href="#" data-parent="${e.parent_index}" id="parentLink">${e.parent_index}</a>`
        : '')
      + ` <b>rounds</b> ${e.n_rounds || 1}`
      + ` <b>Σ compute</b> ${fmt(e.cumulative_compute, 2)}`
      + ` <b>mode</b> ${esc(e.mode || 'continue')}</p>`
    : `<p><b>kind</b> <span style="color:#7ec97e">novel</span></p>`;
  d.innerHTML = `<span class="close" id="detailClose">×</span>`
    + `<h2>experiment ${e.id}</h2>`
    + `<div class="muted">round ${e.round_id} · miner ${esc(e.miner_id)} · `
    + `task ${esc(e.task) || '?'} · gen ${e.generation} · ${ts}</div>`
    + `<p><b>name</b> ${esc(e.name)}<br><b>metric</b> ${fmt(e.metric)} `
    + ` <b>score</b> ${fmt(e.score, 3)} <b>success</b> ${e.success}`
    + (spikes
        ? ` · <span style="color:#e8b87e">spikes skipped: ${spikes}</span>`
        : '')
    + `</p>`
    + contLine
    + `<h2>objectives</h2><pre>${objJson}</pre>`
    + `<h2>analysis</h2><pre>${esc(e.analysis)}</pre>`
    + (e.motivation ? `<h2>motivation</h2><pre>${esc(e.motivation)}</pre>` : '')
    + (e.reasoning ? `<h2>reasoning</h2><pre>${esc(e.reasoning)}</pre>` : '')
    + `<h2>loss curve`
    + (hasLoss
        ? ` <button class="chart-expand" id="lossExpand" title="enlarge">⤢</button></h2>`
          + `<div class="loss-controls">`
          + `<button id="lossLin" class="active">linear</button>`
          + `<button id="lossLog">log y</button>`
          + `</div>`
          + `<canvas id="lossCurve" width="520" height="240"></canvas>`
          + `<div class="muted" id="lossMeta"></div>`
        : `</h2><div class="muted">no loss curve recorded</div>`)
    + `<h2>code <button class="copy-btn" id="copyCode">copy</button></h2>`
    + `<pre id="codePre">${esc(e.code)}</pre>`;
  document.getElementById('detailClose').onclick = closeDetail;
  const pl = document.getElementById('parentLink');
  if (pl) pl.onclick = (ev) => {
    ev.preventDefault();
    showDetail(Number(pl.dataset.parent));
  };
  const copy = document.getElementById('copyCode');
  if (copy) copy.onclick = () => {
    navigator.clipboard.writeText(e.code).then(
      () => { copy.textContent = 'copied'; setTimeout(() => copy.textContent = 'copy', 1200); }
    );
  };
  if (hasLoss) {
    const drawLoss = () => renderLossCurve(
      document.getElementById('lossCurve'), e.loss_curve, e.val_curve, state.lossLogScale,
      document.getElementById('lossMeta'),
    );
    drawLoss();
    document.getElementById('lossLin').onclick = () => {
      state.lossLogScale = false;
      document.getElementById('lossLin').classList.add('active');
      document.getElementById('lossLog').classList.remove('active');
      drawLoss();
    };
    document.getElementById('lossLog').onclick = () => {
      state.lossLogScale = true;
      document.getElementById('lossLog').classList.add('active');
      document.getElementById('lossLin').classList.remove('active');
      drawLoss();
    };
    document.getElementById('lossExpand').onclick = () => openLossModal(e);
  }
  highlightSelection();
}
function closeDetail() {
  document.getElementById('detail').className = '';
  state.detailOpen = false;
  state.selectedId = null;
  highlightSelection();
  if (state.lastData) redraw(state.lastData);
}
function highlightSelection() {
  document.querySelectorAll('tr.row').forEach(tr => {
    if (Number(tr.dataset.expId) === state.selectedId) tr.classList.add('selected');
    else tr.classList.remove('selected');
  });
}

// ── Loss curve ─────────────────────────────────────────────
function normalizeCurve(curve) {
  if (!Array.isArray(curve)) return [];
  return curve.map((v, i) => {
    if (Array.isArray(v)) return { x: v[0], y: v[1] };
    if (v && typeof v === 'object')
      return { x: v.step ?? v.x ?? i, y: v.loss ?? v.y };
    return { x: i, y: v };
  }).filter(p => Number.isFinite(p.x) && Number.isFinite(p.y));
}

let lossState = null;  // { canvas, pts, valPts, px, py, pad, xMin, xMax, yMin, yMax }

function renderLossCurve(canvas, curve, valCurve, logY, metaEl) {
  if (!canvas) return;
  const c = canvas, ctx = c.getContext('2d');
  ctx.fillStyle = '#161820'; ctx.fillRect(0, 0, c.width, c.height);
  const pts = normalizeCurve(curve);
  const valPts = normalizeCurve(valCurve);
  if (pts.length === 0) { drawEmpty(ctx, c, 'no finite points'); return; }
  const trainStepless = pts.every((p, i) => p.x === i);
  if (trainStepless && valPts.length > 0) {
    const valMax = Math.max(...valPts.map(p => p.x));
    const n = pts.length;
    if (n > 1 && valMax > 0) {
      for (let i = 0; i < n; i++) pts[i].x = (i / (n - 1)) * valMax;
    }
  }
  // Log y needs positive losses; drop non-positive and fall back to linear
  // if nothing survives. Keep the curves in sync (skip same indices).
  let effLog = logY;
  if (effLog) {
    const minY = Math.min(...pts.map(p => p.y).concat(valPts.map(p => p.y)));
    if (!(minY > 0)) effLog = false;
  }
  const yT = v => effLog ? Math.log10(v) : v;
  const pad = 40;
  const allPts = pts.concat(valPts);
  const xs = allPts.map(p => p.x), ys = allPts.map(p => yT(p.y));
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMinAll = Math.min(...ys), yMaxAll = Math.max(...ys);
  // Robust y-axis: a single loss spike (e.g. 141k vs ~1.0 nominal) squashes
  // the rest of the curve into a flat line. Cap the visible upper bound at
  // median + 5*(p95 - median) and clamp out-of-range points to the top edge
  // so the spike still shows as a vertical excursion. Disabled on log y
  // since the log already compresses outliers.
  let yMaxDisp = yMaxAll;
  let clipped = 0;
  if (!effLog && ys.length >= 4) {
    const sorted = [...ys].sort((a, b) => a - b);
    const quantile = q => {
      const i = (sorted.length - 1) * q;
      const lo = Math.floor(i), hi = Math.ceil(i);
      return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
    };
    const med = quantile(0.5);
    const p95 = quantile(0.95);
    const cap = med + 5 * Math.max(0, p95 - med);
    if (cap > yMinAll && cap < yMaxAll) {
      yMaxDisp = cap;
      clipped = ys.filter(y => y > cap).length;
    }
  }
  const yMin = yMinAll, yMax = yMaxDisp;
  const xRange = Math.max(1e-9, xMax - xMin);
  const yRange = Math.max(1e-9, yMax - yMin);
  const px = x => pad + (x - xMin) / xRange * (c.width - 2 * pad);
  const py = y => {
    const t = yT(y);
    const clamped = t > yMax ? yMax : t;
    return c.height - pad - (clamped - yMin) / yRange * (c.height - 2 * pad);
  };
  drawGrid(ctx, c, pad);
  // y ticks (in original units, even on log scale)
  ctx.fillStyle = '#777e8b'; ctx.font = '10px ui-monospace, monospace';
  for (let i = 0; i <= 4; i++) {
    const yv = effLog
      ? Math.pow(10, yMin + (yRange * i / 4))
      : yMin + (yRange * i / 4);
    const yy = py(yv);
    ctx.strokeStyle = '#1c1f28';
    ctx.beginPath(); ctx.moveTo(pad, yy); ctx.lineTo(c.width - pad/2, yy); ctx.stroke();
    ctx.fillStyle = '#777e8b';
    ctx.fillText(yv.toFixed(yv >= 100 ? 0 : yv >= 1 ? 2 : 4), 2, yy + 3);
  }
  // train line
  ctx.strokeStyle = '#7ec97e'; ctx.lineWidth = 1.5;
  ctx.beginPath();
  pts.forEach((p, i) => {
    const x = px(p.x), y = py(p.y);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  if (pts.length <= 80) {
    ctx.fillStyle = '#7ec97e';
    for (const p of pts) {
      ctx.beginPath(); ctx.arc(px(p.x), py(p.y), 2, 0, 2*Math.PI); ctx.fill();
    }
  }
  if (valPts.length > 0) {
    ctx.strokeStyle = '#f0a040'; ctx.lineWidth = 1.5;
    ctx.beginPath();
    valPts.forEach((p, i) => {
      const x = px(p.x), y = py(p.y);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = '#f0a040';
    for (const p of valPts) {
      ctx.beginPath(); ctx.arc(px(p.x), py(p.y), 2.5, 0, 2*Math.PI); ctx.fill();
    }
  }
  // legend
  ctx.font = '11px ui-monospace, monospace';
  ctx.fillStyle = '#7ec97e'; ctx.fillText('■ train', c.width - 150, 16);
  if (valPts.length > 0) {
    ctx.fillStyle = '#f0a040'; ctx.fillText('■ val', c.width - 80, 16);
  }
  ctx.fillStyle = '#777e8b';
  ctx.fillText('step →', c.width - 60, c.height - 6);
  ctx.save(); ctx.translate(10, 60); ctx.rotate(-Math.PI/2);
  ctx.fillText(effLog ? 'log loss →' : 'loss →', 0, 0); ctx.restore();
  if (metaEl) {
    const trainYs = pts.map(p => p.y);
    let text = `train: ${pts.length} pts · first=${trainYs[0].toFixed(4)} `
      + `· last=${trainYs[trainYs.length-1].toFixed(4)} `
      + `· min=${Math.min(...trainYs).toFixed(4)}`;
    if (valPts.length > 0) {
      const valYs = valPts.map(p => p.y);
      text += ` · val: ${valPts.length} pts · last=${valYs[valYs.length-1].toFixed(4)} `
        + `· best=${Math.min(...valYs).toFixed(4)}`;
    }
    if (clipped > 0) text += ` · ${clipped} spike(s) clipped from view`;
    metaEl.textContent = text;
  }
  // Save state for hover hit-testing on this canvas.
  lossState = { canvas: c, pts, valPts, px, py, xMin, xMax };
  attachLossHover(c);
}

function attachLossHover(canvas) {
  if (canvas._lossBound) return;
  canvas._lossBound = true;
  canvas.addEventListener('mousemove', ev => {
    if (!lossState || lossState.canvas !== canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const mx = (ev.clientX - rect.left) * scaleX;
    // x in data units
    const xData = lossState.xMin + (mx - 40) / (canvas.width - 80)
      * (lossState.xMax - lossState.xMin);
    const near = (arr) => {
      if (arr.length === 0) return null;
      let best = arr[0], bd = Math.abs(arr[0].x - xData);
      for (const p of arr) {
        const d = Math.abs(p.x - xData);
        if (d < bd) { bd = d; best = p; }
      }
      return best;
    };
    const t = near(lossState.pts), v = near(lossState.valPts);
    if (!t && !v) return hideTip();
    const step = (t || v).x;
    let html = `<span class="k">step</span> <b>${Math.round(step)}</b>`;
    if (t) html += `<br><span class="k">train</span> <b style="color:#7ec97e">${t.y.toFixed(4)}</b>`;
    if (v) html += `<br><span class="k">val</span> <b>${v.y.toFixed(4)}</b>`;
    showTip(html, ev);
  });
  canvas.addEventListener('mouseleave', hideTip);
}

// ── Modal: enlarged chart ──────────────────────────────────
function openModal(title, build) {
  const m = document.getElementById('modal');
  document.getElementById('modalTitle').textContent = title;
  const body = document.getElementById('modalBody');
  body.innerHTML = '';
  build(body);
  m.className = 'modal open';
}
function closeModal() {
  document.getElementById('modal').className = 'modal';
}
function expandChart(id) {
  const ch = charts[id];
  if (!ch || !ch.lastArgs) return;
  const big = document.createElement('canvas');
  big.width = Math.min(1600, window.innerWidth - 120);
  big.height = Math.min(900, window.innerHeight - 160);
  openModal(id, body => {
    body.appendChild(big);
    ch.draw(big, ...ch.lastArgs);
  });
}
function openLossModal(e) {
  const big = document.createElement('canvas');
  big.width = Math.min(1400, window.innerWidth - 120);
  big.height = Math.min(700, window.innerHeight - 200);
  const meta = document.createElement('div');
  meta.className = 'muted';
  openModal(`experiment ${e.id} — loss`, body => {
    body.appendChild(big);
    body.appendChild(meta);
    renderLossCurve(big, e.loss_curve, e.val_curve, state.lossLogScale, meta);
  });
}

// ── Refresh loop ───────────────────────────────────────────
async function refresh() {
  try {
    const [stats, lb, fr, frCM, cont, dp, archs, recent] = await Promise.all([
      get('/api/stats'), get('/api/leaderboard?n=20'),
      get('/api/frontier'), get('/api/frontier_crps_mase'),
      get('/api/continuation_frontier'),
      get('/api/data_pipeline_frontier'),
      get('/api/frozen_archs'),
      get('/api/recent?n=30'),
    ]);
    state.lastData = { stats, lb, fr, frCM, cont, dp, archs, recent };
    redraw(state.lastData);
    document.getElementById('refresh').textContent =
      'refreshed ' + new Date().toLocaleTimeString();
  } catch (e) {
    document.getElementById('refresh').textContent = 'error: ' + e;
  }
}
function redraw({ stats, lb, fr, frCM, cont, dp, archs, recent }) {
  // novel/cont split surfaces continuation activity at a glance —
  // raw count + how many of them actually succeeded.
  const novelLine = `${fmtInt(stats.n_novel_successful)} / ${fmtInt(stats.n_novel)}`;
  const contLine  = `${fmtInt(stats.n_continuation_successful)} / ${fmtInt(stats.n_continuation)}`;
  // scheduled-but-downgraded continuations look like novel rounds in the
  // experiments table; the challenges table lets us count them separately.
  const sched = stats.n_continuation_scheduled || 0;
  const downg = stats.n_continuation_downgraded || 0;
  const schedLine = sched
    ? `${fmtInt(sched - downg)} ran · ${fmtInt(downg)} downgraded`
    : '—';
  const cells = [
    ['total', fmtInt(stats.total)], ['successful', fmtInt(stats.successful)],
    ['failed', fmtInt(stats.failed)], ['miners', fmtInt(stats.n_miners)],
    ['last round', fmtInt(stats.last_round)],
    ['novel (ok/total)', novelLine],
    ['continuation (ok/total)', contLine],
    ['scheduled continuations', schedLine],
    ['best metric', fmt(stats.best_metric)],
    ['mean metric', fmt(stats.mean_metric)],
  ];
  document.getElementById('stats').innerHTML = cells.map(([k, v]) =>
    `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`
  ).join('');

  tableRaw.leaderboard  = lb;
  tableRaw.frontier     = fr;
  tableRaw.frontierCM   = frCM;
  tableRaw.continuation = (cont && cont.all) || [];
  tableRaw.dataPipeline = (dp && dp.all) || [];
  tableRaw.frozenArchs  = archs || [];
  tableRaw.recent       = recent;
  // Recent rows filtered to ts_data_pipeline for the data-pipeline tab.
  tableRaw.recentDP     = (recent || []).filter(
    e => e.task === 'ts_data_pipeline',
  );
  renderTable('leaderboard');
  renderTable('frontier');
  renderTable('frontierCM');
  renderTable('continuation');
  renderTable('dataPipeline');
  renderTable('frozenArchs');
  renderTable('recent');
  renderTable('recentDP');
  document.getElementById('lb-count').textContent = ` (${lb.length})`;
  document.getElementById('cm-count').textContent = ` (${frCM.length})`;
  const cAll = (cont && cont.all) || [];
  const cFront = (cont && cont.frontier) || [];
  document.getElementById('cont-count').textContent =
    ` (${cFront.length} on front / ${cAll.length} total)`;
  const dpAll = (dp && dp.all) || [];
  const dpFront = (dp && dp.frontier) || [];
  document.getElementById('dp-count').textContent =
    ` (${dpFront.length} on front / ${dpAll.length} total)`;
  document.getElementById('fa-count').textContent =
    ` (${(archs || []).length})`;

  // Charts share the combined point set so off-frontier points are
  // still hover/click-able.
  charts.pareto.getArgs = () => [fr, lb.concat(recent)];
  charts.paretoCM.getArgs = () => [frCM, lb.concat(recent)];
  charts.paretoCont.getArgs = () => [cFront, cAll];
  charts.paretoDP.getArgs = () => [dpFront, dpAll];
  renderChart('pareto');
  renderChart('paretoCM');
  renderChart('paretoCont');
  renderChart('paretoDP');
}

// ── Boot ───────────────────────────────────────────────────
for (const [id, m] of Object.entries({
  leaderboard:  { rowFn: row,             rank: true  },
  frontier:     { rowFn: frontRow,        rank: false },
  frontierCM:   { rowFn: frontRowCM,      rank: false },
  continuation: { rowFn: contRow,         rank: false },
  dataPipeline: { rowFn: dpRow,           rank: false },
  frozenArchs:  { rowFn: frozenArchRow,   rank: false },
  recent:       { rowFn: recentRow,       rank: false },
  recentDP:     { rowFn: recentDPRow,     rank: false },
  events:       { rowFn: eventRow,        rank: false },
  checkpoints:  { rowFn: ckRow,           rank: false },
})) {
  tableMeta[id] = m;
  tableState[id] = { sortKey: null, sortDir: 'asc', filter: '' };
  tableRaw[id] = [];
}
// Default sort: newest first for both new tables.
tableState.events.sortKey = 'id'; tableState.events.sortDir = 'desc';
tableState.checkpoints.sortKey = 'mtime'; tableState.checkpoints.sortDir = 'desc';
setupTables();

// ── Service-log filter bar ────────────────────────────────
function applyEventFilters() {
  evState.round_id = document.getElementById('evRound').value.trim();
  evState.miner_id = document.getElementById('evMiner').value.trim();
  evState.endpoint = document.getElementById('evEndpoint').value.trim();
  evState.errors = document.getElementById('evErrors').checked;
  const lim = Number(document.getElementById('evLimit').value);
  if (Number.isFinite(lim) && lim > 0) evState.limit = Math.min(500, Math.round(lim));
  loadEvents();
}
function resetEventFilters() {
  evState.kind = '';
  evState.round_id = '';
  evState.miner_id = '';
  evState.endpoint = '';
  evState.errors = false;
  evState.limit = 100;
  document.getElementById('evRound').value = '';
  document.getElementById('evMiner').value = '';
  document.getElementById('evEndpoint').value = '';
  document.getElementById('evErrors').checked = false;
  document.getElementById('evLimit').value = '100';
  loadEventStats(); loadEvents();
}
document.getElementById('evApply').onclick = applyEventFilters;
document.getElementById('evReset').onclick = resetEventFilters;
document.getElementById('evRefresh').onclick = () => {
  loadEventStats(); loadEvents();
};
// Enter inside any filter input applies. Saves a click.
['evRound', 'evMiner', 'evEndpoint', 'evLimit'].forEach(id => {
  document.getElementById(id).addEventListener('keydown', ev => {
    if (ev.key === 'Enter') applyEventFilters();
  });
});

registerChart('pareto', drawPareto, () => null);
registerChart('paretoCM', drawParetoCM, () => null);
registerChart('paretoCont', drawParetoCont, () => null);
registerChart('paretoDP', drawParetoDP, () => null);

// Expand buttons on the main charts.
document.querySelectorAll('.chart-expand').forEach(btn => {
  const target = btn.dataset.chart;
  if (target) btn.onclick = () => expandChart(target);
});
document.getElementById('modalClose').onclick = closeModal;
document.getElementById('modal').onclick = (ev) => {
  if (ev.target.id === 'modal') closeModal();
};
document.addEventListener('keydown', ev => {
  if (ev.key === 'Escape') {
    const m = document.getElementById('modal');
    if (m.classList.contains('open')) return closeModal();
    if (state.detailOpen) return closeDetail();
  }
});

// ── Tabs ───────────────────────────────────────────────────
function setActiveTab(name) {
  if (!VALID_TABS.includes(name)) return;
  state.activeTab = name;
  document.body.dataset.activeTab = name;
  document.querySelectorAll('.tab').forEach(btn => {
    btn.setAttribute('aria-selected',
      btn.dataset.tab === name ? 'true' : 'false');
  });
  try { localStorage.setItem(TAB_KEY, name); } catch (_e) { /* ignore */ }
  // Canvas dimensions aren't measured while a section is display:none.
  // Re-render the now-visible charts using the last fetched data so
  // they paint correctly on first reveal.
  if (state.lastData) redraw(state.lastData);
  // Lazy-load the heavier tabs on first reveal — and refresh on
  // subsequent reveals so a tab switch picks up new activity.
  if (name === 'service_log') {
    loadEventStats(); loadEvents();
  } else if (name === 'checkpoints') {
    loadCheckpoints();
  }
}

document.querySelectorAll('.tab').forEach(btn => {
  btn.addEventListener('click', () => setActiveTab(btn.dataset.tab));
});
// Apply persisted tab choice before the first refresh paints.
setActiveTab(state.activeTab);

const autoEl = document.getElementById('autoRefresh');
let timer = null;
function startTimer() {
  if (timer) clearInterval(timer);
  timer = setInterval(() => {
    // Pause while detail panel is open so the page doesn't reshuffle
    // under the user's reading. We do still refresh on close.
    if (state.detailOpen || !autoEl.checked) return;
    refresh();
    // The heavier tabs are lazy — only ping them while visible.
    if (state.activeTab === 'service_log') {
      loadEventStats(); loadEvents();
    } else if (state.activeTab === 'checkpoints') {
      loadCheckpoints();
    }
  }, 5000);
}
autoEl.addEventListener('change', () => {
  document.getElementById('refresh').textContent = autoEl.checked
    ? 'auto-refresh on' : 'auto-refresh paused';
});

refresh();
startTimer();
