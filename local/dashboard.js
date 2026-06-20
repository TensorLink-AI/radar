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
const VALID_TABS = ['architecture', 'data_pipeline', 'synth', 'lineage', 'service_log', 'checkpoints'];

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
    rows.push(['kind', `<b style="color:#f0a040">${esc(e.round_kind || 'continuation')}</b>`]);
    if (e.parent_index !== null && e.parent_index !== undefined)
      rows.push(['parent', e.parent_index]);
    if (e.n_rounds)              rows.push(['rounds', e.n_rounds]);
    if (e.cumulative_compute)    rows.push(['Σ compute', fmt(e.cumulative_compute, 2)]);
    if (e.delta !== undefined)   rows.push(['Δ', fmt(e.delta)]);
  } else if (e.round_kind && e.round_kind !== 'new') {
    rows.push(['kind', `<b style="color:#b08fe0">${esc(e.round_kind)}</b>`]);
  } else {
    rows.push(['kind', `<b style="color:#7ec97e">novel</b>`]);
  }
  // Paired per-dataset comparison vs the parent (continuations only).
  const paired = (e.objectives || {}).paired;
  if (paired && paired.n)
    rows.push(['paired', `${paired.n_improved}/${paired.n}`
      + (paired.significant ? ' <b style="color:#7ec97e">✓ sig</b>'
                            : ' <span style="color:#888">not sig</span>')]);
  if (obj(e, 'canary_metric') !== null)
    rows.push(['canary', fmt(obj(e, 'canary_metric'))]);
  if (obj(e, 'aulc') !== null) rows.push(['aulc', fmt(obj(e, 'aulc'))]);
  if (obj(e, 'gift_metric') !== null)
    rows.push(['gift', fmt(obj(e, 'gift_metric'))]);
  if (obj(e, 'frozen_arch_version') !== null)
    rows.push(['arch v', `v${obj(e, 'frozen_arch_version')}`]);
  if (obj(e, 'synth_arch_version') !== null)
    rows.push(['arch v', `v${obj(e, 'synth_arch_version')}`]);
  if (obj(e, 'crps') !== null) rows.push(['crps', fmt(obj(e, 'crps'))]);
  if (obj(e, 'mase') !== null) rows.push(['mase', fmt(obj(e, 'mase'))]);
  if (obj(e, 'flops_equivalent_size') !== null)
    rows.push(['flops', fmtInt(obj(e, 'flops_equivalent_size'))]);
  return rows.map(([k, v]) => `<span class="k">${k}</span> <b>${v}</b>`).join('<br>');
}

// Validator-owned round kinds (see docs/experiment_engine.md). Falls
// back to the old novel/cont split for rows from pre-engine DBs.
const KIND_LABELS = {
  'new':                 { short: 'novel',   cls: 'badge-novel',  title: 'trained from scratch' },
  'continuation':        { short: 'cont',    cls: 'badge-cont',   title: 'warm-started from a parent checkpoint' },
  'continuation:extend': { short: 'extend',  cls: 'badge-cont',   title: 'warm-start, re-trained the parent\'s own generator (more compute, same data)' },
  'continuation:modify': { short: 'modify',  cls: 'badge-cont',   title: 'warm-start, trained a new generator (same model, new data)' },
  'replicate':           { short: 'repl',    cls: 'badge-repl',   title: 'validator noise probe — re-run of a frontier member, new seed' },
  'ablation':            { short: 'ablate',  cls: 'badge-ablate', title: 'minimal one-diff of a frontier member' },
  'recipe_only':         { short: 'recipe',  cls: 'badge-recipe', title: 'frozen architecture — training recipe changed only' },
  'transfer':            { short: 'xfer',    cls: 'badge-xfer',   title: 'smaller-bucket winner scaled into this bucket' },
};
function kindBadge(e) {
  const kind = e.round_kind
    || (e.is_continuation ? 'continuation' : 'new');
  const meta = KIND_LABELS[kind]
    || { short: kind, cls: 'badge-novel', title: kind };
  return `<span class="badge ${meta.cls}" title="${esc(meta.title)}">${esc(meta.short)}</span>`;
}

// Short task label so the leaderboard/recent tables show whether a row
// came from the real torch+GIFT-Eval forecasting challenge, the
// synthetic-data pipeline challenge, or the numpy regression task.
const TASK_LABELS = {
  ts_forecasting: { short: 'forecast', cls: 'badge-task-fc' },
  ts_data_pipeline: { short: 'pipeline', cls: 'badge-task-dp' },
  synthetic_data_generator: { short: 'datagen', cls: 'badge-task-sdg' },
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

function labReportRow(rep) {
  const tr = document.createElement('tr');
  tr.className = 'row';
  tr.onclick = () => openModal(`lab report — experiment ${rep.experiment_id}`,
    body => {
      const pre = document.createElement('pre');
      pre.className = 'code-block';
      pre.textContent = JSON.stringify(rep, null, 2);
      body.appendChild(pre);
    });
  const paired = rep.paired || {};
  const pairedTxt = paired.n
    ? `${paired.n_improved}/${paired.n}${paired.significant ? ' ✓' : ''}`
    : '—';
  const outcome = rep.outcome || {};
  // Reuse the kind badge by faking the experiment-row shape.
  const kindCell = kindBadge({ round_kind: rep.round_kind });
  tr.innerHTML = `<td>${rep.experiment_id}</td><td>${rep.round_id ?? '—'}</td>`
    + `<td>${taskBadge({ task: rep.task })}</td>`
    + `<td>${kindCell}</td>`
    + `<td>${esc(rep.name || '')}</td>`
    + `<td class="num">${fmt(outcome.metric)}</td>`
    + `<td class="num">${fmt(rep.delta)}</td>`
    + `<td class="num">${pairedTxt}</td>`
    + `<td class="verdict" title="${esc(rep.verdict || '')}">${esc(rep.verdict || '')}</td>`;
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

// synthetic_data_generator tab. Same GIFT-only crps×mase scoring as
// ts_forecasting, but the arch is fixed — so the per-row version of
// interest is synth_arch_version (continuation lineages pin to it), not a
// frozen_arch_version.
function synthFrontierRow(e) {
  const tr = document.createElement('tr');
  tr.className = 'row' + (e.id === state.selectedId ? ' selected' : '');
  tr.dataset.expId = e.id;
  tr.onclick = () => showDetail(e.id);
  tr.innerHTML = `<td>${e.id}</td><td>${e.round_id}</td>`
    + `<td>${esc(e.miner_id)}</td><td>${esc(e.name)}</td>`
    + `<td class="num">${fmtInt(obj(e, 'synth_arch_version'))}</td>`
    + `<td>${kindBadge(e)}</td>`
    + `<td class="num">${fmt(obj(e, 'crps'))}</td>`
    + `<td class="num">${fmt(obj(e, 'mase'))}</td>`
    + `<td class="num">${fmtInt(obj(e, 'flops_equivalent_size'))}</td>`
    + `<td class="metric">${fmt(e.metric)}</td>`
    + `<td class="score">${fmt(e.score, 3)}</td>`;
  return tr;
}

function recentSDGRow(e) {
  const tr = document.createElement('tr');
  tr.className = 'row' + (e.id === state.selectedId ? ' selected' : '');
  tr.dataset.expId = e.id;
  tr.onclick = () => showDetail(e.id);
  tr.innerHTML = `<td>${e.id}</td><td>${e.round_id}</td>`
    + `<td>${esc(e.miner_id)}</td><td>${esc(e.name)}</td>`
    + `<td class="num">${fmtInt(obj(e, 'synth_arch_version'))}</td>`
    + `<td>${kindBadge(e)}</td>`
    + `<td class="num">${fmt(obj(e, 'crps'))}</td>`
    + `<td class="num">${fmt(obj(e, 'mase'))}</td>`
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

// Datagen tab: lab reports + service log scoped to synthetic_data_generator.
// Kept separate from the global lab_reports / service_log tabs (which mix
// all tasks and cap at the most-recent window) so the datagen challenge's
// own post-mortems and service calls are always visible in context.
async function loadSynthLogs() {
  const TASK = 'synthetic_data_generator';
  try {
    const [reports, events] = await Promise.all([
      get('/api/lab_reports?n=100&task=' + TASK),
      get('/api/events?limit=200&task=' + encodeURIComponent(TASK)),
    ]);
    tableRaw.synthLabReports = reports || [];
    tableRaw.synthEvents = events || [];
    renderTable('synthLabReports');
    renderTable('synthEvents');
    const lr = document.getElementById('sdg-lr-count');
    if (lr) lr.textContent = ` (${(reports || []).length})`;
    const ev = document.getElementById('sdg-ev-count');
    if (ev) ev.textContent = ` (${(events || []).length})`;
  } catch (e) { /* transient fetch error — next refresh retries */ }
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
  // Continuation runs warm-start a parent's weights, so the parent's curve is
  // the head of this one. Offer a stitched lineage view (default) for them.
  const canLineage = !!(e.is_continuation && e.parent_index != null);
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
          + (canLineage
              ? `<button id="lossThisRun">this run</button>`
                + `<button id="lossLineage" class="active">full lineage</button>`
              : '')
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
    // Lineage view stitches the whole continuation chain; default on for
    // continuation runs. lineageRuns is filled lazily on first need.
    state.lossLineage = canLineage;
    state.lineageRuns = null;
    const canvas = () => document.getElementById('lossCurve');
    const meta = () => document.getElementById('lossMeta');
    const drawLoss = () => {
      if (state.lossLineage && state.lineageRuns && state.lineageRuns.length > 1) {
        renderLineageCurve(canvas(), state.lineageRuns, state.lossLogScale, meta());
      } else {
        renderLossCurve(canvas(), e.loss_curve, e.val_curve, state.lossLogScale, meta());
      }
    };
    const ensureLineage = async () => {
      if (state.lineageRuns !== null) return;
      try {
        const lc = await get(`/api/experiment/${e.id}/lineage_curve`);
        state.lineageRuns = (lc && lc.runs) || [];
      } catch (_e) { state.lineageRuns = []; }
    };
    if (canLineage) ensureLineage().then(drawLoss);
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
    if (canLineage) {
      const thisBtn = document.getElementById('lossThisRun');
      const linBtn = document.getElementById('lossLineage');
      thisBtn.onclick = () => {
        state.lossLineage = false;
        thisBtn.classList.add('active'); linBtn.classList.remove('active');
        drawLoss();
      };
      linBtn.onclick = async () => {
        state.lossLineage = true;
        linBtn.classList.add('active'); thisBtn.classList.remove('active');
        await ensureLineage();
        drawLoss();
      };
    }
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
  drawCurveCore(c, ctx, pts, valPts, logY, metaEl, null);
}

// Stitch a continuation lineage (root → … → leaf) into one continuous
// curve: each run's points are offset past the previous run's width so an
// *extend* (more compute, same data) reads as the tail of its parent's
// curve. ``bounds`` marks where each successive run begins.
function buildLineage(runs) {
  let off = 0;
  const train = [], val = [], bounds = [];
  (runs || []).forEach((r, i) => {
    const tp = normalizeCurve(r.loss_curve);
    const vp = normalizeCurve(r.val_curve);
    // Train losses are logged per step as a bare list (x = index), while val
    // carries real step numbers. Spread the step-less train curve across the
    // run's val step domain so the two share one x-axis within the run —
    // otherwise train gets crushed into the left sliver of the run's width
    // and the polyline draws a flat bridge to the next run. Mirrors the
    // single-run path in renderLossCurve.
    const trainStepless = tp.every((p, k) => p.x === k);
    const valMax = vp.length ? Math.max(...vp.map(p => p.x)) : 0;
    if (trainStepless && valMax > 0 && tp.length > 1) {
      const n = tp.length;
      for (let k = 0; k < n; k++) tp[k].x = (k / (n - 1)) * valMax;
    }
    const maxX = Math.max(
      0,
      ...(tp.length ? tp.map(p => p.x) : [0]),
      ...(vp.length ? vp.map(p => p.x) : [0]),
    );
    if (i > 0) bounds.push({
      x: off, round_id: r.round_id,
      label: r.continuation_kind || r.mode || 'cont',
    });
    tp.forEach(p => train.push({ x: off + p.x, y: p.y }));
    vp.forEach(p => val.push({ x: off + p.x, y: p.y }));
    // Small gap so adjacent runs stay visually distinct even when a run
    // logged a single point.
    off += maxX + Math.max(1, maxX * 0.03);
  });
  return { train, val, bounds };
}

function renderLineageCurve(canvas, runs, logY, metaEl) {
  if (!canvas) return;
  const c = canvas, ctx = c.getContext('2d');
  ctx.fillStyle = '#161820'; ctx.fillRect(0, 0, c.width, c.height);
  const { train, val, bounds } = buildLineage(runs);
  if (train.length === 0) { drawEmpty(ctx, c, 'no finite points'); return; }
  drawCurveCore(c, ctx, train, val, logY, metaEl, {
    bounds, runCount: (runs || []).length,
  });
}

// Shared drawing: axes, train/val polylines, optional lineage run
// boundaries. ``pts``/``valPts`` are already in display x-coordinates.
function drawCurveCore(c, ctx, pts, valPts, logY, metaEl, opts) {
  const bounds = (opts && opts.bounds) || null;
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
  // Lineage run boundaries: a dashed vertical where each warm-started run
  // picks up, labelled with its round + continuation kind.
  if (bounds && bounds.length) {
    ctx.save();
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = '#5b6b8c'; ctx.lineWidth = 1;
    ctx.font = '9px ui-monospace, monospace';
    for (const b of bounds) {
      const bx = px(b.x);
      if (!Number.isFinite(bx) || bx < pad || bx > c.width - pad) continue;
      ctx.beginPath(); ctx.moveTo(bx, pad); ctx.lineTo(bx, c.height - pad); ctx.stroke();
      ctx.fillStyle = '#8fa3c8';
      const lbl = `r${b.round_id ?? '?'}·${b.label || ''}`;
      ctx.fillText(lbl, bx + 2, pad + 10);
    }
    ctx.restore();
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
    let text = (opts && opts.runCount > 1)
      ? `lineage: ${opts.runCount} runs · ` : '';
    text += `train: ${pts.length} pts · first=${trainYs[0].toFixed(4)} `
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
    if (state.lossLineage && state.lineageRuns && state.lineageRuns.length > 1) {
      renderLineageCurve(big, state.lineageRuns, state.lossLogScale, meta);
    } else {
      renderLossCurve(big, e.loss_curve, e.val_curve, state.lossLogScale, meta);
    }
  });
}

// ── Lineage forest ─────────────────────────────────────────
// A two-lane tree of parent→child lineages (forecasting on top,
// pipeline on the bottom) plus the cross-task frozen-arch interaction:
// a forecasting experiment is promoted to an arch snapshot (◆ on the
// divider) that pipeline runs train against. Solid edges = continuation
// lineage; dashed = promote (forecast→arch) and train (arch→pipeline).
const LIN_TASKS = ['ts_forecasting', 'ts_data_pipeline', 'synthetic_data_generator'];
const LIN_COLORS = {
  ts_forecasting: '#7ec5e8',
  ts_data_pipeline: '#e8c47e',
  synthetic_data_generator: '#8fd49a',
};
const LIN_LANE_LABELS = {
  ts_forecasting: 'forecasting',
  ts_data_pipeline: 'pipeline',
  synthetic_data_generator: 'datagen (fixed arch)',
};
const LIN_ARCH_COLOR = '#a07cc7';
const linState = {
  task: 'all', interactions: true, singletons: false, data: null, hits: [],
};

async function loadLineage() {
  try {
    linState.data = await get('/api/lineage');
  } catch (_e) {
    linState.data = { nodes: [], archs: [], edges: [] };
  }
  renderLineage();
}

// Tidy per-lane forest layout: column = in-lane lineage depth, row =
// leaf order (internal nodes centre over their children).
function layoutLineage(data, opts) {
  const COL_W = 64, ROW_H = 30, PAD_X = 40, PAD_TOP = 26, PAD_BOT = 20;
  const LANE_GAP = 70;  // room for arch nodes + lane labels on the divider
  const taskOK = t =>
    opts.task === 'all' ? LIN_TASKS.includes(t) : t === opts.task;
  const all = (data.nodes || []).filter(n =>
    taskOK(n.task) && (n.in_lineage || opts.singletons));
  const byId = {};
  for (const n of all) byId[n.id] = n;

  function layoutLane(task) {
    const lane = all.filter(n => n.task === task);
    const laneSet = new Set(lane.map(n => n.id));
    const kids = {};
    for (const n of lane) {
      const p = n.parent_index;
      if (p != null && laneSet.has(p)) (kids[p] = kids[p] || []).push(n.id);
    }
    const colOf = {};
    function col(id, seen) {
      if (colOf[id] != null) return colOf[id];
      const p = byId[id].parent_index;
      let c = 0;
      if (p != null && laneSet.has(p) && !(seen && seen.has(id))) {
        const s = seen || new Set(); s.add(id);
        c = col(p, s) + 1;
      }
      colOf[id] = c; return c;
    }
    for (const n of lane) col(n.id);
    const roots = lane
      .filter(n => { const p = n.parent_index; return !(p != null && laneSet.has(p)); })
      .sort((a, b) => a.id - b.id);
    const rowOf = {};
    let nextRow = 0;
    function assign(id) {
      const ks = (kids[id] || []).slice().sort((a, b) => a - b);
      if (ks.length === 0) { rowOf[id] = nextRow++; return rowOf[id]; }
      const rs = ks.map(assign);
      rowOf[id] = rs.reduce((a, b) => a + b, 0) / rs.length;
      return rowOf[id];
    }
    for (const r of roots) { assign(r.id); nextRow += 0.6; }
    const pos = {};
    let maxCol = 0;
    for (const n of lane) {
      pos[n.id] = { col: colOf[n.id], row: rowOf[n.id] };
      if (colOf[n.id] > maxCol) maxCol = colOf[n.id];
    }
    return { lane, pos, maxCol, rows: nextRow };
  }

  // Stack the present lanes top→bottom in fixed task order. In 'all' mode show
  // every task that has nodes; a single-task filter shows just that lane.
  const laneTasks = (opts.task === 'all' ? LIN_TASKS : [opts.task])
    .filter(t => all.some(n => n.task === t));
  const layouts = laneTasks.map(t => Object.assign({ task: t }, layoutLane(t)));
  const maxCol = Math.max(0, ...layouts.map(l => l.maxCol));
  const width = Math.max(900, PAD_X * 2 + (maxCol + 1) * COL_W);

  const xOf = c => PAD_X + c * COL_W + 14;
  const placed = {};
  const nodeDraws = [];
  const hits = [];
  const lanes = [];
  const dividers = [];
  // The frozen-arch promotion interaction lives on the divider between the
  // forecasting and pipeline lanes; null when both aren't currently shown.
  let archDividerY = null;
  let y = PAD_TOP;
  layouts.forEach((layout, i) => {
    if (i > 0) {
      const divY = y - LANE_GAP / 2;
      dividers.push(divY);
      if (layouts[i - 1].task === 'ts_forecasting'
          && layout.task === 'ts_data_pipeline') archDividerY = divY;
    }
    const top = y;
    lanes.push({
      task: layout.task, top,
      label: LIN_LANE_LABELS[layout.task] || layout.task,
    });
    for (const n of layout.lane) {
      const p = layout.pos[n.id];
      const x = xOf(p.col), ny = top + p.row * ROW_H + 14;
      placed[n.id] = { x, y: ny };
      nodeDraws.push({
        node: n, x, y: ny, color: LIN_COLORS[n.task] || '#9aa4b2',
        cont: !!n.is_continuation, selected: n.id === state.selectedId,
      });
      hits.push({ sx: x, sy: ny, r: 8, kind: 'exp', node: n });
    }
    y = top + Math.max(1, layout.rows) * ROW_H + LANE_GAP;
  });
  // No lanes (no nodes in view) → keep a sane canvas so the empty-state
  // message still renders rather than a 0/negative-height canvas.
  const height = Math.max(PAD_TOP + ROW_H + PAD_BOT, (y - LANE_GAP) + PAD_BOT);

  const edgeDraws = [];
  for (const e of (data.edges || [])) {
    if (e.kind !== 'lineage') continue;
    const a = placed[e.from], b = placed[e.to];
    if (a && b) edgeDraws.push({ x1: a.x, y1: a.y, x2: b.x, y2: b.y, dashed: false, color: '#4a5163' });
  }

  const archDraws = [];
  if (opts.interactions && archDividerY != null) {
    for (const a of (data.archs || [])) {
      const src = placed[a.source_experiment_id];
      const consumers = (data.edges || [])
        .filter(e => e.kind === 'train' && e.from === `arch:${a.version}`)
        .map(e => placed[e.to]).filter(Boolean);
      if (!src && consumers.length === 0) continue;
      const cAvg = consumers.length
        ? consumers.reduce((s, c) => s + c.x, 0) / consumers.length : null;
      const ax = (src && cAvg != null) ? (src.x + cAvg) / 2 : (src ? src.x : cAvg);
      archDraws.push({ arch: a, x: ax, y: archDividerY });
      hits.push({ sx: ax, sy: archDividerY, r: 8, kind: 'arch', node: a });
      if (src) edgeDraws.push({ x1: src.x, y1: src.y, x2: ax, y2: archDividerY, dashed: true, color: '#8a7ec0' });
      for (const c of consumers)
        edgeDraws.push({ x1: ax, y1: archDividerY, x2: c.x, y2: c.y, dashed: true, color: '#8a7ec0' });
    }
  }

  return {
    width, height, nodeDraws, archDraws, edgeDraws, hits,
    lanes, dividers, nExp: nodeDraws.length, nArch: archDraws.length,
  };
}

function drawLinEdge(ctx, e) {
  ctx.save();
  ctx.strokeStyle = e.color; ctx.lineWidth = 1.3;
  ctx.setLineDash(e.dashed ? [4, 3] : []);
  ctx.beginPath();
  ctx.moveTo(e.x1, e.y1);
  const dx = e.x2 - e.x1, dy = e.y2 - e.y1;
  if (Math.abs(dx) >= Math.abs(dy)) {
    const cx = e.x1 + dx * 0.5;
    ctx.bezierCurveTo(cx, e.y1, cx, e.y2, e.x2, e.y2);
  } else {
    const cy = e.y1 + dy * 0.5;
    ctx.bezierCurveTo(e.x1, cy, e.x2, cy, e.x2, e.y2);
  }
  ctx.stroke();
  ctx.restore();
}

function drawDiamond(ctx, x, y, r) {
  ctx.beginPath();
  ctx.moveTo(x, y - r); ctx.lineTo(x + r, y);
  ctx.lineTo(x, y + r); ctx.lineTo(x - r, y);
  ctx.closePath();
  ctx.fill(); ctx.stroke();
}

function drawLineage(canvas, L) {
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#161820'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (L.nExp === 0) { drawEmpty(ctx, canvas, 'no lineages to show'); return; }
  ctx.strokeStyle = '#252a36'; ctx.lineWidth = 1; ctx.setLineDash([]);
  for (const dy of (L.dividers || [])) {
    ctx.beginPath(); ctx.moveTo(0, dy); ctx.lineTo(canvas.width, dy); ctx.stroke();
  }
  ctx.fillStyle = '#5a6172'; ctx.font = '600 11px ui-monospace, monospace';
  for (const lane of (L.lanes || [])) {
    ctx.fillText(lane.label, 8, Math.max(11, lane.top - 14));
  }
  for (const e of L.edgeDraws) drawLinEdge(ctx, e);
  ctx.setLineDash([]);
  for (const d of L.nodeDraws) {
    ctx.fillStyle = d.selected ? '#f0a040' : d.color;
    drawMarker(ctx, d.x, d.y, d.selected ? 6 : 5, d.cont);
  }
  ctx.strokeStyle = '#11131a'; ctx.lineWidth = 1;
  for (const a of L.archDraws) {
    ctx.fillStyle = LIN_ARCH_COLOR;
    drawDiamond(ctx, a.x, a.y, 6);
  }
}

function archTooltip(a) {
  const rows = [
    ['frozen arch', `v${a.version}`],
    ['source exp', a.source_experiment_id ?? '—'],
    ['source metric', fmt(a.source_metric)],
    ['consumers', fmtInt(a.n_consumers)],
  ];
  return rows.map(([k, v]) => `<span class="k">${k}</span> <b>${v}</b>`).join('<br>');
}

function renderLineage() {
  const canvas = document.getElementById('lineageTree');
  if (!canvas || !linState.data) return;
  const L = layoutLineage(linState.data, linState);
  canvas.width = L.width; canvas.height = L.height;
  drawLineage(canvas, L);
  linState.hits = L.hits;
  const el = document.getElementById('lin-count');
  if (el) el.textContent = ` (${L.nExp} runs · ${L.nArch} arch)`;
}

function buildLinChips() {
  const host = document.getElementById('linTaskChips');
  if (!host) return;
  const opts = [['all', 'all'], ['ts_forecasting', 'forecasting'],
                ['ts_data_pipeline', 'pipeline'],
                ['synthetic_data_generator', 'datagen']];
  host.innerHTML = opts.map(([v, label]) =>
    `<button class="chip" data-lintask="${v}"`
    + (linState.task === v ? ' data-active="true"' : '') + '>'
    + `<span class="chip-dot"></span><span class="chip-label">${label}</span></button>`
  ).join('');
  host.querySelectorAll('[data-lintask]').forEach(b => {
    b.onclick = () => {
      linState.task = b.dataset.lintask;
      buildLinChips(); renderLineage();
    };
  });
}

function setupLineage() {
  const canvas = document.getElementById('lineageTree');
  if (!canvas) return;
  const at = ev => {
    const rect = canvas.getBoundingClientRect();
    const sx = (ev.clientX - rect.left) * (canvas.width / rect.width);
    const sy = (ev.clientY - rect.top) * (canvas.height / rect.height);
    return nearest(linState.hits, sx, sy, 12);
  };
  canvas.addEventListener('mousemove', ev => {
    const hit = at(ev);
    if (hit) {
      showTip(hit.kind === 'arch' ? archTooltip(hit.node) : expTooltip(hit.node), ev);
      canvas.style.cursor = 'pointer';
    } else { hideTip(); canvas.style.cursor = 'default'; }
  });
  canvas.addEventListener('mouseleave', hideTip);
  canvas.addEventListener('click', ev => {
    const hit = at(ev);
    if (!hit) return;
    if (hit.kind === 'arch') {
      if (hit.node.source_experiment_id) showDetail(hit.node.source_experiment_id);
    } else showDetail(hit.node.id);
  });
  const it = document.getElementById('linInteractions');
  const sg = document.getElementById('linSingletons');
  const rf = document.getElementById('linRefresh');
  if (it) it.onchange = e => { linState.interactions = e.target.checked; renderLineage(); };
  if (sg) sg.onchange = e => { linState.singletons = e.target.checked; renderLineage(); };
  if (rf) rf.onclick = loadLineage;
  buildLinChips();
}

// ── Noise floor strip (Lab reports tab) ────────────────────
function renderNoise(noise) {
  const host = document.getElementById('noiseStats');
  if (!host) return;
  const overall = (noise && noise.overall) || {};
  const cells = [];
  if (!overall.n_pairs) {
    host.innerHTML = `<div class="stat"><div class="k">noise floor</div>`
      + `<div class="v muted">no replicate pairs yet — schedule with --replicate_pct</div></div>`;
    return;
  }
  cells.push(['replicate pairs', fmtInt(overall.n_pairs)]);
  cells.push(['σ (metric)', fmt(overall.sigma_metric, 5)]);
  cells.push(['σ (relative)', overall.sigma_rel != null
    ? (overall.sigma_rel * 100).toFixed(2) + '%' : '—']);
  cells.push(['95% threshold', fmt(overall.threshold, 5)]);
  for (const [task, floor] of Object.entries((noise && noise.by_task) || {})) {
    cells.push([`σ · ${task}`, `${fmt(floor.sigma_metric, 5)} (${floor.n_pairs}p)`]);
  }
  host.innerHTML = cells.map(([k, v]) =>
    `<div class="stat"><div class="k">${esc(k)}</div><div class="v">${v}</div></div>`
  ).join('');
}

// ── Refresh loop ───────────────────────────────────────────
async function refresh() {
  // Prefix failures with the endpoint — a bare fetch rejection reads
  // "Failed to fetch" with no hint of which of the 11 calls died.
  const api = p => get(p).catch(e => {
    throw new Error(`${p}: ${e.message || e}`);
  });
  try {
    const [stats, lb, fr, frCM, cont, dp, archs, recent, synth, labReports, noise] = await Promise.all([
      api('/api/stats'), api('/api/leaderboard?n=20'),
      api('/api/frontier'), api('/api/frontier_crps_mase'),
      api('/api/continuation_frontier'),
      api('/api/data_pipeline_frontier'),
      api('/api/frozen_archs'),
      api('/api/recent?n=30'),
      api('/api/synth_frontier'),
      api('/api/lab_reports?n=100'),
      api('/api/noise'),
    ]);
    state.lastData = { stats, lb, fr, frCM, cont, dp, archs, recent, synth, labReports, noise };
    redraw(state.lastData);
    document.getElementById('refresh').textContent =
      'refreshed ' + new Date().toLocaleTimeString();
  } catch (e) {
    document.getElementById('refresh').textContent = 'error: ' + e;
  }
}
function redraw({ stats, lb, fr, frCM, cont, dp, archs, recent, synth, labReports, noise }) {
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
  // Special-round counts + the noise floor — present only on DBs
  // written by the experiment-engine validator.
  const nSpecial = (stats.n_replicate || 0) + (stats.n_ablation || 0)
    + (stats.n_recipe_only || 0) + (stats.n_transfer || 0);
  if (nSpecial) {
    cells.push(['special rounds',
      `${fmtInt(stats.n_replicate || 0)}r · ${fmtInt(stats.n_ablation || 0)}a`
      + ` · ${fmtInt(stats.n_recipe_only || 0)}rec · ${fmtInt(stats.n_transfer || 0)}x`]);
  }
  if (stats.noise && stats.noise.n_pairs) {
    cells.push(['noise floor (σ / 95%)',
      `${fmt(stats.noise.sigma_metric, 4)} / ${fmt(stats.noise.threshold, 4)}`]);
  }
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
  // synthetic_data_generator tab: its own crps×mase frontier + recent runs.
  const synthAll        = (synth && synth.all) || [];
  const synthFront      = (synth && synth.frontier) || [];
  tableRaw.synthFrontier = synthFront;
  tableRaw.recentSDG    = (recent || []).filter(
    e => e.task === 'synthetic_data_generator',
  );
  tableRaw.labReports = labReports || [];
  renderTable('leaderboard');
  renderTable('frontier');
  renderTable('frontierCM');
  renderTable('continuation');
  renderTable('dataPipeline');
  renderTable('frozenArchs');
  renderTable('recent');
  renderTable('recentDP');
  renderTable('synthFrontier');
  renderTable('recentSDG');
  renderTable('labReports');
  const lrCount = document.getElementById('lr-count');
  if (lrCount) lrCount.textContent = ` (${(labReports || []).length})`;
  renderNoise(noise);
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
  const sdgCount = document.getElementById('sdg-count');
  if (sdgCount) sdgCount.textContent =
    ` (${synthFront.length} on front / ${synthAll.length} total)`;
  const sdgArchv = document.getElementById('sdg-archv');
  if (sdgArchv) {
    const vs = ((synth && synth.versions) || []).filter(v => v);
    sdgArchv.textContent = vs.length ? 'v' + vs.join('/') : '';
  }

  // Charts share the combined point set so off-frontier points are
  // still hover/click-able.
  charts.pareto.getArgs = () => [fr, lb.concat(recent)];
  charts.paretoCM.getArgs = () => [frCM, lb.concat(recent)];
  charts.paretoCont.getArgs = () => [cFront, cAll];
  charts.paretoDP.getArgs = () => [dpFront, dpAll];
  charts.paretoSDG.getArgs = () => [synthFront, synthAll];
  renderChart('pareto');
  renderChart('paretoCM');
  renderChart('paretoCont');
  renderChart('paretoDP');
  renderChart('paretoSDG');
}

// ── Boot ───────────────────────────────────────────────────
for (const [id, m] of Object.entries({
  leaderboard:  { rowFn: row,             rank: true  },
  frontier:     { rowFn: frontRow,        rank: false },
  frontierCM:   { rowFn: frontRowCM,      rank: false },
  continuation: { rowFn: contRow,         rank: false },
  dataPipeline: { rowFn: dpRow,           rank: false },
  frozenArchs:  { rowFn: frozenArchRow,   rank: false },
  synthFrontier:{ rowFn: synthFrontierRow,rank: false },
  recent:       { rowFn: recentRow,       rank: false },
  recentDP:     { rowFn: recentDPRow,     rank: false },
  recentSDG:    { rowFn: recentSDGRow,    rank: false },
  events:       { rowFn: eventRow,        rank: false },
  checkpoints:  { rowFn: ckRow,           rank: false },
  labReports:   { rowFn: labReportRow,    rank: false },
  synthLabReports: { rowFn: labReportRow, rank: false },
  synthEvents:  { rowFn: eventRow,        rank: false },
})) {
  tableMeta[id] = m;
  tableState[id] = { sortKey: null, sortDir: 'asc', filter: '' };
  tableRaw[id] = [];
}
// Default sort: newest first for both new tables.
tableState.events.sortKey = 'id'; tableState.events.sortDir = 'desc';
tableState.checkpoints.sortKey = 'mtime'; tableState.checkpoints.sortDir = 'desc';
tableState.labReports.sortKey = 'experiment_id'; tableState.labReports.sortDir = 'desc';
tableState.synthLabReports.sortKey = 'experiment_id'; tableState.synthLabReports.sortDir = 'desc';
tableState.synthEvents.sortKey = 'id'; tableState.synthEvents.sortDir = 'desc';
setupTables();
setupLineage();

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
registerChart('paretoSDG', drawParetoCM, () => null);

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
  } else if (name === 'lineage') {
    loadLineage();
  } else if (name === 'synth') {
    loadSynthLogs();
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
    } else if (state.activeTab === 'lineage') {
      loadLineage();
    } else if (state.activeTab === 'synth') {
      loadSynthLogs();
    }
  }, 5000);
}
autoEl.addEventListener('change', () => {
  document.getElementById('refresh').textContent = autoEl.checked
    ? 'auto-refresh on' : 'auto-refresh paused';
});

refresh();
startTimer();
