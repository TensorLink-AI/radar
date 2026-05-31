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

const state = {
  selectedId: null,
  detailOpen: false,
  lossLogScale: false,
  lastData: null,  // last refresh payload, used by modal/redraw
};

const charts = {};  // id -> { canvas, points, draw, lastArgs }

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
  if (obj(e, 'crps') !== null) rows.push(['crps', fmt(obj(e, 'crps'))]);
  if (obj(e, 'mase') !== null) rows.push(['mase', fmt(obj(e, 'mase'))]);
  if (obj(e, 'flops_equivalent_size') !== null)
    rows.push(['flops', fmtInt(obj(e, 'flops_equivalent_size'))]);
  return rows.map(([k, v]) => `<span class="k">${k}</span> <b>${v}</b>`).join('<br>');
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
    + `<td class="metric">${fmt(e.metric)}</td>`
    + `<td class="score">${fmt(e.score, 3)}</td>`
    + (e.success ? `<td class="ok">ok</td>` : `<td class="fail">fail</td>`);
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
  const pad = 36;
  const xRange = Math.max(1e-6, xMax - xMin), yRange = Math.max(1e-6, yMax - yMin);
  const px = x => pad + (Math.log10(Math.max(1, x)) - xMin) / xRange * (c.width - 2 * pad);
  const py = y => c.height - pad - (y - yMin) / yRange * (c.height - 2 * pad);
  drawGrid(ctx, c, pad);
  // all points
  for (const e of points) {
    const x = px(obj(e, 'flops_equivalent_size') || 1), y = py(e.metric);
    const selected = e.id === state.selectedId;
    ctx.fillStyle = selected ? '#f0a040' : '#3a4050';
    ctx.beginPath();
    ctx.arc(x, y, selected ? 5 : 3, 0, 2*Math.PI);
    ctx.fill();
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
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, 2*Math.PI);
    ctx.fill();
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
  const pad = 40;
  const xRange = Math.max(1e-6, xMax - xMin), yRange = Math.max(1e-6, yMax - yMin);
  const px = x => pad + (x - xMin) / xRange * (c.width - 2 * pad);
  const py = y => c.height - pad - (y - yMin) / yRange * (c.height - 2 * pad);
  drawGrid(ctx, c, pad);
  for (const e of points) {
    const x = px(obj(e, 'crps')), y = py(obj(e, 'mase'));
    const selected = e.id === state.selectedId;
    ctx.fillStyle = selected ? '#f0a040' : '#3a4050';
    ctx.beginPath();
    ctx.arc(x, y, selected ? 5 : 3, 0, 2*Math.PI);
    ctx.fill();
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
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, 2*Math.PI);
    ctx.fill();
  }
  drawAxesLabels(ctx, c, 'crps (lower=better) →', 'mase (lower=better) →');
  drawYTicks(ctx, c, pad, yMin, yMax, 4);
  return { points: hits };
}

// ── Shared chart helpers ───────────────────────────────────
function drawGrid(ctx, c, pad) {
  ctx.strokeStyle = '#2a2e3a'; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad, pad/2); ctx.lineTo(pad, c.height - pad);
  ctx.lineTo(c.width - pad/2, c.height - pad);
  ctx.stroke();
}
function drawYTicks(ctx, c, pad, yMin, yMax, n) {
  ctx.fillStyle = '#777e8b'; ctx.font = '10px ui-monospace, monospace';
  const range = Math.max(1e-9, yMax - yMin);
  for (let i = 0; i <= n; i++) {
    const yv = yMin + (range * i / n);
    const yy = c.height - pad - (yv - yMin) / range * (c.height - 2 * pad);
    ctx.strokeStyle = '#1c1f28';
    ctx.beginPath(); ctx.moveTo(pad, yy); ctx.lineTo(c.width - pad/2, yy); ctx.stroke();
    ctx.fillStyle = '#777e8b';
    ctx.fillText(yv.toFixed(yv >= 100 ? 0 : yv >= 1 ? 2 : 3), 2, yy + 3);
  }
}
function drawAxesLabels(ctx, c, xLabel, yLabel) {
  ctx.fillStyle = '#777e8b'; ctx.font = '11px ui-monospace, monospace';
  ctx.fillText(xLabel, c.width - ctx.measureText(xLabel).width - 8, c.height - 8);
  ctx.save(); ctx.translate(12, 80); ctx.rotate(-Math.PI/2);
  ctx.fillText(yLabel, 0, 0); ctx.restore();
}
function drawEmpty(ctx, c, msg) {
  ctx.fillStyle = '#777e8b'; ctx.font = '12px ui-monospace, monospace';
  ctx.fillText(msg, 20, 30);
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
  d.innerHTML = `<span class="close" id="detailClose">×</span>`
    + `<h2>experiment ${e.id}</h2>`
    + `<div class="muted">round ${e.round_id} · miner ${esc(e.miner_id)} · `
    + `task ${esc(e.task) || '?'} · gen ${e.generation} · ${ts}</div>`
    + `<p><b>name</b> ${esc(e.name)}<br><b>metric</b> ${fmt(e.metric)} `
    + ` <b>score</b> ${fmt(e.score, 3)} <b>success</b> ${e.success}</p>`
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
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const xRange = Math.max(1e-9, xMax - xMin);
  const yRange = Math.max(1e-9, yMax - yMin);
  const px = x => pad + (x - xMin) / xRange * (c.width - 2 * pad);
  const py = y => c.height - pad - (yT(y) - yMin) / yRange * (c.height - 2 * pad);
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
    const [stats, lb, fr, frCM, recent] = await Promise.all([
      get('/api/stats'), get('/api/leaderboard?n=20'),
      get('/api/frontier'), get('/api/frontier_crps_mase'),
      get('/api/recent?n=30'),
    ]);
    state.lastData = { stats, lb, fr, frCM, recent };
    redraw(state.lastData);
    document.getElementById('refresh').textContent =
      'refreshed ' + new Date().toLocaleTimeString();
  } catch (e) {
    document.getElementById('refresh').textContent = 'error: ' + e;
  }
}
function redraw({ stats, lb, fr, frCM, recent }) {
  const cells = [
    ['total', fmtInt(stats.total)], ['successful', fmtInt(stats.successful)],
    ['failed', fmtInt(stats.failed)], ['miners', fmtInt(stats.n_miners)],
    ['last round', fmtInt(stats.last_round)],
    ['best metric', fmt(stats.best_metric)],
    ['mean metric', fmt(stats.mean_metric)],
  ];
  document.getElementById('stats').innerHTML = cells.map(([k, v]) =>
    `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`
  ).join('');

  const lbEl = document.getElementById('leaderboard');
  lbEl.innerHTML = '';
  lb.forEach((e, i) => lbEl.appendChild(row(e, i + 1)));
  document.getElementById('lb-count').textContent = ` (${lb.length})`;

  const frEl = document.getElementById('frontier');
  frEl.innerHTML = '';
  fr.forEach(e => frEl.appendChild(frontRow(e)));

  const frCMEl = document.getElementById('frontierCM');
  frCMEl.innerHTML = '';
  frCM.forEach(e => frCMEl.appendChild(frontRowCM(e)));
  document.getElementById('cm-count').textContent = ` (${frCM.length})`;

  const rEl = document.getElementById('recent');
  rEl.innerHTML = '';
  recent.forEach(e => rEl.appendChild(recentRow(e)));

  // Charts share the combined point set so off-frontier points are
  // still hover/click-able.
  charts.pareto.getArgs = () => [fr, lb.concat(recent)];
  charts.paretoCM.getArgs = () => [frCM, lb.concat(recent)];
  renderChart('pareto');
  renderChart('paretoCM');
}

// ── Boot ───────────────────────────────────────────────────
registerChart('pareto', drawPareto, () => null);
registerChart('paretoCM', drawParetoCM, () => null);

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

const autoEl = document.getElementById('autoRefresh');
let timer = null;
function startTimer() {
  if (timer) clearInterval(timer);
  timer = setInterval(() => {
    // Pause while detail panel is open so the page doesn't reshuffle
    // under the user's reading. We do still refresh on close.
    if (!state.detailOpen && autoEl.checked) refresh();
  }, 5000);
}
autoEl.addEventListener('change', () => {
  document.getElementById('refresh').textContent = autoEl.checked
    ? 'auto-refresh on' : 'auto-refresh paused';
});

refresh();
startTimer();
