"use strict";

const state = {
  pointsetId: null,
  data: null,
  plans: [],
  selectedPlan: null,
  jobResult: null,
  compare: [],
  gcpToggle: {},
  corrToggle: {},
  groupToggle: {},
  locked: {},
  splits: {},
};

const COLORS = ["#4db6ff", "#ffb454", "#5dd39e", "#c792ea", "#ff6b6b", "#7fdbff"];

async function api(path, opts = {}) {
  const resp = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  const text = await resp.text();
  const body = text ? JSON.parse(text) : null;
  if (!resp.ok) {
    const detail = body && body.detail ? body.detail : resp.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return body;
}

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) node.setAttribute(k, v);
  }
  for (const child of [].concat(children)) {
    if (child == null) continue;
    node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
  }
  return node;
}

function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

function fmt(x, d = 3) {
  if (x == null || Number.isNaN(x)) return "-";
  return Number(x).toFixed(d);
}

async function loadDemo() {
  setBanner("");
  const demo = await api("/api/demo");
  const res = await api("/api/pointsets", { method: "POST", body: JSON.stringify(demo) });
  state.pointsetId = res.id;
  await refreshAll();
}

async function refreshAll() {
  if (!state.pointsetId) return;
  state.data = await api(`/api/pointsets/${state.pointsetId}/data`);
  const ps = state.data.pointset;
  document.getElementById("dataset-info").textContent =
    `#${ps.id} ${ps.name} · 单位 ${ps.source_unit}（×${ps.unit_scale_to_m}→m）· ` +
    `CRS ${ps.source_crs || "-"}`;
  renderDatasetPanel();
  const plans = await api(`/api/pointsets/${state.pointsetId}/plans`);
  state.plans = plans.plans;
  renderPlanList();
  if (state.selectedPlan) {
    const stillThere = state.plans.find((p) => p.id === state.selectedPlan);
    if (stillThere) await selectPlan(state.selectedPlan);
  }
}

function renderDatasetPanel() {
  const panel = document.getElementById("dataset-panel");
  clear(panel);
  for (const strip of state.data.strips) {
    panel.appendChild(el("div", { class: "row" }, [
      el("span", { style: `color:${COLORS[stripIndex(strip.strip_uid) % COLORS.length]}` }, "■ "),
      `${strip.strip_uid}: ${strip.points.length} 点`,
    ]));
  }
  panel.appendChild(el("div", { class: "small" },
    `GCP ${state.data.gcps.length} 个 · 候选对应 ${state.data.correspondences.length} 对`));
}

function stripIndex(uid) {
  return state.data.strips.findIndex((s) => s.strip_uid === uid);
}

function modelShort(v) {
  return v.replace("-v1", "").replace("strip_", "");
}

function renderPlanList() {
  const root = document.getElementById("plan-list");
  clear(root);
  for (const plan of state.plans) {
    const tags = [el("span", { class: "tag" }, modelShort(plan.model_version))];
    if (plan.published) tags.push(el("span", { class: "tag pub" }, "已发布"));
    const card = el("div", { class: "card" + (plan.id === state.selectedPlan ? " active" : "") }, [
      el("div", { class: "row", style: "justify-content:space-between" }, [
        el("span", { class: "name" }, `#${plan.id} ${plan.name}`),
        el("span", { class: "row" }, tags),
      ]),
      el("div", { class: "meta" }, plan.parent_plan_id ? `派生自 #${plan.parent_plan_id}` : "根方案"),
      el("div", { class: "row", style: "margin-top:5px" }, [
        el("button", { onclick: () => selectPlan(plan.id) }, "打开"),
        el("button", { onclick: () => toggleCompare(plan.id) },
          state.compare.includes(plan.id) ? "✓ 比较中" : "比较"),
      ]),
    ]);
    root.appendChild(card);
  }
}

async function createPlan() {
  const model = document.getElementById("new-model").value;
  const name = document.getElementById("new-name").value || `${modelShort(model)} ${Date.now() % 100000}`;
  let drift_breaks = {};
  if (model === "strip_drift-v1") {
    for (const strip of state.data.strips) drift_breaks[strip.strip_uid] = [0, 0.5, 1];
  }
  try {
    const plan = await api(`/api/pointsets/${state.pointsetId}/plans`, {
      method: "POST",
      body: JSON.stringify({ name, model_version: model, drift_breaks }),
    });
    state.selectedPlan = plan.id;
    await refreshAll();
  } catch (e) { setBanner(e.message, true); }
}

async function selectPlan(planId) {
  state.selectedPlan = planId;
  const plan = state.plans.find((p) => p.id === planId) ||
               await api(`/api/plans/${planId}`);
  initTogglesFromConfig(plan.config);
  renderPlanList();
  renderPlanControls(plan);
  document.getElementById("plan-controls").classList.remove("hidden");
  // Always-200 probe for solved state (no noisy failed-resource log).
  const probe = await api(`/api/plans/${planId}/latest`);
  if (probe.result) {
    const corrected = await api(`/api/plans/${planId}/corrected`);
    const plansData = await api(`/api/pointsets/${state.pointsetId}/plans`);
    const fresh = plansData.plans.find((p) => p.id === planId);
    renderResults(fresh, corrected);
  } else {
    clearResults("该方案尚未求解。点击“求解”运行当前配置。");
  }
}

async function fetchResultAndRender(planId) {
  const corrected = await api(`/api/plans/${planId}/corrected`);
  const plansData = await api(`/api/pointsets/${state.pointsetId}/plans`);
  const fresh = plansData.plans.find((p) => p.id === planId);
  renderResults(fresh, corrected);
}

function initTogglesFromConfig(cfg) {
  state.gcpToggle = {};
  state.corrToggle = {};
  state.groupToggle = {};
  state.locked = {};
  for (const g of state.data.gcps) {
    state.gcpToggle[g.id] = !cfg.disabled_gcp_ids.includes(g.id);
    state.locked[g.id] = cfg.locked_gcp_ids.includes(g.id);
  }
  for (const c of state.data.correspondences) {
    state.corrToggle[c.id] = !cfg.disabled_corr_ids.includes(c.id);
  }
  for (const group of corrGroups()) {
    state.groupToggle[group] = !cfg.disabled_corr_groups.includes(group);
  }
  state.splits = {};
  for (const [sid, brks] of Object.entries(cfg.drift_breaks || {})) {
    state.splits[sid] = brks.slice(1, -1).join(", ");
  }
  document.getElementById("sigma").value = cfg.outlier_sigma;
  document.getElementById("reject").checked = cfg.reject_outliers;
}

function corrGroups() {
  const groups = new Set();
  for (const c of state.data.correspondences) {
    const pa = state.data.strips.find((st) => st.points.some((p) => p.id === c.point_a_id));
    const pb = state.data.strips.find((st) => st.points.some((p) => p.id === c.point_b_id));
    if (pa && pb) groups.add(`${pa.strip_uid}:${pb.strip_uid}`);
  }
  return [...groups].sort();
}

function renderPlanControls(plan) {
  const cfg = plan.config;

  const gcpRoot = document.getElementById("gcp-list");
  clear(gcpRoot);
  for (const g of state.data.gcps) {
    const lock = el("input", { type: "checkbox" });
    lock.checked = !!state.locked[g.id];
    lock.addEventListener("change", () => {
      state.locked[g.id] = lock.checked;
      if (lock.checked) state.gcpToggle[g.id] = true;
      renderPlanControls(plan);
    });
    const enable = el("input", { type: "checkbox" });
    enable.checked = !!state.gcpToggle[g.id];
    enable.disabled = !!state.locked[g.id];
    enable.addEventListener("change", () => { state.gcpToggle[g.id] = enable.checked; });
    gcpRoot.appendChild(el("label", { class: "chk" }, [
      enable,
      `${g.code} (点 ${g.point_id})`,
      el("span", { class: "small", style: "margin-left:auto" }, "锁定 "),
      lock,
    ]));
  }

  const groupRoot = document.getElementById("corr-groups");
  clear(groupRoot);
  for (const group of corrGroups()) {
    const cb = el("input", { type: "checkbox" });
    cb.checked = !!state.groupToggle[group];
    cb.addEventListener("change", () => {
      state.groupToggle[group] = cb.checked;
      renderPlanControls(plan);
    });
    gcpLabelMembers(group).forEach((id) => { state.corrToggle[id] = cb.checked; });
    groupRoot.appendChild(el("label", { class: "chk" }, [cb, `组 ${group}`]));
  }

  const corrRoot = document.getElementById("corr-list");
  clear(corrRoot);
  for (const c of state.data.correspondences) {
    const group = groupOf(c);
    const cb = el("input", { type: "checkbox" });
    cb.checked = !!state.corrToggle[c.id] && !!state.groupToggle[group];
    cb.disabled = !state.groupToggle[group];
    cb.addEventListener("change", () => { state.corrToggle[c.id] = cb.checked; });
    corrRoot.appendChild(el("label", { class: "chk" }, [
      cb,
      `#${c.id} 点${c.point_a_id} ↔ 点${c.point_b_id}`,
      el("span", { class: "small" }, ` (${group})`),
    ]));
  }

  const splitRoot = document.getElementById("split-controls");
  clear(splitRoot);
  if (plan.model_version === "strip_drift-v1") {
    for (const strip of state.data.strips) {
      const inp = el("input", { type: "text", style: "width:130px",
        placeholder: "如 0.5 或 0.33, 0.66" });
      inp.value = state.splits[strip.strip_uid] || "";
      inp.addEventListener("change", () => { state.splits[strip.strip_uid] = inp.value; });
      splitRoot.appendChild(el("div", { class: "row" }, [
        el("span", { style: "width:34px" }, strip.strip_uid), inp,
      ]));
    }
    splitRoot.appendChild(el("div", { class: "small" },
      "内部断点必须严格介于 0 与 1 之间；恰好在断点上的点只属于前一段。"));
  } else {
    splitRoot.appendChild(el("div", { class: "small" },
      "仅 strip_drift-v1 支持分段；先派生为漂移方案再分段。"));
  }
}

function gcpLabelMembers(group) {
  return state.data.correspondences
    .filter((c) => groupOf(c) === group)
    .map((c) => c.id);
}

function groupOf(c) {
  const pa = state.data.strips.find((st) => st.points.some((p) => p.id === c.point_a_id));
  const pb = state.data.strips.find((st) => st.points.some((p) => p.id === c.point_b_id));
  return pa && pb ? `${pa.strip_uid}:${pb.strip_uid}` : "?";
}

function collectChanges() {
  const disabled_gcp_ids = state.data.gcps
    .filter((g) => !state.gcpToggle[g.id] && !state.locked[g.id])
    .map((g) => g.id);
  const locked_gcp_ids = state.data.gcps
    .filter((g) => state.locked[g.id]).map((g) => g.id);
  const disabled_corr_ids = state.data.correspondences
    .filter((c) => !state.corrToggle[c.id]).map((c) => c.id);
  const disabled_corr_groups = corrGroups().filter((g) => !state.groupToggle[g]);
  const changes = {
    disabled_gcp_ids, locked_gcp_ids,
    disabled_corr_ids, disabled_corr_groups,
    outlier_sigma: parseFloat(document.getElementById("sigma").value),
    reject_outliers: document.getElementById("reject").checked,
  };
  const plan = state.plans.find((p) => p.id === state.selectedPlan);
  if (plan && plan.model_version === "strip_drift-v1") {
    const drift_breaks = {};
    for (const strip of state.data.strips) {
      const raw = (state.splits[strip.strip_uid] || "").trim();
      const inner = raw ? raw.split(",").map((x) => parseFloat(x.trim())) : [];
      if (inner.some((x) => Number.isNaN(x) || x <= 0 || x >= 1)) {
        throw new Error(`航带 ${strip.strip_uid} 的断点必须是 (0,1) 内的数字`);
      }
      if (new Set(inner).size !== inner.length) {
        throw new Error(`航带 ${strip.strip_uid} 存在重复断点`);
      }
      drift_breaks[strip.strip_uid] = [0, ...inner.slice().sort((a, b) => a - b), 1];
    }
    changes.drift_breaks = drift_breaks;
  }
  return changes;
}

async function forkFromControls() {
  let changes;
  try { changes = collectChanges(); }
  catch (e) { setBanner(e.message, true); return null; }
  const defaultName = `派生 ${state.selectedPlan}`;
  const name = document.getElementById("fork-name").value || defaultName;
  try {
    const plan = await api(`/api/plans/${state.selectedPlan}/fork`, {
      method: "POST",
      body: JSON.stringify({ name, changes }),
    });
    state.selectedPlan = plan.id;
    document.getElementById("fork-name").value = "";
    await refreshAll();
    return plan.id;
  } catch (e) { setBanner(e.message, true); return null; }
}

async function runSelectedPlan() {
  setBanner("求解中…");
  const targetId = state.selectedPlan;
  if (!targetId) { setBanner("请先选择方案", true); return; }
  try {
    const r = await api(`/api/plans/${targetId}/jobs`, { method: "POST" });
    await refreshAll();
    const fresh = state.plans.find((p) => p.id === targetId);
    const corrected = await api(`/api/plans/${targetId}/corrected`);
    renderResults(fresh, corrected, r);
  } catch (e) { setBanner(e.message, true); }
}

function canonicalConfigForCompare(cfg) {
  const keys = ["disabled_gcp_ids", "locked_gcp_ids", "disabled_corr_ids",
                "disabled_corr_groups", "outlier_sigma", "reject_outliers", "drift_breaks"];
  const out = {};
  for (const k of keys) out[k] = cfg[k];
  return out;
}

function setBanner(text, bad = false) {
  const b = document.getElementById("status-banner");
  clear(b);
  if (text) b.appendChild(el("div", { class: bad ? "error" : "candidates" }, text));
}

function clearResults(text) {
  clear(document.getElementById("results"));
  clear(document.getElementById("views"));
  document.getElementById("results").appendChild(el("div", { class: "small" }, text || ""));
}

function renderResults(plan, corrected, jobResponse) {
  clearResults("");
  const results = document.getElementById("results");
  const job = jobResponse ? jobResponse.job : null;
  const result = job ? job.result : lastResultOf(plan);

  const head = el("div", { class: "row", style: "justify-content:space-between" }, [
    el("strong", {}, `方案 #${plan.id} ${plan.name}`),
    el("span", { class: "small" },
      `${plan.model_version} · 规则 ${plan.rule_version} · RMS ${fmt(result ? result.rms_m : null, 4)} m`),
  ]);
  results.appendChild(head);
  if (jobResponse && jobResponse.replayed) {
    results.appendChild(el("div", { class: "small" }, "命中相同输入+规则指纹，返回已存作业（未重复产生审计事件）。"));
  }

  if (!result) { results.appendChild(el("div", { class: "small" }, "无求解结果。")); return; }

  if (result.status === "rank_deficient") {
    const box = el("div", { class: "error" });
    box.appendChild(el("div", {},
      `自由度不足 / 几何退化：秩 ${result.rank} / ${result.n_params}。未受约束方向 ${result.n_params - result.rank} 个，未使用正则化强行选择。`));
    for (const lbl of result.null_labels) {
      box.appendChild(el("div", {}, `未约束方向: ${lbl}`));
    }
    results.appendChild(box);

    const cand = el("div", { class: "candidates" }, [
      el("div", {}, "多个同样可行的候选（残差行空间相同）："),
    ]);
    for (const c of result.candidates) {
      cand.appendChild(el("div", { class: "small" },
        `• ${c.name}` + (c.weighted_residual_norm != null ? `（加权残差范数 ${fmt(c.weighted_residual_norm, 5)}）` : "")));
    }
    results.appendChild(cand);
  } else {
    results.appendChild(el("div", { class: "tag ok", style: "display:inline-block;margin:4px 0" },
      `收敛 ${result.status === "ok" ? "✓" : "✗"} · 秩 ${result.rank}/${result.n_params} · 迭代 ${result.iterations}`));
  }

  if (result.rejected_corr_ids && result.rejected_corr_ids.length) {
    results.appendChild(el("div", { class: "small", style: "color:var(--warn)" },
      `被拒（保留为证据）对应：#${result.rejected_corr_ids.join(", #")}`));
  }

  // Residual tables
  const tableWrap = el("div", { class: "grid3" });
  tableWrap.appendChild(residualTable("控制点残差 (m)", result.gcp_residuals, "point_id"));
  tableWrap.appendChild(residualTable("对应残差 (m)", result.corr_residuals, "corr_id"));
  results.appendChild(tableWrap);

  drawViews(corrected, result, plan);
}

function lastResultOf(plan) {
  // corrected view exists only for succeeded jobs; result fetched separately
  return null;
}

function residualTable(title, rows, idKey) {
  const card = el("div", { class: "card" });
  card.appendChild(el("div", { class: "name" }, title));
  const table = el("table");
  table.appendChild(el("thead", {}, el("tr", {}, [
    el("th", {}, "ID"), el("th", {}, "dx"), el("th", {}, "dy"), el("th", {}, "dz"),
    el("th", {}, "|r|"), el("th", {}, "状态"),
  ])));
  const body = el("tbody");
  for (const r of rows || []) {
    const bad = r.accepted === false;
    body.appendChild(el("tr", { style: bad ? "color:var(--warn)" : "" }, [
      el("td", {}, String(r[idKey])),
      el("td", {}, fmt(r.residual_xyz_m[0], 3)),
      el("td", {}, fmt(r.residual_xyz_m[1], 3)),
      el("td", {}, fmt(r.residual_xyz_m[2], 3)),
      el("td", {}, fmt(r.norm_m, 3)),
      el("td", {}, bad ? "已拒" : (r.locked ? "锁定" : "采用")),
    ]));
  }
  table.appendChild(body);
  const scroll = el("div", { class: "scroll" }, table);
  card.appendChild(scroll);
  return card;
}

function gatherSeries(corrected) {
  // Returns per-strip arrays of corrected points + raw local (both in metres).
  const rawByStrip = {};
  for (const strip of state.data.strips) {
    rawByStrip[strip.strip_uid] = strip.points.map((p) => ({
      id: p.id, xyz: p.xyz || [p.x * unitScale(), p.y * unitScale(), p.z * unitScale()],
    }));
  }
  return corrected.strips;
}

function unitScale() { return state.data.pointset.unit_scale_to_m; }

function bounds(points, axes) {
  let min = [Infinity, Infinity], max = [-Infinity, -Infinity];
  for (const p of points) {
    const q = p.corrected_xyz_m || p.xyz;
    for (let k = 0; k < 2; k++) {
      const v = q[axes[k]];
      if (v < min[k]) min[k] = v;
      if (v > max[k]) max[k] = v;
    }
  }
  if (!isFinite(min[0])) { min = [0, 0]; max = [1, 1]; }
  return [min, max];
}

function makeCanvas(host, title, width, height) {
  const wrap = el("div", { class: "card" }, [el("div", { class: "name" }, title)]);
  const canvas = el("canvas", { width, height });
  wrap.appendChild(canvas);
  host.appendChild(wrap);
  const ctx = canvas.getContext("2d");
  return { canvas, ctx, wrap };
}

function projectFn(canvas, min, max, axes, pad = 34) {
  const w = canvas.width - pad * 2;
  const h = canvas.height - pad * 2;
  const sx = w / Math.max(max[0] - min[0], 1e-9);
  const sy = h / Math.max(max[1] - min[1], 1e-9);
  return (q) => [pad + (q[axes[0]] - min[0]) * sx,
                canvas.height - pad - (q[axes[1]] - min[1]) * sy];
}

function drawAxes(ctx, canvas, project, min, max, axes, labels) {
  ctx.strokeStyle = "#22303e"; ctx.fillStyle = "#93a1b0"; ctx.font = "10px sans-serif";
  ctx.lineWidth = 1;
  for (let g = 0; g <= 4; g++) {
    const xVal = min[0] + (max[0] - min[0]) * g / 4;
    const [x] = project([xVal, min[1], 0]);
    ctx.beginPath(); ctx.moveTo(x, canvas.height - 28); ctx.lineTo(x, 28); ctx.stroke();
    ctx.fillText(xVal.toFixed(1), x - 10, canvas.height - 10);
    const yVal = min[1] + (max[1] - min[1]) * g / 4;
    const [, y] = project([min[0], yVal, 0]);
    ctx.beginPath(); ctx.moveTo(28, y); ctx.lineTo(canvas.width - 10, y); ctx.stroke();
    ctx.fillText(yVal.toFixed(1), 4, y + 3);
  }
  ctx.fillStyle = "#c9d4df";
  ctx.fillText(labels[0], canvas.width - 40, canvas.height - 10);
  ctx.fillText(labels[1], 6, 18);
}

function drawViews(corrected, result, plan) {
  const host = document.getElementById("views");
  clear(host);

  // Collect all corrected points across strips for common bounds.
  const strips = Object.entries(corrected.strips);
  const all = strips.flatMap(([uid, pts]) => pts);
  const width = Math.min(640, Math.floor(host.clientWidth || 640));

  // --- Plan view (x-y) ---
  const planView = makeCanvas(host, "平面图（校正后 x–y，米）", width, 320);
  const [pmin, pmax] = bounds(all, [0, 1]);
  padBounds(pmin, pmax);
  const projXY = projectFn(planView.canvas, pmin, pmax, [0, 1]);
  drawAxes(planView.ctx, planView.canvas, projXY, pmin, pmax, [0, 1], ["x (m)", "y (m)"]);
  strips.forEach(([uid, pts], si) => {
    drawStripPolyline(planView.ctx, pts, projXY, COLORS[si % COLORS.length]);
  });
  drawGcpPlan(planView.ctx, result, projXY);
  addLegend(planView.wrap, strips.map(([uid], i) => [COLORS[i % COLORS.length], uid]).concat([["#ff6b6b", "GCP"]]));

  // --- Elevation profile (s-z) per strip ---
  const prof = makeCanvas(host, "高程剖面（沿航带 s–z，米）", width, 260);
  drawProfile(prof, strips, result);

  // --- Residual histogram ---
  const hist = makeCanvas(host, "残差分布（|r| 米）", width, 220);
  drawHistogram(hist.ctx, hist.canvas, result, plan);
}

function padBounds(min, max) {
  for (let k = 0; k < 2; k++) {
    if (max[k] - min[k] < 1e-6) { max[k] += 1; min[k] -= 1; }
    const pad = (max[k] - min[k]) * 0.06;
    min[k] -= pad; max[k] += pad;
  }
}

function drawStripPolyline(ctx, points, project, color) {
  ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 1.6;
  ctx.beginPath();
  points.forEach((p, i) => {
    const [x, y] = project(p.corrected_xyz_m);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  for (const p of points) {
    const [x, y] = project(p.corrected_xyz_m);
    ctx.beginPath(); ctx.arc(x, y, 2.2, 0, Math.PI * 2); ctx.fill();
  }
}

function drawGcpPlan(ctx, result, project) {
  // GCP residual rows carry residual_xyz but not corrected coords directly;
  // plot control positions from imported data matched by point id.
  ctx.fillStyle = "#ff6b6b";
  const gcpByCode = {};
  for (const g of state.data.gcps) gcpByCode[g.point_id] = g;
  for (const r of (result.gcp_residuals || [])) {
    const g = gcpByCode[r.point_id];
    if (!g) continue;
    const q = [g.control_x, g.control_y, g.control_z].map((v) => v * 1.0);
    const [x, y] = project(q);
    ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "#ff6b6b"; ctx.strokeRect(x - 6, y - 6, 12, 12);
  }
}

function addLegend(wrap, entries) {
  const legend = el("div", { class: "legend" });
  for (const [color, label] of entries) {
    legend.appendChild(el("span", {}, [el("i", { style: `background:${color}` }), label]));
  }
  wrap.appendChild(legend);
}

function drawProfile(prof, strips, result) {
  const { ctx, canvas, wrap } = prof;
  const pad = 34;
  let zmin = Infinity, zmax = -Infinity;
  for (const [, pts] of strips) for (const p of pts) {
    const z = p.corrected_xyz_m[2];
    if (z < zmin) zmin = z; if (z > zmax) zmax = z;
  }
  if (!isFinite(zmin)) { zmin = 0; zmax = 1; }
  if (zmax - zmin < 1e-6) { zmin -= 1; zmax += 1; }
  const w = canvas.width - pad * 2, h = canvas.height - pad * 2;
  const sy = h / (zmax - zmin);
  const toPx = (s, z) => [pad + s * w, canvas.height - pad - (z - zmin) * sy];
  ctx.strokeStyle = "#22303e"; ctx.fillStyle = "#93a1b0"; ctx.font = "10px sans-serif";
  for (let g = 0; g <= 4; g++) {
    const z = zmin + (zmax - zmin) * g / 4;
    const [, y] = toPx(0, z);
    ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(canvas.width - pad, y); ctx.stroke();
    ctx.fillText(z.toFixed(2), 4, y + 3);
  }
  ctx.fillStyle = "#c9d4df"; ctx.fillText("s（沿航带归一化）", canvas.width - 120, canvas.height - 10);
  ctx.fillText("z (m)", 6, 16);

  strips.forEach(([uid, pts], si) => {
    ctx.strokeStyle = COLORS[si % COLORS.length];
    ctx.fillStyle = COLORS[si % COLORS.length];
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    pts.forEach((p, i) => {
      const [x, y] = toPx(p.s, p.corrected_xyz_m[2]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    // Mark drift break points (endpoint ownership: break points ride earlier segment).
    for (const p of pts) {
      const [x, y] = toPx(p.s, p.corrected_xyz_m[2]);
      ctx.beginPath(); ctx.arc(x, y, 2.2, 0, Math.PI * 2); ctx.fill();
    }
  });

  // Residual markers: GCP norm vertical bars colored by threshold.
  const sigma = parseFloat(document.getElementById("sigma").value) || 0.5;
  const gcpByPoint = {};
  for (const g of state.data.gcps) gcpByPoint[g.point_id] = g;
  for (const r of result.gcp_residuals || []) {
    const g = gcpByPoint[r.point_id]; if (!g) continue;
    const point = strips.flatMap(([, pts]) => pts).find((p) => p.id === r.point_id);
    if (!point) continue;
    const [x, y] = toPx(point.s, point.corrected_xyz_m[2]);
    ctx.fillStyle = r.norm_m > sigma ? "#ff6b6b" : "#5dd39e";
    ctx.beginPath(); ctx.arc(x, y - 8, 3.4, 0, Math.PI * 2); ctx.fill();
  }
  addLegend(wrap, strips.map(([uid], i) => [COLORS[i % COLORS.length], uid]).concat(
    [["#5dd39e", "GCP ≤σ"], ["#ff6b6b", "GCP >σ"]]));
}

function drawHistogram(ctx, canvas, result, plan) {
  const pad = 34;
  const norms = [];
  for (const r of (result.corr_residuals || [])) norms.push({ v: r.norm_m, accepted: r.accepted });
  for (const r of (result.gcp_residuals || [])) norms.push({ v: r.norm_m, accepted: true });
  if (!norms.length) {
    ctx.fillStyle = "#93a1b0"; ctx.fillText("无残差数据", pad, pad);
    return;
  }
  const maxV = Math.max(...norms.map((n) => n.v), 0.01);
  const bins = 20;
  const acc = new Array(bins).fill(0);
  const rej = new Array(bins).fill(0);
  for (const n of norms) {
    const idx = Math.min(bins - 1, Math.floor(n.v / (maxV + 1e-9) * bins));
    if (n.accepted) acc[idx] += 1; else rej[idx] += 1;
  }
  const w = canvas.width - pad * 2, h = canvas.height - pad * 2;
  const bw = w / bins;
  const maxCount = Math.max(...acc.map((a, i) => a + rej[i]), 1);
  ctx.fillStyle = "#93a1b0"; ctx.font = "10px sans-serif";
  ctx.fillText("0", pad - 8, canvas.height - pad + 4);
  ctx.fillText(maxV.toFixed(2), canvas.width - pad - 20, canvas.height - pad + 4);
  const sigma = parseFloat(document.getElementById("sigma").value) || 0.5;
  for (let i = 0; i < bins; i++) {
    const x = pad + i * bw;
    const ah = acc[i] / maxCount * h;
    const rh = rej[i] / maxCount * h;
    ctx.fillStyle = "#4db6ff";
    ctx.fillRect(x + 1, canvas.height - pad - ah - rh, bw - 2, ah);
    ctx.fillStyle = "#ffb454";
    ctx.fillRect(x + 1, canvas.height - pad - rh, bw - 2, rh);
  }
  if (sigma <= maxV) {
    const sx = pad + sigma / maxV * w;
    ctx.strokeStyle = "#ff6b6b"; ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(sx, pad); ctx.lineTo(sx, canvas.height - pad); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#ff6b6b"; ctx.fillText(`σ=${sigma}m`, sx + 3, pad + 10);
  }
  ctx.fillStyle = "#c9d4df"; ctx.fillText("残差范数 |r| (m)", canvas.width - 120, canvas.height - 10);
  ctx.fillText("数量", 6, 16);
}

function toggleCompare(planId) {
  const idx = state.compare.indexOf(planId);
  if (idx >= 0) state.compare.splice(idx, 1);
  else {
    if (state.compare.length >= 2) state.compare.shift();
    state.compare.push(planId);
  }
  renderPlanList();
  renderCompare();
}

async function renderCompare() {
  const root = document.getElementById("compare-panel");
  clear(root);
  if (state.compare.length === 0) {
    root.appendChild(el("span", { class: "small" }, "用每个方案卡片的“加入比较”选择两个方案。"));
    return;
  }
  if (state.compare.length === 1) {
    root.appendChild(el("span", { class: "small" }, "已选择 1 个，再选择一个进行并排比较。"));
    return;
  }
  const [a, b] = state.compare;
  const data = await api(`/api/plans/${a}/compare/${b}`);
  const grid = el("div", { class: "compare" });
  for (const side of ["plan_a", "plan_b"]) {
    const item = data[side];
    const cell = el("div", { class: "card" });
    cell.appendChild(el("div", { class: "name" }, `#${item.plan.id} ${item.plan.name}`));
    cell.appendChild(el("div", { class: "small" },
      `${item.plan.model_version} · 状态 ${item.result ? item.result.status : "未求解"}`));
    if (item.result) {
      cell.appendChild(el("div", { class: "small" },
        `RMS ${fmt(item.result.rms_m, 4)} m · 秩 ${item.result.rank}/${item.result.n_params}`));
      const cv = el("canvas", { width: 360, height: 220 });
      cell.appendChild(cv);
      drawMiniPlan(cv.getContext("2d"), cv, item.points.strips);
      const zc = el("canvas", { width: 360, height: 140 });
      cell.appendChild(zc);
      drawMiniProfile(zc.getContext("2d"), zc, item.points.strips);
    }
    grid.appendChild(cell);
  }
  root.appendChild(grid);
}

function drawMiniPlan(ctx, canvas, stripsObj) {
  const strips = Object.entries(stripsObj);
  const all = strips.flatMap(([, pts]) => pts);
  let xmin = Infinity, ymin = Infinity, xmax = -Infinity, ymax = -Infinity;
  for (const p of all) {
    const q = p.corrected_xyz_m;
    xmin = Math.min(xmin, q[0]); ymin = Math.min(ymin, q[1]);
    xmax = Math.max(xmax, q[0]); ymax = Math.max(ymax, q[1]);
  }
  const pad = 26;
  const sx = (canvas.width - pad * 2) / Math.max(xmax - xmin, 1e-9);
  const sy = (canvas.height - pad * 2) / Math.max(ymax - ymin, 1e-9);
  const proj = (q) => [pad + (q[0] - xmin) * sx, canvas.height - pad - (q[1] - ymin) * sy];
  ctx.fillStyle = "#0c1117"; ctx.fillRect(0, 0, canvas.width, canvas.height);
  strips.forEach(([uid, pts], i) => {
    ctx.strokeStyle = COLORS[i % COLORS.length]; ctx.fillStyle = COLORS[i % COLORS.length];
    ctx.beginPath();
    pts.forEach((p, j) => {
      const [x, y] = proj(p.corrected_xyz_m);
      if (j === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    for (const p of pts) {
      const [x, y] = proj(p.corrected_xyz_m);
      ctx.beginPath(); ctx.arc(x, y, 1.8, 0, Math.PI * 2); ctx.fill();
    }
  });
}

function drawMiniProfile(ctx, canvas, stripsObj) {
  const strips = Object.entries(stripsObj);
  let zmin = Infinity, zmax = -Infinity;
  for (const [, pts] of strips) for (const p of pts) {
    zmin = Math.min(zmin, p.corrected_xyz_m[2]);
    zmax = Math.max(zmax, p.corrected_xyz_m[2]);
  }
  const pad = 22;
  const w = canvas.width - pad * 2, h = canvas.height - pad * 2;
  const sy = h / Math.max(zmax - zmin, 1e-9);
  ctx.fillStyle = "#0c1117"; ctx.fillRect(0, 0, canvas.width, canvas.height);
  strips.forEach(([uid, pts], i) => {
    ctx.strokeStyle = COLORS[i % COLORS.length];
    ctx.beginPath();
    pts.forEach((p, j) => {
      const x = pad + p.s * w;
      const y = canvas.height - pad - (p.corrected_xyz_m[2] - zmin) * sy;
      if (j === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
}

async function showAudit() {
  const data = await api("/api/audit?limit=300");
  const win = el("div", {
    style: "position:fixed;inset:60px;background:var(--panel);border:1px solid var(--line);" +
           "border-radius:10px;padding:16px;overflow:auto;z-index:50",
  });
  win.appendChild(el("div", { class: "row", style: "justify-content:space-between" }, [
    el("strong", {}, "审计日志（事件键唯一，重放不重复）"),
    el("button", { onclick: () => win.remove() }, "关闭"),
  ]));
  const table = el("table");
  table.appendChild(el("thead", {}, el("tr", {}, [
    el("th", {}, "ID"), el("th", {}, "时间"), el("th", {}, "事件"),
    el("th", {}, "作业"), el("th", {}, "方案"), el("th", {}, "内容"),
  ])));
  const body = el("tbody");
  for (const e of data.events) {
    body.appendChild(el("tr", {}, [
      el("td", {}, String(e.id)), el("td", {}, e.created_at.slice(11, 23)),
      el("td", {}, e.event_type),
      el("td", {}, String(e.job_id ?? "-")), el("td", {}, String(e.plan_id ?? "-")),
      el("td", {}, JSON.stringify(e.payload)),
    ]));
  }
  table.appendChild(body);
  win.appendChild(table);
  document.body.appendChild(win);
}

async function publishSelected() {
  if (!state.selectedPlan) return;
  try {
    await api(`/api/plans/${state.selectedPlan}/publish`, { method: "POST" });
    setBanner("方案已发布（不可变）。", false);
    await refreshAll();
  } catch (e) { setBanner(e.message, true); }
}

function wire() {
  document.getElementById("btn-demo").addEventListener("click", () => loadDemo().catch((e) => setBanner(e.message, true)));
  document.getElementById("btn-new-plan").addEventListener("click", createPlan);
  document.getElementById("btn-fork").addEventListener("click", forkFromControls);
  document.getElementById("btn-run").addEventListener("click", runSelectedPlan);
  document.getElementById("btn-publish").addEventListener("click", publishSelected);
  document.getElementById("btn-compare").addEventListener("click", () => {
    if (state.selectedPlan) { toggleCompare(state.selectedPlan); }
  });
  document.getElementById("btn-audit").addEventListener("click", showAudit);
}
wire();
