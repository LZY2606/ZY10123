const state = { dataset: null, schemes: [], details: {}, selected: { left: '', right: '' }, weights: {}, disabledControls: new Set(), lockedControls: new Set(), disabledCorrs: new Set(), disabledGroups: new Set() };
const $ = (id) => document.getElementById(id);
const api = async (path, options = {}) => {
  const response = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...options });
  const data = await response.json();
  if (!response.ok) throw Object.assign(new Error(data.message || response.statusText), data);
  return data;
};

function esc(value) { return String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function fmt(value, digits=3) { return value === null || value === undefined ? '—' : Number(value).toFixed(digits); }
function jobId() { return 'job-' + (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random().toString(16).slice(2)); }

async function initialize() {
  $('jobId').value = jobId();
  $('refreshButton').onclick = loadDatasets;
  $('solveButton').onclick = solve;
  $('publishButton').onclick = publishCurrent;
  $('datasetSelect').onchange = loadDataset;
  $('modelSelect').onchange = renderModelDependentControls;
  $('leftScheme').onchange = event => selectScheme('left', event.target.value);
  $('rightScheme').onchange = event => selectScheme('right', event.target.value);
  await loadDatasets();
}

async function loadDatasets() {
  const data = await api('/api/datasets');
  $('datasetSelect').innerHTML = data.datasets.map(d => `<option value="${esc(d.dataset_id)}">${esc(d.name)}</option>`).join('');
  if (data.datasets.length) await loadDataset();
}

async function loadDataset() {
  const datasetId = $('datasetSelect').value;
  state.dataset = await api(`/api/datasets/${datasetId}`);
  const schemeData = await api(`/api/datasets/${datasetId}/schemes`);
  state.schemes = schemeData.schemes;
  state.weights = {};
  state.disabledControls = new Set();
  state.lockedControls = new Set();
  state.disabledCorrs = new Set();
  state.disabledGroups = new Set();
  $('jobId').value = jobId();
  $('schemeLabel').value = `${$('modelSelect').selectedOptions[0].text.split('：')[0]} 试验 ${new Date().toLocaleTimeString()}`;
  renderModelDependentControls();
  renderSchemeOptions();
}

function renderStrips() {
  $('splitInputs').innerHTML = state.dataset.strips.map(s => `
    <div class="split-row"><span>${esc(s.strip_id)} [${fmt(s.time_start,1)}, ${fmt(s.time_end,1)}]</span>
    <input data-strip="${esc(s.strip_id)}" type="number" step="0.01" placeholder="严格位于内部" /></div>`).join('');
}

function renderModelDependentControls() {
  renderStrips();
  renderControls();
  $('schemeLabel').value = `${$('modelSelect').selectedOptions[0].text.split('：')[0]} 试验 ${new Date().toLocaleTimeString()}`;
}

function renderControls() {
  const model = $('modelSelect').value;
  $('splitInputs').parentElement.style.display = model === 'rigid_v1' ? 'none' : 'grid';
  $('controlList').innerHTML = state.dataset.controls.map(c => {
    const checkedDisabled = state.disabledControls.has(c.control_id) ? 'checked' : '';
    const checkedLocked = state.lockedControls.has(c.control_id) ? 'checked' : '';
    return `<div class="evidence-row" data-id="${esc(c.control_id)}">
      <strong>${esc(c.control_id)}</strong><span>${esc(c.point_id)}</span>
      <label><input type="checkbox" data-action="disable-control" ${checkedDisabled}/>禁用</label>
      <label><input type="checkbox" data-action="lock-control" ${checkedLocked}/>锁定</label>
      <input class="weight" type="number" min="0" step="0.1" value="${fmt(c.weight,1)}" data-action="weight-control" data-key="control:${esc(c.control_id)}"/>
    </div>`;
  }).join('');
  $('correspondenceList').innerHTML = state.dataset.correspondences.map(c => {
    const disabled = state.disabledCorrs.has(c.correspondence_id) || state.disabledGroups.has(c.group_id);
    return `<div class="evidence-row ${disabled ? 'disabled' : ''}" data-id="${esc(c.correspondence_id)}">
      <strong>${esc(c.correspondence_id)}</strong><span>${esc(c.left_point_id)}↔${esc(c.right_point_id)}</span>
      <label><input type="checkbox" data-action="disable-corr" ${disabled ? 'checked' : ''}/>禁用</label>
      <label>${esc(c.group_id || '无组')} <input type="checkbox" data-action="disable-group" ${c.group_id && state.disabledGroups.has(c.group_id) ? 'checked' : ''} ${c.group_id ? '' : 'disabled'}/></label>
      <input class="weight" type="number" min="0" step="0.1" value="${fmt(c.weight,1)}" data-action="weight-corr" data-key="correspondence:${esc(c.correspondence_id)}"/>
    </div>`;
  }).join('');

  document.querySelectorAll('[data-action]').forEach(input => input.addEventListener('change', event => {
    const row = event.target.closest('.evidence-row');
    const id = row.dataset.id;
    if (event.target.dataset.action === 'disable-control') { event.target.checked ? state.disabledControls.add(id) : state.disabledControls.delete(id); renderControls(); }
    if (event.target.dataset.action === 'lock-control') { event.target.checked ? state.lockedControls.add(id) : state.lockedControls.delete(id); renderControls(); }
    if (event.target.dataset.action === 'disable-corr') { event.target.checked ? state.disabledCorrs.add(id) : state.disabledCorrs.delete(id); renderControls(); }
    if (event.target.dataset.action === 'disable-group') {
      const group = state.dataset.correspondences.find(c => c.correspondence_id === id)?.group_id;
      if (group) { event.target.checked ? state.disabledGroups.add(group) : state.disabledGroups.delete(group); renderControls(); }
    }
    if (event.target.dataset.action.startsWith('weight')) {
    const sourceList = event.target.dataset.action === 'weight-control' ? state.dataset.controls : state.dataset.correspondences;
    const sourceId = event.target.dataset.key.split(':')[1];
    const source = sourceList.find(item => item.control_id === sourceId || item.correspondence_id === sourceId);
    if (Number(event.target.value) === source.weight) delete state.weights[event.target.dataset.key];
    else state.weights[event.target.dataset.key] = Number(event.target.value);
  }
  }));
}

function collectPayload() {
  const splits = {};
  document.querySelectorAll('#splitInputs input[data-strip]').forEach(input => { if (input.value !== '') splits[input.dataset.strip] = Number(input.value); });
  const limitValue = $('residualLimit').value;
  return {
    job_id: $('jobId').value,
    dataset_id: $('datasetSelect').value,
    label: $('schemeLabel').value,
    model: $('modelSelect').value,
    parent_scheme_id: $('parentSelect').value || null,
    disabled_control_ids: [...state.disabledControls],
    locked_control_ids: [...state.lockedControls],
    disabled_correspondence_ids: [...state.disabledCorrs],
    disabled_correspondence_group_ids: [...state.disabledGroups],
    weights: state.weights,
    segment_splits: splits,
    residual_limit_m: limitValue === '' ? null : Number(limitValue),
    max_iterations: 20,
    step_tolerance: 1e-11
  };
}

async function solve() {
  const payload = collectPayload();
  $('solveStatus').textContent = '求解中…'; $('diagnostics').textContent = '';
  try {
    const result = await api('/api/schemes/solve', { method: 'POST', body: JSON.stringify(payload) });
    if (result.error) {
      $('diagnostics').textContent = JSON.stringify(result.error, null, 2);
      $('solveStatus').textContent = '秩不足或约束冲突；未创建方案。';
    } else {
      $('solveStatus').textContent = `已创建 ${result.scheme_id}`;
      await loadDataset();
      state.selected.left = result.scheme_id;
      await renderSchemeOptions();
    }
  } catch (error) {
    $('solveStatus').textContent = '请求失败';
    $('diagnostics').textContent = JSON.stringify(error, null, 2);
  }
}

async function publishCurrent() {
  const id = state.selected.left || state.schemes[0]?.scheme_id;
  if (!id) return;
  await api('/api/schemes/publish', { method: 'POST', body: JSON.stringify({ job_id: jobId(), scheme_id: id }) });
  await loadDataset();
  state.selected.left = id;
  delete state.details[id];
  renderSchemeOptions();
}

function renderSchemeOptions() {
  const options = state.schemes.map(s => `<option value="${esc(s.scheme_id)}">${esc(s.label)} (${esc(s.model_version)})</option>`).join('');
  $('parentSelect').innerHTML = '<option value=""></option>' + options;
  $('leftScheme').innerHTML = options;
  $('rightScheme').innerHTML = '<option value=""></option>' + options;
  const left = state.selected.left || state.schemes[0]?.scheme_id || '';
  const right = state.selected.right || state.schemes[1]?.scheme_id || '';
  $('leftScheme').value = left; $('rightScheme').value = right;
  selectScheme('left', left); selectScheme('right', right);
}

async function selectScheme(side, id) {
  state.selected[side] = id;
  if (id && !state.details[id]) state.details[id] = await api(`/api/schemes/${id}`);
  renderComparison();
}

function residualValues(detail) {
  const result = detail.scheme.result;
  return Object.entries(result.residuals)
    .filter(([, r]) => r.status === 'accepted' || r.status === 'rejected' || r.status === 'locked')
    .map(([id, r]) => ({ id, ...r }));
}

function renderComparison() {
  const grid = $('schemeGrid');
  grid.innerHTML = '';
  [['A', state.selected.left], ['B', state.selected.right]].forEach(([side,id]) => {
    if (!id || !state.details[id]) { grid.insertAdjacentHTML('beforeend', `<article class="scheme-card"><h3>方案 ${side}</h3><p>选择方案以并排比较。</p></article>`); return; }
    const detail = state.details[id];
    const node = $('schemeTemplate').content.firstElementChild.cloneNode(true);
    const scheme = detail.scheme, result = scheme.result;
    node.querySelector('h3').textContent = scheme.label;
    const badge = node.querySelector('.badge'); badge.textContent = scheme.status; badge.className = 'badge ' + scheme.status;
    node.querySelector('.meta').innerHTML = `
      <dt>模型</dt><dd>${esc(scheme.model_version)}</dd>
      <dt>方向</dt><dd>${esc(result.transform.direction)}：${esc(result.transform.equation)}</dd>
      <dt>秩/自由度</dt><dd>${result.rank}/${result.degrees_of_freedom}</dd>
      <dt>拒识</dt><dd>${esc(result.rejected_correspondence_ids.join(', ') || '无')}</dd>
      <dt>单位</dt><dd>${esc(state.dataset.dataset.common_length_unit)}；残差为 ${esc(state.dataset.dataset.common_length_unit)}</dd>
      <dt>控制几何</dt><dd>秩 ${result.control_geometry_rank}${result.control_geometry_degenerate ? '；退化' : ''}</dd>${result.warnings.length ? `<dt>警告</dt><dd>${esc(result.warnings.join('；'))}</dd>` : ''}`;
    drawScatter(node.querySelector('.plan'), detail.transformed_points, 0, 1);
    drawScatter(node.querySelector('.profile'), detail.transformed_points, 0, 2);
    drawHistogram(node.querySelector('.histogram'), residualValues(detail));
    node.querySelector('.residual-table').innerHTML = renderResiduals(detail);
    grid.appendChild(node);
  });
}

function project(value, min, max, start, length) { return start + ((value-min)/(max-min || 1))*length; }
function drawScatter(svg, points, xi, yi) {
  const width = 280, height = svg.viewBox.baseVal.height, margin=22;
  const xs = points.map(p=>p.common[xi]), ys = points.map(p=>p.common[yi]);
  const minX=Math.min(...xs)-.2, maxX=Math.max(...xs)+.2, minY=Math.min(...ys)-.2, maxY=Math.max(...ys)+.2;
  const palette = ['#1769aa', '#4c9bd6', '#d77a35', '#e8a25c', '#5a8f3c', '#8a6ab8'];
  const colorFor = (strip, segment = 0) => {
    const hash = [...`${strip}#${segment}`].reduce((sum, char) => sum + char.charCodeAt(0), 0);
    return palette[hash % palette.length];
  };
  svg.innerHTML = `<line x1="${margin}" y1="${height-margin}" x2="${width-4}" y2="${height-margin}" stroke="#91a0ad"/><line x1="${margin}" y1="4" x2="${margin}" y2="${height-margin}" stroke="#91a0ad"/>`;
  points.forEach(p => {
    const x = project(p.common[xi], minX, maxX, margin, width-margin-8);
    const y = height-margin - project(p.common[yi], minY, maxY, 0, height-margin-8);
    svg.insertAdjacentHTML('beforeend', `<circle cx="${x}" cy="${y}" r="3" fill="${colorFor(p.strip_id, p.segment_index || 0)}"><title>${esc(p.point_id)} ${esc(p.strip_id)} #${p.segment_index || 0}</title></circle>`);
  });
}
function drawHistogram(svg, residuals) {
  const accepted = residuals.filter(r => r.status === 'accepted' || r.status === 'locked').map(r=>r.norm);
  const rejected = residuals.filter(r => r.status === 'rejected').map(r=>r.norm);
  const max = Math.max(0.1, ...residuals.map(r=>r.norm));
  const bins = new Array(10).fill(0);
  accepted.forEach(v => bins[Math.min(9, Math.floor(v/max*10))]++);
  const width=280,height=140,margin=22, barWidth=(width-margin-8)/bins.length;
  const binMax=Math.max(1,...bins);
  svg.innerHTML = `<line x1="${margin}" y1="${height-margin}" x2="${width-4}" y2="${height-margin}" stroke="#91a0ad"/>`;
  bins.forEach((count,i)=>{
    const h = count/binMax*(height-margin-8);
    svg.insertAdjacentHTML('beforeend', `<rect x="${margin+i*barWidth}" y="${height-margin-h}" width="${barWidth-1}" height="${h}" fill="#1769aa"/>`);
  });
  rejected.forEach(v => {
    const x=margin+(v/max)*(width-margin-8);
    svg.insertAdjacentHTML('beforeend', `<line x1="${x}" y1="5" x2="${x}" y2="${height-margin}" stroke="#c2412d" stroke-dasharray="3 2"><title>拒识 ${fmt(v)}m</title></line>`);
  });
}
function renderResiduals(detail) {
  return `<div class="residual-row"><strong>ID</strong><span>类型</span><span>状态</span><span>权重</span><span>残差 m</span></div>` +
    Object.entries(detail.scheme.result.residuals).map(([id,r]) =>
      `<div class="residual-row ${r.status}"><span>${esc(id)}</span><span>${r.kind}</span><span>${r.status}</span><span>${fmt(r.weight,1)}</span><span>${fmt(r.norm,4)}</span></div>`).join('');
}

initialize().catch(error => { $('solveStatus').textContent = '初始化失败'; $('diagnostics').textContent = error.stack; });
