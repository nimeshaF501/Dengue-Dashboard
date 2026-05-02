;

// ==================== GLOBALS ====================
let data = null;
let currentPage = 'home';
let currentModel = 'model1_base';
let currentHorizon = 1;
let selectedDistrict = 'all';
let dateSliderValue = 100;
let charts = {};
let maps = {};
let geoLayers = {};

const MODEL_KEYS = ['model1_base','model2_rf','model3_gb','model4_lstm','model5_mlp'];
const MODEL_NAMES = {
  model1_base: 'Bayesian Poisson GAM (Base)',
  model2_rf: 'GAM + Random Forest',
  model3_gb: 'GAM + Gradient Boosting',
  model4_lstm: 'GAM + LSTM',
  model5_mlp: 'GAM + MLP Neural Network'
};
const MODEL_SHORT = {
  model1_base: 'Base GAM', model2_rf: 'GAM+RF', model3_gb: 'GAM+GB',
  model4_lstm: 'GAM+LSTM', model5_mlp: 'GAM+MLP'
};
const MODEL_COLORS = {
  model1_base: '#3498DB', model2_rf: '#E74C3C', model3_gb: '#9B59B6',
  model4_lstm: '#1ABC9C', model5_mlp: '#F39C12'
};

// ==================== DATA LOADING ====================
async function loadData() {
  const errBox = document.getElementById('loadingError');
  try {
    const r = await fetch('model_outputs.json');
    if (!r.ok) throw new Error('HTTP ' + r.status + ' - model_outputs.json not found');
    data = await r.json();
    if (!data.meta || !data.model_outputs) throw new Error('Invalid JSON structure');
    document.getElementById('loadingScreen').style.display = 'none';
    document.getElementById('topHeader').style.display = 'flex';
    document.getElementById('filterBar').style.display = 'flex';
    initFilters();
    updateAllPages();
  } catch (e) {
    if (errBox) errBox.textContent = 'Error: ' + e.message + '\n\nMake sure model_outputs.json is in the same folder.';
  }
}

// ==================== THEME ====================
function toggleTheme() {
  const b = document.body;
  b.getAttribute('data-theme') === 'dark' ? b.removeAttribute('data-theme') : b.setAttribute('data-theme', 'dark');
  Object.values(charts).forEach(c => c && c.update());
}

function getChartColors() {
  const d = document.body.getAttribute('data-theme') === 'dark';
  return {
    text: d ? '#F1F5F9' : '#2C3E50',
    grid: d ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)',
    tooltipBg: d ? '#1E293B' : '#fff',
    tooltipText: d ? '#F1F5F9' : '#2C3E50'
  };
}

// ==================== PAGE SWITCHING ====================
function switchPage(page) {
  currentPage = page;
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.getElementById('page' + page.charAt(0).toUpperCase() + page.slice(1)).classList.add('active');
  const filterBar = document.getElementById('filterBar');
  if (page === 'analysis') filterBar.style.display = 'none';
  else filterBar.style.display = 'flex';
  setTimeout(() => updatePage(page), 50);
}

function updateAllPages() {
  updatePage('home');
}

function updatePage(page) {
  if (!data) return;
  if (page === 'home') renderHome();
  if (page === 'risk') renderRisk();
  if (page === 'analysis') renderAnalysis();
  if (page === 'performance') renderPerformance();
}

// ==================== FILTERS ====================
function initFilters() {
  const sel = document.getElementById('filterDistrict');
  if (!sel || !data || !data.meta || !data.meta.districts) return;
  sel.innerHTML = '<option value="all">All Districts</option>';
  data.meta.districts.forEach(d => {
    const opt = document.createElement('option');
    opt.value = d; opt.textContent = d;
    sel.appendChild(opt);
  });
}

function applyFilters() {
  selectedDistrict = document.getElementById('filterDistrict').value;
  currentModel = document.getElementById('filterModel').value;
  dateSliderValue = parseInt(document.getElementById('dateSlider').value);
  const label = document.getElementById('dateLabel');
  if (label) {
    if (dateSliderValue >= 100) label.textContent = 'All Years';
    else label.textContent = 'Last ' + dateSliderValue + '%';
  }
  updatePage(currentPage);
}

// ==================== CHOROPLETH ====================
function getChoroplethColor(value, min, max) {
  if (max <= min) return '#EBF5FB';
  const t = Math.min(Math.max((value - min) / (max - min), 0), 1);
  let r, g, b;
  if (t < 0.5) { const s = t * 2; r = 255; g = Math.round(230 - 180 * s); b = Math.round(150 - 150 * s); }
  else { const s = (t - 0.5) * 2; r = Math.round(255 - 55 * s); g = Math.round(50 - 50 * s); b = 0; }
  return 'rgb(' + r + ',' + g + ',' + b + ')';
}
function getRiskLevel(cases, avg) {
  const ratio = cases / Math.max(avg, 0.1);
  if (ratio > 2.0) return { label: 'Critical', class: 'risk-critical' };
  if (ratio > 1.5) return { label: 'High', class: 'risk-high' };
  if (ratio > 1.0) return { label: 'Moderate', class: 'risk-moderate' };
  return { label: 'Low', class: 'risk-low' };
}

function initMap(mapId) {
  if (maps[mapId]) return maps[mapId];
  const m = L.map(mapId, { center: [7.8731, 80.7718], zoom: 7, zoomControl: false, attributionControl: false });
  L.control.zoom({ position: 'bottomright' }).addTo(m);
  L.control.attribution({ position: 'bottomleft' }).addTo(m);
  L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy;CARTO', subdomains: 'abcd', maxZoom: 19
  }).addTo(m);
  maps[mapId] = m;
  return m;
}

function renderMapTo(mapId, modelKey, horizon) {
  const m = initMap(mapId);
  if (!data || !data.model_outputs || !data.model_outputs[modelKey]) return;
  const mout = data.model_outputs[modelKey];
  const forecasts = mout.forecasts || {};
  const values = [];
  Object.entries(forecasts).forEach(([d, fcList]) => {
    if (!Array.isArray(fcList)) return;
    fcList.filter(f => f && f.weeks_ahead === horizon).forEach(f => { values.push(f.predicted || 0); });
  });
  const minVal = values.length ? Math.min(...values) : 0;
  const maxVal = values.length ? Math.max(...values) : 100;
  if (geoLayers[mapId]) { m.removeLayer(geoLayers[mapId]); geoLayers[mapId] = null; }
  geoLayers[mapId] = L.geoJson(SL_GEOJSON, {
    style: function (feature) {
      const dname = feature.properties.name;
      const fcList = forecasts[dname] || [];
      const fc = fcList.filter(f => f && f.weeks_ahead === horizon);
      const pred = fc.length ? fc.reduce((s, f) => s + (f.predicted || 0), 0) / fc.length : 0;
      return { fillColor: getChoroplethColor(pred, minVal, maxVal), weight: 2, opacity: 1, color: '#555', dashArray: '', fillOpacity: 0.75 };
    },
    onEachFeature: function (feature, layer) {
      const dname = feature.properties.name;
      const fcList = forecasts[dname] || [];
      const fc = fcList.filter(f => f && f.weeks_ahead === horizon);
      const pred = fc.length ? fc.reduce((s, f) => s + (f.predicted || 0), 0) / fc.length : 0;
      const actual = fc.length ? fc.reduce((s, f) => s + (f.actual || 0), 0) / fc.length : 0;
      const risk = getRiskLevel(pred, actual || 1);
      layer.bindTooltip('<div class="leaflet-tooltip-custom"><strong>' + dname + '</strong><br>Predicted: <b style="color:' + (MODEL_COLORS[modelKey] || '#3498DB') + '">' + Math.round(pred) + '</b><br>Actual: <b>' + Math.round(actual) + '</b><br><span class="risk-badge ' + risk.class + '">' + risk.label + '</span></div>');
      layer.on({
        mouseover: function (e) { e.target.setStyle({ weight: 3, color: '#222', fillOpacity: 0.9 }); },
        mouseout: function (e) { if (geoLayers[mapId]) geoLayers[mapId].resetStyle(e.target); }
      });
    }
  }).addTo(m);
  m.fitBounds(geoLayers[mapId].getBounds(), { padding: [20, 20] });
}

// ==================== CHART UTILS ====================
function createChart(ctx, type, labels, datasets, options) {
  const c = getChartColors();
  return new Chart(ctx, {
    type: type,
    data: { labels: labels, datasets: datasets },
    options: Object.assign({
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { labels: { usePointStyle: true, padding: 12, font: { size: 11, weight: '500' }, color: c.text } },
        tooltip: { backgroundColor: c.tooltipBg, titleColor: c.tooltipText, bodyColor: c.text, borderColor: '#ccc', borderWidth: 1, padding: 10, cornerRadius: 6 }
      },
      scales: {
        x: { grid: { color: c.grid, drawBorder: false }, ticks: { color: c.text, font: { size: 10 }, maxTicksLimit: 14 } },
        y: { grid: { color: c.grid, drawBorder: false }, ticks: { color: c.text, font: { size: 10 } }, beginAtZero: true }
      },
      animation: { duration: 500, easing: 'easeInOutQuart' }
    }, options || {})
  });
}

// ==================== PAGE 1: HOME ====================
function renderHome() {
  if (!data) return;
  const kpis = data.kpis;
  document.getElementById('kpiTotalCases').textContent = Math.round(kpis.total_cases).toLocaleString();
  document.getElementById('kpiAvgMonth').textContent = kpis.avg_per_month.toFixed(0);
  document.getElementById('kpiAvgDistrict').textContent = kpis.avg_per_district.toFixed(0);
  document.getElementById('kpiPeakMonth').textContent = kpis.peak_month.label;
  document.getElementById('kpiPeakMonthSub').textContent = Math.round(kpis.peak_month.cases).toLocaleString() + ' cases';
  document.getElementById('kpiPeakDistrict').textContent = kpis.peak_district.name;
  document.getElementById('kpiPeakDistSub').textContent = Math.round(kpis.peak_district.cases).toLocaleString() + ' cases';
  document.getElementById('homeMapBadge').textContent = MODEL_SHORT[currentModel];
  renderMapTo('homeMap', currentModel, currentHorizon);
  updateHomeCharts();
}

function updateHomeCharts() {
  if (!data) return;
  const nat = data.national_weekly || [];
  const natLabels = nat.map(w => w.year + '-W' + w.week);
  const natActual = nat.map(w => w.cases);
  const modelSeries = {};
  MODEL_KEYS.forEach(mk => {
    const series = data.model_national_forecasts && data.model_national_forecasts[mk] ? data.model_national_forecasts[mk] : [];
    const weekMap = {};
    series.forEach(s => { weekMap[s.week] = s.predicted; });
    modelSeries[mk] = nat.map(w => weekMap[w.week] !== undefined ? weekMap[w.week] : null);
  });

  const tsDatasets = [{ label: 'Actual', data: natActual, borderColor: '#2C3E50', backgroundColor: 'transparent', borderWidth: 2, pointRadius: 0, tension: 0.3 }];
  MODEL_KEYS.forEach(mk => {
    tsDatasets.push({ label: MODEL_SHORT[mk], data: modelSeries[mk], borderColor: MODEL_COLORS[mk], backgroundColor: 'transparent', borderWidth: 1.5, pointRadius: 0, borderDash: [4, 2], tension: 0.3 });
  });

  if (charts.homeTimeSeries) { charts.homeTimeSeries.destroy(); charts.homeTimeSeries = null; }
  const ctx1 = document.getElementById('homeTimeSeriesChart');
  if (ctx1) charts.homeTimeSeries = createChart(ctx1.getContext('2d'), 'line', natLabels.slice(-104), tsDatasets.map(ds => ({ ...ds, data: ds.data.slice(-104) })));

  if (charts.homeMonthly) { charts.homeMonthly.destroy(); charts.homeMonthly = null; }
  const ctx2 = document.getElementById('homeMonthlyChart');
  if (ctx2) {
    charts.homeMonthly = createChart(ctx2.getContext('2d'), 'bar', data.monthly_time_series.labels, [{
      label: 'Monthly Cases', data: data.monthly_time_series.values, backgroundColor: 'rgba(52,152,219,0.7)', borderColor: '#3498DB', borderWidth: 1
    }]);
  }

  if (charts.homeDistrict) { charts.homeDistrict.destroy(); charts.homeDistrict = null; }
  const ctx3 = document.getElementById('homeDistrictChart');
  if (ctx3) {
    charts.homeDistrict = createChart(ctx3.getContext('2d'), 'bar', data.district_bar.labels, [{
      label: 'Total Cases', data: data.district_bar.values,
      backgroundColor: data.district_bar.values.map((_, i) => {
        const colors = ['#E74C3C', '#E67E22', '#F39C12', '#2ECC71', '#3498DB', '#9B59B6'];
        return colors[i % colors.length] + 'aa';
      }), borderWidth: 1
    }]);
  }
}

// ==================== PAGE 2: RISK MAP ====================
function renderRisk() {
  if (!data) return;
  document.getElementById('riskMapBadge').textContent = MODEL_SHORT[currentModel];
  renderMapTo('riskMap', currentModel, currentHorizon);
  const tbody = document.getElementById('riskTableBody');
  if (!tbody) return;
  tbody.innerHTML = '';
  const mout = data.model_outputs[currentModel];
  if (!mout || !mout.forecasts) return;
  const rows = [];
  Object.entries(mout.forecasts).forEach(([d, fcs]) => {
    if (!Array.isArray(fcs)) return;
    const fc = fcs.filter(f => f && f.weeks_ahead === currentHorizon);
    if (!fc.length) return;
    const pred = fc.reduce((s, f) => s + (f.predicted || 0), 0) / fc.length;
    const actual = fc.reduce((s, f) => s + (f.actual || 0), 0) / fc.length;
    const error = Math.abs(pred - actual);
    const risk = getRiskLevel(pred, actual || 1);
    const prov = (data.districts[d] && data.districts[d].province) || 'Unknown';
    rows.push({ district: d, province: prov, predicted: pred, actual: actual, error: error, risk: risk });
  });
  rows.sort((a, b) => b.predicted - a.predicted);
  rows.forEach(r => {
    const tr = document.createElement('tr');
    tr.innerHTML = '<td><strong>' + r.district + '</strong></td>' +
      '<td>' + r.province + '</td>' +
      '<td><strong style="color:' + MODEL_COLORS[currentModel] + '">' + Math.round(r.predicted) + '</strong></td>' +
      '<td>' + Math.round(r.actual) + '</td>' +
      '<td>' + r.error.toFixed(1) + '</td>' +
      '<td><span class="risk-badge ' + r.risk.class + '">' + r.risk.label + '</span></td>';
    tbody.appendChild(tr);
  });
}

// ==================== PAGE 3: ANALYSIS ====================
function renderAnalysis() {
  if (!data) return;
  const c = getChartColors();

  // Boxplot
  const bp = data.boxplot;
  const districts = data.meta.districts || Object.keys(bp);
  const boxData = districts.map(d => {
    const b = bp[d] || { min: 0, q1: 0, median: 0, q3: 0, max: 0 };
    return [b.min, b.q1, b.median, b.q3, b.max];
  });

  if (charts.boxplot) { charts.boxplot.destroy(); charts.boxplot = null; }
  const ctxBox = document.getElementById('boxplotChart');
  if (ctxBox) {
    charts.boxplot = new Chart(ctxBox.getContext('2d'), {
      type: 'bar',
      data: {
        labels: districts,
        datasets: [
          { label: 'Min-Q1', data: districts.map((d, i) => boxData[i][1] - boxData[i][0]), backgroundColor: 'rgba(200,200,200,0.5)', stack: 'stack1' },
          { label: 'Q1-Median', data: districts.map((d, i) => boxData[i][2] - boxData[i][1]), backgroundColor: 'rgba(52,152,219,0.5)', stack: 'stack1' },
          { label: 'Median-Q3', data: districts.map((d, i) => boxData[i][3] - boxData[i][2]), backgroundColor: 'rgba(46,204,113,0.5)', stack: 'stack1' },
          { label: 'Q3-Max', data: districts.map((d, i) => boxData[i][4] - boxData[i][3]), backgroundColor: 'rgba(231,76,60,0.5)', stack: 'stack1' }
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { labels: { color: c.text } } },
        scales: {
          x: { stacked: true, grid: { color: c.grid }, ticks: { color: c.text, font: { size: 9 } } },
          y: { stacked: true, grid: { color: c.grid }, ticks: { color: c.text } }
        }
      }
    });
  }

  // District actual
  if (charts.distActual) { charts.distActual.destroy(); charts.distActual = null; }
  const ctxAct = document.getElementById('distActualChart');
  if (ctxAct) {
    const dist = selectedDistrict !== 'all' ? selectedDistrict : (data.meta.districts[0] || 'Colombo');
    const hist = (data.districts[dist] && data.districts[dist].historical) || [];
    const labels = hist.slice(-52).map(h => 'W' + h.week);
    const values = hist.slice(-52).map(h => h.cases);
    charts.distActual = createChart(ctxAct.getContext('2d'), 'line', labels, [{
      label: dist + ' Actual', data: values, borderColor: '#E74C3C', backgroundColor: 'rgba(231,76,60,0.1)', fill: true, borderWidth: 2, pointRadius: 1
    }]);
  }

  // District models
  if (charts.distModels) { charts.distModels.destroy(); charts.distModels = null; }
  const ctxMod = document.getElementById('distModelsChart');
  if (ctxMod) {
    const dist = selectedDistrict !== 'all' ? selectedDistrict : (data.meta.districts[0] || 'Colombo');
    const modelDatasets = [];
    MODEL_KEYS.forEach(mk => {
      const mout = data.model_outputs[mk];
      if (!mout || !mout.forecasts || !mout.forecasts[dist]) return;
      const fcs = mout.forecasts[dist];
      const vals = fcs.slice(-52).map(f => f.predicted);
      modelDatasets.push({ label: MODEL_SHORT[mk], data: vals, borderColor: MODEL_COLORS[mk], backgroundColor: 'transparent', borderWidth: 1.5, pointRadius: 0, borderDash: [3, 2] });
    });
    charts.distModels = createChart(ctxMod.getContext('2d'), 'line',
      modelDatasets.length > 0 ? modelDatasets[0].data.map((_, i) => 'W' + (i + 1)) : [],
      modelDatasets
    );
  }

  // Outbreak table
  const obt = document.getElementById('outbreakTableBody');
  if (obt) {
    obt.innerHTML = '';
    const ob = data.outbreaks || {};
    const rows = Object.entries(ob).map(([d, info]) => ({
      district: d, threshold: info.threshold, count: info.outbreak_count,
      latest: info.outbreaks.length > 0 ? info.outbreaks[info.outbreaks.length - 1] : null
    })).sort((a, b) => b.count - a.count);
    rows.forEach(r => {
      const tr = document.createElement('tr');
      tr.innerHTML = '<td><strong>' + r.district + '</strong></td>' +
        '<td>' + r.threshold.toFixed(1) + '</td>' +
        '<td>' + r.count + '</td>' +
        '<td>' + (r.latest ? 'Week ' + r.latest.week + ', ' + r.latest.year + ' (' + Math.round(r.latest.cases) + ')' : 'None') + '</td>';
      obt.appendChild(tr);
    });
  }

  // ACF
  if (charts.acf) { charts.acf.destroy(); charts.acf = null; }
  const ctxAcf = document.getElementById('acfChart');
  if (ctxAcf && data.acf) {
    const lags = data.acf.lags.slice(0, 25);
    const vals = data.acf.values.slice(0, 25);
    charts.acf = createChart(ctxAcf.getContext('2d'), 'bar', lags.map(l => 'Lag ' + l), [{
      label: 'ACF', data: vals, backgroundColor: vals.map(v => Math.abs(v) > 0.2 ? 'rgba(231,76,60,0.7)' : 'rgba(52,152,219,0.5)'),
      borderColor: vals.map(v => Math.abs(v) > 0.2 ? '#E74C3C' : '#3498DB'), borderWidth: 1
    }], {
      plugins: { legend: { display: false } },
      scales: { y: { grid: { color: c.grid, drawBorder: false }, ticks: { color: c.text } } }
    });
  }

  // Stationarity
  const st = data.stationarity || {};
  document.getElementById('statMean').textContent = (st.national_mean || 0).toFixed(1);
  document.getElementById('statStd').textContent = (st.national_std || 0).toFixed(1);
  document.getElementById('statDiff').textContent = (st.mean_difference || 0).toFixed(1);
  document.getElementById('statWeeks').textContent = st.total_weeks || '--';
  const resEl = document.getElementById('statResult');
  if (resEl) {
    resEl.textContent = st.adf_result || 'Analysis pending';
    resEl.style.color = st.is_stationary ? 'var(--secondary)' : 'var(--emergency)';
  }
}

// ==================== PAGE 4: PERFORMANCE ====================
function renderPerformance() {
  if (!data) return;
  MODEL_KEYS.forEach((key, idx) => {
    const mout = data.model_outputs[key];
    if (!mout || !mout.overall) return;
    const ov = mout.overall;
    const r2el = document.getElementById('perfR2_' + (idx + 1));
    const rmsel = document.getElementById('perfRMSE_' + (idx + 1));
    if (r2el) r2el.textContent = 'R2 = ' + (ov.r2 !== undefined ? ov.r2.toFixed(3) : '--');
    if (rmsel) rmsel.textContent = 'RMSE: ' + (ov.rmse !== undefined ? ov.rmse.toFixed(1) : '--') + ' | MAE: ' + (ov.mae !== undefined ? ov.mae.toFixed(1) : '--');
  });

  const tbody = document.getElementById('perfComparisonBody');
  if (tbody) {
    tbody.innerHTML = '';
    const tr1 = document.createElement('tr');
    tr1.innerHTML = '<td><strong>Overall RMSE</strong></td>' + MODEL_KEYS.map(k => '<td>' + (data.model_outputs[k].overall.rmse !== undefined ? data.model_outputs[k].overall.rmse.toFixed(2) : '--') + '</td>').join('');
    tbody.appendChild(tr1);
    const tr2 = document.createElement('tr');
    tr2.innerHTML = '<td><strong>Overall MAE</strong></td>' + MODEL_KEYS.map(k => '<td>' + (data.model_outputs[k].overall.mae !== undefined ? data.model_outputs[k].overall.mae.toFixed(2) : '--') + '</td>').join('');
    tbody.appendChild(tr2);
    const tr3 = document.createElement('tr');
    const allR2 = MODEL_KEYS.map(k => data.model_outputs[k].overall.r2 || 0);
    const bestR2 = Math.max(...allR2);
    tr3.innerHTML = '<td><strong>Overall R2</strong></td>' + MODEL_KEYS.map(k => {
      const ov = data.model_outputs[k].overall;
      const isBest = ov.r2 === bestR2;
      return '<td class="' + (isBest ? 'best-metric' : '') + '">' + (ov.r2 !== undefined ? ov.r2.toFixed(3) : '--') + '</td>';
    }).join('');
    tbody.appendChild(tr3);
    for (let w = 1; w <= 4; w++) {
      const tr = document.createElement('tr');
      const wR2 = MODEL_KEYS.map(k => {
        const h = data.model_outputs[k].by_horizon || {};
        return (h[w] && h[w].r2 !== undefined) ? h[w].r2 : 0;
      });
      const bestWR2 = Math.max(...wR2);
      tr.innerHTML = '<td><strong>Week +' + w + ' R2</strong></td>' + MODEL_KEYS.map((k, i) => {
        const h = data.model_outputs[k].by_horizon || {};
        const r2 = (h[w] && h[w].r2 !== undefined) ? h[w].r2 : 0;
        return '<td class="' + (r2 === bestWR2 ? 'best-metric' : '') + '">' + r2.toFixed(3) + '</td>';
      }).join('');
      tbody.appendChild(tr);
    }
  }

  const modelKey = currentModel;
  document.getElementById('perfHorizonBadge').textContent = MODEL_SHORT[modelKey];
  const ht = document.getElementById('perfHorizonBody');
  if (ht) {
    ht.innerHTML = '';
    const mout = data.model_outputs[modelKey];
    if (mout && mout.by_horizon) {
      Object.entries(mout.by_horizon).sort((a, b) => parseInt(a[0]) - parseInt(b[0])).forEach(([w, m]) => {
        const rmse = m.rmse !== undefined ? m.rmse : 0;
        const mae = m.mae !== undefined ? m.mae : 0;
        const r2 = m.r2 !== undefined ? m.r2 : 0;
        const assess = r2 > 0.9 ? 'Excellent' : r2 > 0.8 ? 'Good' : r2 > 0.7 ? 'Moderate' : 'Poor';
        const badge = r2 > 0.9 ? 'risk-moderate' : r2 > 0.8 ? 'risk-low' : r2 > 0.7 ? 'risk-high' : 'risk-critical';
        const tr = document.createElement('tr');
        tr.innerHTML = '<td><strong>Week +' + w + '</strong></td>' +
          '<td>' + rmse.toFixed(2) + '</td>' +
          '<td>' + mae.toFixed(2) + '</td>' +
          '<td><strong>' + r2.toFixed(3) + '</strong></td>' +
          '<td><span class="risk-badge ' + badge + '">' + assess + '</span></td>';
        ht.appendChild(tr);
      });
    }
  }

  document.getElementById('perfMapBadge').textContent = MODEL_SHORT[modelKey];
  renderMapTo('perfMap', modelKey, currentHorizon);
}

function exportPage() {
  console.log('Export not yet implemented');
}

// ==================== INIT ====================
document.addEventListener('DOMContentLoaded', function () {
  loadData();
});
