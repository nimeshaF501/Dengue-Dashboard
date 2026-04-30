#!/usr/bin/env python3
"""
Sri Lanka Dengue Forecasting - 5 Model Ensemble
Run: python run_models.py -> generates model_outputs.json
"""
import os, sys, json, pickle, warnings, random
from collections import deque, defaultdict
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.special import gammaln
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
warnings.filterwarnings('ignore')
SEED = 42
random.seed(SEED); np.random.seed(SEED)

# ============== CONFIG ==============
DATA_CSV = "srilanka_dengue_data.csv"  # <-- Your CSV file
SHAPEFILE = "C:/Users/FERNANDO/OneDrive - Colombo International Nautical & Engineering College/Desktop/Lap/MSC/Research/Data Sets/lka_adm_20220816_shp/lka_admbnda_adm2_slsd_20220816.shp"

OUTPUT_JSON = "model_outputs.json"

# ============== HELPERS ==============
def bounded_r2(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 2: return 0.0
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    if ss_tot < 1e-10: return 0.0
    return max(0.0, min(1.0, 1 - ss_res / ss_tot))

def neg_log_post(beta, X, y, ps, lf):
    eta = X @ beta; mu = np.exp(eta)
    ll = np.sum(y * eta - mu) - lf
    lp = -0.5 * np.sum((beta / ps)**2)
    return -(ll + lp)

def predict_base(X, beta):
    return np.exp(np.column_stack([np.ones(X.shape[0]), X]) @ beta)

# ============== 1) LOAD DATA ==============
print("="*60 + "\nDENGUE 5-MODEL ENSEMBLE\n" + "="*60)
if not os.path.exists(DATA_CSV):
    print(f"[ERROR] {DATA_CSV} not found"); sys.exit(1)

df = pd.read_csv(DATA_CSV, parse_dates=['start.date'])
dmap = {'NuwaraEliya': 'Nuwara Eliya', 'Hambanthota': 'Hambantota', 'Kalmune': 'Ampara'}
df['district'] = df['district'].map(lambda x: dmap.get(x, x))

d2p = {'Colombo': 'Western', 'Gampaha': 'Western', 'Kalutara': 'Western',
    'Kandy': 'Central', 'Matale': 'Central', 'Nuwara Eliya': 'Central',
    'Galle': 'Southern', 'Hambantota': 'Southern', 'Matara': 'Southern',
    'Jaffna': 'Northern', 'Kilinochchi': 'Northern', 'Mannar': 'Northern',
    'Vavuniya': 'Northern', 'Mullaitivu': 'Northern',
    'Batticaloa': 'Eastern', 'Ampara': 'Eastern', 'Trincomalee': 'Eastern',
    'Kurunegala': 'North Western', 'Puttalam': 'North Western',
    'Anuradhapura': 'North Central', 'Polonnaruwa': 'North Central',
    'Badulla': 'Uva', 'Monaragala': 'Uva',
    'Ratnapura': 'Sabaragamuwa', 'Kegalle': 'Sabaragamuwa'}

# ============== 2) SPATIAL WEIGHTS ==============
print("[1/5] Spatial adjacency...")
district_order = sorted(df['district'].unique())
if 'Kalmune' in district_order: district_order.remove('Kalmune')
n = len(district_order)

# Use shapefile if provided, otherwise equal weights
if SHAPEFILE and os.path.exists(SHAPEFILE):
    try:
        import geopandas as gpd
        gdf = gpd.read_file(SHAPEFILE)
        gdf = gdf[['ADM2_EN', 'geometry']].rename(columns={'ADM2_EN': 'district_shp'})
        gdf['district_shp'] = gdf['district_shp'].str.strip().replace({'NuwaraEliya': 'Nuwara Eliya'})
        gdf = gdf[gdf['district_shp'].isin(district_order)].reset_index(drop=True)
        sindex = gdf.sindex
        neigh = {i: set() for i in gdf.index}
        for i, geom in gdf.geometry.items():
            for j in list(sindex.intersection(geom.bounds)):
                if i != j and geom.touches(gdf.geometry.iloc[j]):
                    neigh[i].add(j); neigh[j].add(i)
        district_order = list(gdf['district_shp'])
        A = np.zeros((n, n), float)
        for i, js in neigh.items():
            for j in js: A[i, j] = 1.0
        rs = A.sum(axis=1)
        W = A.copy()
        for i in range(n):
            if rs[i] > 0: W[i, :] = A[i, :] / rs[i]
        print("  Using shapefile spatial weights")
    except Exception as e:
        print(f"  Shapefile failed ({e}), using equal weights")
        W = np.ones((n, n), float) / n
else:
    W = np.ones((n, n), float) / n

d2i = {d: i for i, d in enumerate(district_order)}

# ============== 3) PANEL DATA ==============
print("[2/5] Panel data...")
df['week'] = df['start.date'].dt.isocalendar().week
df['year'] = df['start.date'].dt.year
start_date = df['start.date'].min()
df['week_index'] = ((df['start.date'] - start_date).dt.days // 7).astype(int)

pivot = df.pivot_table(index='week_index', columns='district', values='cases', aggfunc='sum').fillna(0)
pivot = pivot.reindex(columns=district_order, fill_value=0)
for i in range(1, 6): globals()[f'pl{i}'] = pivot.shift(i).fillna(0)
sl1 = pl1.values @ W.T; sl2 = pl2.values @ W.T

records = []
for idx, w in enumerate(pivot.index):
    for j, d in enumerate(district_order):
        records.append({'week_index': int(w), 'district': d, 'cases': float(pivot.iloc[idx, j]),
            'cases_lag1': float(pl1.iloc[idx, j]), 'cases_lag2': float(pl2.iloc[idx, j]),
            'cases_lag3': float(pl3.iloc[idx, j]), 'cases_lag4': float(pl4.iloc[idx, j]),
            'cases_lag5': float(pl5.iloc[idx, j]),
            'spatial_lag1': float(sl1[idx, j]), 'spatial_lag2': float(sl2[idx, j])})
panel = pd.DataFrame(records)
panel['Province'] = panel['district'].map(lambda d: d2p.get(d, 'Unknown'))

# Feature engineering
panel['sin52'] = np.sin(2 * np.pi * (panel['week_index'] % 52) / 52)
panel['cos52'] = np.cos(2 * np.pi * (panel['week_index'] % 52) / 52)
panel['trend'] = panel['week_index'] / panel['week_index'].max()
for i in range(1, 6): panel[f'log_cases_lag{i}'] = np.log1p(panel[f'cases_lag{i}'])
panel['log_spatial_lag1'] = np.log1p(np.maximum(panel['spatial_lag1'], 0))
panel['log_spatial_lag2'] = np.log1p(np.maximum(panel['spatial_lag2'], 0))
panel['rolling_mean_3'] = (panel['log_cases_lag1'] + panel['log_cases_lag2'] + panel['log_cases_lag3']) / 3
panel['lag1_x_lag2'] = panel['log_cases_lag1'] * panel['log_cases_lag2']
panel['spatial_x_lag1'] = panel['log_spatial_lag1'] * panel['log_cases_lag1']
panel['sin52_x_log_cases_lag1'] = panel['sin52'] * panel['log_cases_lag1']
panel['cos52_x_log_cases_lag1'] = panel['cos52'] * panel['log_cases_lag1']
panel['sin52_sq'] = panel['sin52']**2; panel['cos52_sq'] = panel['cos52']**2
panel['log_cases_lag1_sq'] = panel['log_cases_lag1']**2
panel['log_spatial_lag1_sq'] = panel['log_spatial_lag1']**2
panel = panel.dropna()

prov_d = pd.get_dummies(panel['Province'], prefix='prov').astype(int)
panel = pd.concat([panel, prov_d], axis=1)

# Train/validation split
vs = pd.Timestamp("2025-01-01")
vsw = int(((vs - start_date).days // 7))
train = panel[panel['week_index'] < vsw].copy()
val = panel[panel['week_index'] >= vsw].copy()
cap_max = max(1.5 * train['cases'].quantile(0.99), 100)

fc = ['sin52', 'cos52', 'trend'] + [f'log_cases_lag{i}' for i in range(1, 6)] + \
    ['log_spatial_lag1', 'log_spatial_lag2', 'rolling_mean_3', 'sin52_sq', 'cos52_sq',
     'log_cases_lag1_sq', 'log_spatial_lag1_sq', 'lag1_x_lag2', 'spatial_x_lag1',
     'sin52_x_log_cases_lag1', 'cos52_x_log_cases_lag1'] + list(prov_d.columns)
fc = [c for c in fc if c in panel.columns]

Xr = train[fc].values.astype(np.float64)
y = train['cases'].values.astype(np.float64)
Xi = np.column_stack([np.ones(len(Xr)), Xr])
mask = ~np.isnan(Xr).any(axis=1)
Xr, y, Xi = Xr[mask], y[mask], Xi[mask]
print(f"  Train: {len(y)} samples, {len(fc)} features, cap={cap_max:.0f}")

# ============== 4) BASE GAM ==============
print("[3/5] Model 1: Bayesian GAM...")
lf = np.sum(gammaln(y + 1))
r = minimize(neg_log_post, np.zeros(Xi.shape[1]), args=(Xi, y, 5.0, lf),
             method='L-BFGS-B', options={'maxiter': 2000, 'disp': False})
beta = r.x
print(f"  Converged: {r.success}, iters={r.nit}")

bp_train = predict_base(Xr, beta)
tc = train.iloc[mask].copy()
tc['residual'] = y - bp_train

# ============== 5) CORRECTION MODELS ==============
print("[4/5] Correction models...")
sc = StandardScaler(); Xrs = sc.fit_transform(Xr)

def tcorr(cls, kw, scaled=False):
    md = {}; Xu = Xrs if scaled else Xr
    for d in district_order:
        m = tc['district'] == d; Xd, yd = Xu[m], tc.loc[m, 'residual'].values
        if len(Xd) >= (50 if scaled else 10): md[d] = cls(**kw).fit(Xd, yd)
        else: md[d] = None
    return md

drf = tcorr(RandomForestRegressor, {'n_estimators': 100, 'random_state': SEED, 'n_jobs': -1})
dgb = tcorr(GradientBoostingRegressor, {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1, 'random_state': SEED})
dmlp = tcorr(MLPRegressor, {'hidden_layer_sizes': (50, 25), 'activation': 'relu', 'solver': 'adam', 'max_iter': 500, 'random_state': SEED, 'early_stopping': True}, scaled=True)

# LSTM (PyTorch)
dlstm = {}; TORCH = False
try:
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH = True
    class LSTMNet(nn.Module):
        def __init__(self): super().__init__(); self.lstm = nn.LSTM(1, 32, batch_first=True); self.fc = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
        def forward(self, x): out, _ = self.lstm(x); return self.fc(out[:, -1, :])

    LW = 52
    for d in tqdm(district_order, desc="  LSTM"):
        rs = tc[tc['district'] == d]['residual'].values
        if len(rs) <= LW: dlstm[d] = None; continue
        Xs, Ys = [], []
        for i in range(len(rs) - LW):
            Xs.append(rs[i:i+LW].reshape(-1, 1)); Ys.append(rs[i+LW])
        Xs, Ys = np.array(Xs, np.float32), np.array(Ys, np.float32).reshape(-1, 1)
        ds = TensorDataset(torch.tensor(Xs), torch.tensor(Ys))
        dl = DataLoader(ds, batch_size=32, shuffle=True)
        model = LSTMNet(); opt = torch.optim.Adam(model.parameters(), lr=0.001)
        model.train()
        for _ in range(30):
            for xb, yb in dl:
                opt.zero_grad(); nn.MSELoss()(model(xb), yb).backward(); opt.step()
        dlstm[d] = model
    print("  LSTM trained")
except ImportError:
    print("  PyTorch not found - LSTM will equal base predictions")

# ============== 6) ROLLING FORECAST ==============
print("[5/5] Rolling forecast...")
vw = sorted(val['week_index'].unique()); mw = max(vw)
act = {(r['week_index'], r['district']): r['cases'] for _, r in panel.iterrows()}
res = {f'model{i}_{k}': [] for i, k in enumerate(['base', 'rf', 'gb', 'lstm', 'mlp'], 1)}

for ow in tqdm(vw[:-4], desc="  Rolling"):
    ch = {}
    for d in district_order:
        h = [act.get((w, d), 0) if w >= 0 else 0 for w in range(ow - 5, ow + 1)]
        ch[d] = deque(h, maxlen=6)
    cv = np.array([act.get((ow, d), 0) for d in district_order])

    for wh in [1, 2, 3, 4]:
        fw = ow + wh
        if fw > mw: continue
        bd = val[val['week_index'] == fw].copy()
        if bd.empty: continue

        l1, l2, l3, l4, l5 = cv, *[np.array([ch[d][-i] for d in district_order]) for i in range(2, 6)]
        s1, s2 = l1 @ W.T, l2 @ W.T

        rows = []
        for d in bd['district']:
            ix = d2i.get(d)
            v = (l1[ix], l2[ix], l3[ix], l4[ix], l5[ix]) if ix is not None else (0, 0, 0, 0, 0)
            sv = (s1[ix], s2[ix]) if ix is not None else (0, 0)
            lg = {f'log{i}': np.log1p(max(v[i-1], 0)) for i in range(1, 6)}
            ls1, ls2 = np.log1p(max(sv[0], 0)), np.log1p(max(sv[1], 0))
            s52 = np.sin(2*np.pi*(fw%52)/52); c52 = np.cos(2*np.pi*(fw%52)/52)
            rows.append({'sin52': s52, 'cos52': c52, 'trend': fw / panel['week_index'].max(),
                'log_cases_lag1': lg['log1'], 'log_cases_lag2': lg['log2'], 'log_cases_lag3': lg['log3'],
                'log_cases_lag4': lg['log4'], 'log_cases_lag5': lg['log5'],
                'log_spatial_lag1': ls1, 'log_spatial_lag2': ls2,
                'rolling_mean_3': (lg['log1']+lg['log2']+lg['log3'])/3,
                'lag1_x_lag2': lg['log1']*lg['log2'], 'spatial_x_lag1': ls1*lg['log1'],
                'sin52_sq': s52**2, 'cos52_sq': c52**2, 'log_cases_lag1_sq': lg['log1']**2,
                'log_spatial_lag1_sq': ls1**2, 'sin52_x_log_cases_lag1': s52*lg['log1'],
                'cos52_x_log_cases_lag1': c52*lg['log1'],
                **{p: 1 if p.replace('prov_','') == d2p.get(d,'') else 0 for p in prov_d.columns}})

        fdf = pd.DataFrame(rows)
        for c in fc:
            if c not in fdf.columns: fdf[c] = 0
        Xfc = fdf[fc].values.astype(np.float64)
        Xfc_s = sc.transform(Xfc)
        bp = np.clip(predict_base(Xfc, beta), 0, cap_max)

        # Model 2: RF
        rp = bp.copy()
        for i, d in enumerate(bd['district']):
            if drf.get(d): rp[i] += drf[d].predict(Xfc[i].reshape(1, -1))[0]
        rp = np.clip(rp, 0, cap_max)

        # Model 3: GB
        gp = bp.copy()
        for i, d in enumerate(bd['district']):
            if dgb.get(d): gp[i] += dgb[d].predict(Xfc[i].reshape(1, -1))[0]
        gp = np.clip(gp, 0, cap_max)

        # Model 5: MLP
        mp = bp.copy()
        for i, d in enumerate(bd['district']):
            if dmlp.get(d): mp[i] += dmlp[d].predict(Xfc_s[i].reshape(1, -1))[0]
        mp = np.clip(mp, 0, cap_max)

        # Model 4: LSTM
        lp = bp.copy()
        if TORCH:
            for i, d in enumerate(bd['district']):
                if dlstm.get(d):
                    rh = list(tc[tc['district'] == d]['residual'].values)
                    if len(rh) < LW: rh = [0.0] * (LW - len(rh)) + rh
                    seq = np.array(rh[-LW:]).reshape(1, LW, 1).astype(np.float32)
                    with torch.no_grad(): lp[i] += dlstm[d](torch.tensor(seq)).numpy()[0, 0]
        lp = np.clip(lp, 0, cap_max)

        for mi, preds in enumerate([bp, rp, gp, lp, mp], 1):
            key = f'model{mi}_' + ['base', 'rf', 'gb', 'lstm', 'mlp'][mi - 1]
            for j, (_, r) in enumerate(bd.iterrows()):
                res[key].append({'origin_week': int(ow), 'forecast_week': int(fw),
                    'weeks_ahead': int(wh), 'district': r['district'],
                    'actual_cases': float(r['cases']), 'predicted_cases': float(preds[j])})

        nc = np.zeros(n)
        for i, d in enumerate(bd['district']):
            nc[d2i[d]] = bp[i]; ch[d].append(bp[i])
        cv = nc

# ============== 7) METRICS ==============
print("\n" + "="*60 + "\nPERFORMANCE\n" + "="*60)
mi = {'model1_base': {'name': 'Bayesian Poisson GAM (Base)', 'short': 'Base GAM'},
      'model2_rf': {'name': 'GAM + Random Forest', 'short': 'GAM+RF'},
      'model3_gb': {'name': 'GAM + Gradient Boosting', 'short': 'GAM+GB'},
      'model4_lstm': {'name': 'GAM + LSTM', 'short': 'GAM+LSTM'},
      'model5_mlp': {'name': 'GAM + MLP Neural Network', 'short': 'GAM+MLP'}}
mout = {}

for key, info in mi.items():
    dr = pd.DataFrame(res[key]).dropna()
    if len(dr) == 0: continue
    yt, yp = dr['actual_cases'].values, dr['predicted_cases'].values
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    r2 = float(bounded_r2(yt, yp))

    hm = {}
    for w in sorted(dr['weeks_ahead'].unique()):
        sub = dr[dr['weeks_ahead'] == w]
        hm[int(w)] = {'rmse': float(np.sqrt(mean_squared_error(sub['actual_cases'], sub['predicted_cases']))),
            'mae': float(mean_absolute_error(sub['actual_cases'], sub['predicted_cases'])),
            'r2': float(bounded_r2(sub['actual_cases'], sub['predicted_cases']))}

    dfc = defaultdict(list)
    for _, r in dr.iterrows():
        dfc[r['district']].append({'week': int(r['forecast_week']), 'actual': float(r['actual_cases']),
            'predicted': float(r['predicted_cases']), 'weeks_ahead': int(r['weeks_ahead'])})

    mout[key] = {'name': info['name'], 'short_name': info['short'],
        'overall': {'rmse': rmse, 'mae': mae, 'r2': r2},
        'by_horizon': hm, 'forecasts': dict(dfc), 'raw': dr.to_dict('records')}
    print(f"{info['name']}: RMSE={rmse:.2f} MAE={mae:.2f} R2={r2:.3f}")
    for w, m in sorted(hm.items()): print(f"  W+{w}: RMSE={m['rmse']:.2f} R2={m['r2']:.3f}")

# ============== 8) SAVE ==============
print("\nSaving output...")
hist = defaultdict(list)
for _, r in panel.iterrows():
    hist[r['district']].append({'year': int(r['week_index'] // 52 + 2006),
        'week': int(r['week_index'] % 52 + 1), 'cases': float(r['cases'])})

output = {
    'meta': {'generated_at': datetime.now().isoformat(), 'models': list(mi.keys()),
        'model_names': {k: v['name'] for k, v in mi.items()},
        'validation_start': '2025-01-01', 'n_districts': n, 'districts': district_order},
    'historical': dict(hist),
    'model_outputs': mout,
    'districts': {d: {'coordinates': {'lat': 7.8731, 'lng': 80.7718},
        'historical': hist[d], 'province': d2p.get(d, 'Unknown')} for d in district_order}
}

with open(OUTPUT_JSON, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"Saved: {OUTPUT_JSON} ({os.path.getsize(OUTPUT_JSON) / 1024 / 1024:.1f} MB)")