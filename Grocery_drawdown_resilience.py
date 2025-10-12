import pandas as pd, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ================== PATHS ==================
BASE   = Path(r"C:\\Users\\60848\\OneDrive - Bain\\Desktop\\Project_Genome\\global_platform_data\\share_price\\Grocery")
GLOBAL = Path(r"C:\\Users\\60848\\OneDrive - Bain\\Desktop\\Project_Genome\\global_platform_data\\grocery_data.csv")
OUTDIR = Path(r"C:\\Users\\60848\\OneDrive - Bain\\Desktop\\Project_Genome\\casework\\Woolworths\\SVC_data")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ================== HELPERS ==================
def to_datetime_any(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    na = dt.isna()
    if na.any():
        nums = pd.to_numeric(s, errors="coerce")
        m = na & nums.notna()
        if m.any():
            dt.loc[m] = pd.to_datetime(nums[m], unit="D", origin="1899-12-30", errors="coerce")
    return dt

def base_symbol(t: str) -> str:
    t = str(t).upper().strip()
    for sep in [".", "-", "/", " "]:
        if sep in t:
            t = t.split(sep, 1)[0]
    return t

def extract_file_ticker(p: Path) -> str:
    name = p.name
    if name.startswith("_") and name.lower().endswith("_price.csv"):
        return name[1:-10].strip()
    return ""

# ================== BUILD PRICES ==================
price_files = list(BASE.glob("_*_price.csv"))
frames = []
for p in price_files:
    try:
        df = pd.read_csv(p)
        if {"Date","Price"}.issubset(df.columns):
            df["Date"]  = to_datetime_any(df["Date"])
            df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
            df = df.dropna(subset=["Date","Price"]).copy()
            raw  = extract_file_ticker(p)
            base = base_symbol(raw)
            df["Ticker_base"] = base
            frames.append(df[["Date","Price","Ticker_base"]])
    except Exception:
        continue

prices = pd.concat(frames, ignore_index=True)

# ================== PEAK-TO-TROUGH DRAWDOWN ==================
g = prices.dropna(subset=["Date","Price"]).sort_values(["Ticker_base","Date"]).copy()
g["idx100"] = g.groupby("Ticker_base")["Price"].transform(lambda s: s / s.iloc[0] * 100)

# log drawdown on idx100 (safe)
g["lnP"]    = np.log(g["idx100"].replace(0, np.nan))
g["lnPeak"] = g.groupby("Ticker_base")["lnP"].cummax()
g["dd_log"] = g["lnP"] - g["lnPeak"]  # <= 0 where valid

# worst drawdown per ticker as %
pt = (np.exp(g.groupby("Ticker_base")["dd_log"].min()) - 1) * 100
pt = pt.replace([np.inf, -np.inf], np.nan).dropna()
pt = pt.rename("PeakToTrough_pct").reset_index()
pt["PeakToTrough_pct"] = pt["PeakToTrough_pct"].round(2)
pt.to_csv(OUTDIR / "price_peak_to_trough_by_ticker.csv", index=False)

# ================== FUNDAMENTALS ==================
panel = pd.read_csv(GLOBAL)
panel["Ticker_base"] = panel["Ticker"].astype(str).map(base_symbol)
panel["Year"] = pd.to_numeric(panel["Year"], errors="coerce")
panel = panel[panel["Year"].between(2015, 2024, inclusive="both")].copy()

for c in ["Revenue_growth_3_f","EVA_ratio_bespoke","EBIT","NPAT","Revenue","Funds_employed","NAV_1_f"]:
    if c in panel.columns:
        panel[c] = pd.to_numeric(panel[c], errors="coerce")

# ===== SAFE RATIO ENGINEERING (prevents inf) =====
rev  = panel["Revenue"].astype(float)
fe   = panel["Funds_employed"].astype(float)
ebit = panel["EBIT"].astype(float)
npat = panel["NPAT"].astype(float)

panel["EBIT_margin"] = np.divide(ebit, rev, out=np.full_like(ebit, np.nan, dtype="float64"), where=(rev.to_numpy()!=0))
panel["NPAT_margin"] = np.divide(npat, rev, out=np.full_like(npat, np.nan, dtype="float64"), where=(rev.to_numpy()!=0))
panel["Cap_eff"]     = np.divide(ebit, fe,  out=np.full_like(ebit, np.nan, dtype="float64"), where=(fe.to_numpy()!=0))

panel["EVA_pos"] = panel["EVA_ratio_bespoke"] > 0
panel["RG3_pos"] = panel["Revenue_growth_3_f"] > 0

# per-company aggregation (means + counts)
feat = (panel.groupby("Ticker_base", as_index=False)
        .agg(
            Revenue_growth_3_f_avg=("Revenue_growth_3_f", "mean"),
            EBIT_margin_avg=("EBIT_margin", "mean"),
            NPAT_margin_avg=("NPAT_margin", "mean"),
            EVA_ratio_bespoke_avg=("EVA_ratio_bespoke", "mean"),
            Capital_efficiency_avg=("Cap_eff", "mean"),
            NAV_1_f_avg=("NAV_1_f", "mean"),
            Years_EVA_pos=("EVA_pos", "sum"),
            Years_RG3_pos=("RG3_pos", "sum"),
        ))

# ================== MERGE & CLEAN FEATURES ==================
df = pt.merge(feat, on="Ticker_base", how="inner").reset_index(drop=True)

# Business-friendly relabels
rename_map = {
    "Revenue_growth_3_f_avg": "Topline growth",
    "EBIT_margin_avg": "Business efficiency",
    "NPAT_margin_avg": "Profitability",
    "EVA_ratio_bespoke_avg": "Economic profits",
    "Capital_efficiency_avg": "Capital efficiency",
    "NAV_1_f_avg": "Productive growth in capital",
    "Years_EVA_pos": "EP consistency",
    "Years_RG3_pos": "Growth consistency",
}
df = df.rename(columns=rename_map)

# De-inf / winsorize features
features_clean = [
    "Topline growth",
    "Business efficiency",
    "Profitability",
    "Economic profits",
    "Capital efficiency",
    "Productive growth in capital",
    "EP consistency",
    "Growth consistency",
]
for c in features_clean:
    df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

q_lo = df[features_clean].quantile(0.01)
q_hi = df[features_clean].quantile(0.99)
df[features_clean] = df[features_clean].clip(lower=q_lo, upper=q_hi, axis=1)

# ensure finites for model
mask_finite = np.isfinite(df["PeakToTrough_pct"])
for c in features_clean:
    mask_finite &= np.isfinite(df[c])
df = df.loc[mask_finite].reset_index(drop=True)

# ================== MODEL ==================
X = df.drop(columns=["Ticker_base","PeakToTrough_pct"])
y = df["PeakToTrough_pct"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)

print(f"R²: {r2_score(y_test, pred):.3f} | MAE: {mean_absolute_error(y_test, pred):.2f}%")

# ================== FEATURE IMPORTANCE ==================
fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
fi.rename("importance").to_csv(OUTDIR / "drawdown_model_feature_importances.csv")

# --- Plot (feature importances) ---
plt.figure(figsize=(8,5))
plt.barh(fi.index, fi.values)
plt.title("Retail equity drawdown resilience")
plt.xlabel("Relative Importance")
plt.tight_layout()
plt.show()

# ================== SAVE PREDICTIONS ==================
df.assign(predicted_drawdown_pct=rf.predict(X)).to_csv(
    OUTDIR / "drawdown_model_dataset_with_preds.csv", index=False)

# ================== 12-MONTH WINDOW AROUND TROUGH (ALL EQUITIES) ==================
# Trough date per ticker (idx of min dd_log), drop NaN troughs
min_idx = g.groupby("Ticker_base")["dd_log"].idxmin()
min_idx = min_idx.dropna()
troughs = g.loc[min_idx, ["Ticker_base","Date","dd_log"]].dropna(subset=["Date"]).rename(columns={"Date":"TroughDate"})

win_list = []
for _, row in troughs.iterrows():
    tkr = row["Ticker_base"]
    td  = row["TroughDate"]
    start = td - pd.DateOffset(months=6)
    end   = td + pd.DateOffset(months=6)
    w = g[(g["Ticker_base"] == tkr) & (g["Date"].between(start, end, inclusive="both"))].copy()
    if w.empty:
        continue
    base = w.iloc[0]["idx100"]
    if not np.isfinite(base) or base == 0:
        continue
    w["rebased"] = w["idx100"] / base * 100.0
    w["rel_day"] = (w["Date"] - td).dt.days  # 0 at trough
    win_list.append(w[["Ticker_base","rel_day","rebased"]])

if win_list:
    around = pd.concat(win_list, ignore_index=True)
    plt.figure(figsize=(9,5))
    for tkr, sub in around.groupby("Ticker_base"):
        plt.plot(sub["rel_day"], sub["rebased"], linewidth=1)
    plt.axvline(0, linestyle="--")
    plt.title("Retail share price paths: 6 months before/after equity trough (rebased to 100)")
    plt.xlabel("12 months around equity trough (days)")
    plt.ylabel("Index (start of window = 100)")
    plt.tight_layout()
    plt.show()
else:
    print("No sufficient windowed data to plot around troughs.")

print("All outputs saved to:", OUTDIR)
