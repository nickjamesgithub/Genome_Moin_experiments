# --- Imports ---
import re
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# ====== CONFIG ======
RESPONSE = "TSR_CIQ_no_buybacks"

# Base list (we'll prune BVE/NAV and then add engineered features)
FEATURE_COLS = [
    "Profit_margin","ROE","ROA","EVA_Margin","EVA_momentum","EVA_shock","EVA_Profitable_Growth","EVA_Productivity_Gains","Gross_margin",
    "Economic_profit_1_f","EP_growth_2_f","EP_growth_3_f","Revenue_growth_1_f",
    "Revenue_growth_2_f","Revenue_growth_3_f",
    "profit_margin_1_f","profit_margin_growth_2_f","profit_margin_growth_3_f",
    "BVE_per_share_1_f","Dividend_Yield","Buyback_Yield","EBIT_margin"#,"Capex_to_Revenue"
]

FEATURE_LABELS = {
    "Profit_margin":"Profit margin","ROE":"ROE",
    "ROA":"ROA","EVA_Margin":"EVA Margin","EVA_momentum":"EVA momentum","EVA_shock":"EVA shock",
    "EVA_Profitable_Growth":"EVA Profitable Growth","EVA_Productivity_Gains":"EVA Productivity Gains",
    "Gross_margin":"Gross margin",
    "Economic_profit_1_f":"EP (1-year)","EP_growth_2_f":"EP growth (2-year)","EP_growth_3_f":"EP growth (3-year)",
    "Revenue_growth_1_f":"Revenue growth (1-year)","Revenue_growth_2_f":"Revenue growth (2-year)","Revenue_growth_3_f":"Revenue growth (3-year)",
    "profit_margin_1_f":"Profit margin (1-year)","profit_margin_growth_2_f":"Profit margin growth (2-year)","profit_margin_growth_3_f":"Profit margin growth (3-year)",
    "Dividend_Yield":"Dividend Yield","Buyback_Yield":"Buyback Yield",
    # Engineered labels:
    "ROFE":"ROFE",
    "ROFE - WACC":"ROFE - WACC",
    "ROFE_1_f":"ROFE (1-year)",
    "ROFE_growth_2_f":"ROFE (2-year)",
    "ROFE_growth_3_f":"ROFE (3-year)",
    "EBIT_margin":"EBIT margin",
    # "Capex_to_Revenue":"Capex / Revenue"
}

# --- Load data ---
df = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\grocery_data.csv")

# --- Engineer ROFE + ROFE - WACC ---
for col in ["EBIT","Funds_employed","WACC_Damodaran"]:
    if col not in df.columns:
        raise KeyError(f"Missing required column for engineering '{col}'")
ebit = pd.to_numeric(df["EBIT"], errors="coerce")
fe   = pd.to_numeric(df["Funds_employed"], errors="coerce").replace(0, np.nan)
wacc = pd.to_numeric(df["WACC_Damodaran"], errors="coerce")  # assumes same unit as ROFE
df["ROFE"] = ebit / fe
df["ROFE - WACC"] = df["ROFE"] - wacc

# --- Engineer EBIT_margin and Capex_to_Revenue ---
for col in ["Revenue","CAPEX"]:
    if col not in df.columns:
        raise KeyError(f"Missing required column for engineering '{col}'")
rev   = pd.to_numeric(df["Revenue"], errors="coerce").replace(0, np.nan)
capex = pd.to_numeric(df["CAPEX"], errors="coerce")

df["EBIT_margin"]      = ebit / rev
df["Capex_to_Revenue"] = capex / rev

# --- Detect entity/time for per-entity rolling (fallback to whole-frame order) ---
ENTITY_CANDIDATES = ["Ticker","Company","ISIN","GVKEY","permno","PERMNO","CUSIP","Entity","Firm","Sid","SID"]
TIME_CANDIDATES   = ["Year","year","fyear","FY","Period","period","Date","date"]

entity_col = next((c for c in ENTITY_CANDIDATES if c in df.columns), None)
time_col   = next((c for c in TIME_CANDIDATES if c in df.columns), None)

if time_col:
    df = df.sort_values([entity_col, time_col] if entity_col else [time_col])

# --- ROFE rolling means: 1y / 2y / 3y (per entity when available) ---
def _rolling_mean(s: pd.Series, window: int) -> pd.Series:
    if entity_col:
        return s.groupby(df[entity_col], dropna=False).transform(lambda x: x.rolling(window=window, min_periods=1).mean())
    else:
        return s.rolling(window=window, min_periods=1).mean()

df["ROFE_1_f"]        = _rolling_mean(df["ROFE"], 1)  # equals ROFE but kept for naming symmetry
df["ROFE_growth_2_f"] = _rolling_mean(df["ROFE"], 2)
df["ROFE_growth_3_f"] = _rolling_mean(df["ROFE"], 3)

# --- Remove ALL BVE* and NAV* features (case-insensitive) ---
drop_pattern = re.compile(r"(BVE|NAV)", flags=re.IGNORECASE)
FEATURE_COLS = [c for c in FEATURE_COLS if not drop_pattern.search(c)]
to_drop_cols = [c for c in df.columns if drop_pattern.search(c)]
df = df.drop(columns=to_drop_cols, errors="ignore")

# Add engineered features to model inputs (ensure uniqueness)
FEATURE_COLS += ["ROFE","ROFE - WACC","ROFE_1_f","ROFE_growth_2_f","ROFE_growth_3_f","EBIT_margin","Capex_to_Revenue"]
FEATURE_COLS = list(dict.fromkeys(FEATURE_COLS))  # de-dupe while preserving order

# --- Safety: ensure required columns exist ---
need = FEATURE_COLS + [RESPONSE]
missing = [c for c in need if c not in df.columns]
if missing:
    raise KeyError(f"Missing columns in dataframe: {missing}")

# --- Build X, y with numeric coercion & strict finite filtering ---
y = pd.to_numeric(df[RESPONSE], errors="coerce").replace([np.inf, -np.inf], np.nan)
X = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
X = X.where(X.abs() < 1e30)
mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
X, y = X.loc[mask].astype(np.float64), y.loc[mask].astype(np.float64)

print(f"Rows (clean): {len(X):,} | Features: {X.shape[1]}")
print(f"{RESPONSE} range: [{y.min():.4g}, {y.max():.4g}]")

# --- Fit Random Forest ---
rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
rf.fit(X, y)

# --- Feature importances (pretty labels) ---
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
importances_pretty = importances.rename(index=lambda n: FEATURE_LABELS.get(n, n))
print("\nTop feature importances:")
print(importances_pretty.head(25))

# --- Plot: Top-25 Importances (with labels) ---
top = importances_pretty.head(25)[::-1]
plt.figure(figsize=(10, 6), dpi=150)
ax = top.plot(kind="barh")
for bar, val in zip(ax.patches, top.values):
    ax.text(bar.get_width()*1.01, bar.get_y()+bar.get_height()/2, f"{val:.3f}", va="center")
plt.title("TSR Drivers — Retail grocery (ML model)")
plt.xlabel("Importance"); plt.ylabel("Feature"); plt.tight_layout()
plt.savefig("Grocery_TSR_Drivers_RF.png", dpi=150)
plt.show()

# ================== SHAP BEESWARM ==================
try:
    import shap
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)
    X_pretty = X.rename(columns=lambda c: FEATURE_LABELS.get(c, c))

    plt.figure(figsize=(10, 7), dpi=150)
    shap.summary_plot(shap_values, X_pretty, plot_type="dot", max_display=25, show=False)
    plt.title("TSR Drivers — Grocery (SHAP Beeswarm)")
    plt.tight_layout()
    plt.savefig("Grocery_TSR_SHAP_Beeswarm.png", dpi=150, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(10, 7), dpi=150)
    shap.summary_plot(shap_values, X_pretty, plot_type="violin", max_display=25, show=False)
    plt.title("TSR Drivers — Grocery (SHAP Violin)")
    plt.tight_layout()
    plt.savefig("Grocery_TSR_SHAP_Violin.png", dpi=150, bbox_inches="tight")
    plt.show()
except ImportError:
    raise SystemExit("SHAP is not installed. Please run: pip install shap --upgrade")
