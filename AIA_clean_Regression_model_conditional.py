# =========================
# TSR MODEL — TIDY PIPELINE
# =========================
# - Exact label filtering for Country_label & insurance_type
# - Median imputation (no row nuking)
# - Robust winsorization (per-feature tails)
# - Safe final clamp to avoid float32 overflow in trees
# - Clear, minimal diagnostics
# - SHAP plots guarded for small-N

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

# ---------------------
# CONFIG (edit as needed)
# ---------------------
CSV_PATH = r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\Clean\Clean_insurance_data_250925.csv"

# Country universe:
#   "all" | "asia" | "non_asia" | "developed" | "emerging"
#   "asia_developed" | "asia_emerging" | "non_asia_developed" | "non_asia_emerging" | "custom"
UNIVERSE_MODE = "non_asia"
UNIVERSE_CUSTOM = []  # if UNIVERSE_MODE == "custom", e.g., ["USA","Australia","Japan"]

# Insurer type (EXACT to your CSV):
#   "all" | "pc" | "life" | "multiline" | "reinsurance" | "other" | "custom"
TYPE_MODE = "multiline"
TYPE_CUSTOM = ["P&C", "Life & Health"]  # if TYPE_MODE == "custom"

# Optional IFRS cutoff
LIMIT_TO_2023 = True

# Feature coverage handling
MIN_FEATURE_COVERAGE = 0.30  # keep columns with >=30% non-NaN before impute
MIN_FEATURES_FALLBACK = 10   # ensure at least this many best-covered features

# Target + features
RESPONSE = "TSR_CIQ_no_buybacks"
FEATURE_COLS = [
    "PBV","Profit_margin","ROE","ROE_above_Cost_of_equity","ROA",
    "BVE_per_share","CROTE_TE","EVA_Margin","EVA_momentum","EVA_shock",
    "EVA_Profitable_Growth","EVA_Productivity_Gains","Economic_profit_1_f",
    "EP_growth_2_f","EP_growth_3_f","Revenue_growth_1_f","Revenue_growth_2_f",
    "Revenue_growth_3_f","NAV_1_f","NAV_growth_2_f","NAV_growth_3_f",
    "profit_margin_1_f","profit_margin_growth_2_f","profit_margin_growth_3_f",
    "BVE_per_share_1_f","Dividend_Yield","Buyback_Yield"
]
FEATURE_LABELS = {
    "PBV":"P/BV","Profit_margin":"Profit margin","ROE":"ROE","ROE_above_Cost_of_equity":"ROE - CoE","ROA":"ROA",
    "BVE_per_share":"BVE/share","CROTE_TE":"CROTE - CoE","EVA_Margin":"EVA Margin","EVA_momentum":"EVA momentum",
    "EVA_shock":"EVA shock","EVA_Profitable_Growth":"EVA Profitable Growth","EVA_Productivity_Gains":"EVA Productivity Gains",
    "Economic_profit_1_f":"EP (1y)","EP_growth_2_f":"EP growth (2y)","EP_growth_3_f":"EP growth (3y)",
    "Revenue_growth_1_f":"Revenue growth (1y)","Revenue_growth_2_f":"Revenue growth (2y)","Revenue_growth_3_f":"Revenue growth (3y)",
    "NAV_1_f":"NAV (1y)","NAV_growth_2_f":"NAV growth (2y)","NAV_growth_3_f":"NAV growth (3y)",
    "profit_margin_1_f":"Profit margin (1y)","profit_margin_growth_2_f":"Profit margin growth (2y)",
    "profit_margin_growth_3_f":"Profit margin growth (3y)","BVE_per_share_1_f":"BVE/share (1y)",
    "Dividend_Yield":"Dividend Yield","Buyback_Yield":"Buyback Yield"
}

# ---------------------
# GROUP SETS (exact spellings as in your CSV)
# ---------------------
ASIA = {"Malaysia","South_Korea","India","Saudi_Arabia","Hong_Kong","China","Thailand","Singapore","Japan"}
NON_ASIA = {"Australia","Italy","Denmark","Netherlands","Belgium","France","United_Kingdom","USA","Switzerland","Canada","Germany"}
DEVELOPED = NON_ASIA | {"South_Korea","Hong_Kong","Singapore","Japan"}
EMERGING = {"Malaysia","India","Saudi_Arabia","China","Thailand"}

UNIVERSE_MAP = {
    "asia": ASIA,
    "non_asia": NON_ASIA,
    "developed": DEVELOPED,
    "emerging": EMERGING,
    "asia_developed": ASIA & DEVELOPED,
    "asia_emerging": ASIA & EMERGING,
    "non_asia_developed": NON_ASIA & DEVELOPED,
    "non_asia_emerging": NON_ASIA & EMERGING,
}
TYPE_MAP = {
    "pc": "P&C",
    "life": "Life & Health",
    "multiline": "Multiline",
    "reinsurance": "Reinsurance",
    "other": "OTHER",
}

# ---------------------
# HELPERS
# ---------------------
def winsorize_df(df: pd.DataFrame, lower_q=0.001, upper_q=0.999) -> pd.DataFrame:
    """Clip each column to robust quantiles (keeps structure, kills absurd tails)."""
    out = df.copy()
    for c in out.columns:
        v = out[c].to_numpy()
        finite = np.isfinite(v)
        n = finite.sum()
        if n >= 10:
            lo = np.nanpercentile(v[finite], lower_q * 100.0)
            hi = np.nanpercentile(v[finite], upper_q * 100.0)
            if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                out[c] = np.clip(v, lo, hi)
    return out

# ---------------------
# LOAD
# ---------------------
df = pd.read_csv(CSV_PATH)
print(f"Loaded rows: {len(df):,}")
if LIMIT_TO_2023 and "Year" in df.columns:
    df = df[df["Year"] <= 2023].copy()
    print(f"Year<=2023 rows: {len(df):,}")

# ---------------------
# FILTER — COUNTRY
# ---------------------
if UNIVERSE_MODE == "all":
    stacked = df.copy()
elif UNIVERSE_MODE == "custom":
    keep_countries = set(UNIVERSE_CUSTOM)
    stacked = df[df["Country_label"].isin(keep_countries)].copy()
else:
    keep_countries = UNIVERSE_MAP.get(UNIVERSE_MODE)
    if keep_countries is None:
        raise ValueError(f"Unknown UNIVERSE_MODE: {UNIVERSE_MODE}")
    stacked = df[df["Country_label"].isin(keep_countries)].copy()

print(f"After country filter: {len(stacked):,}/{len(df):,} rows")
if stacked.empty:
    raise ValueError("No rows after country filtering.")

# ---------------------
# FILTER — TYPE
# ---------------------
if TYPE_MODE == "all":
    pass
elif TYPE_MODE == "custom":
    keep_types = set(TYPE_CUSTOM)  # exact match
    stacked = stacked[stacked["insurance_type"].isin(keep_types)].copy()
else:
    keep_type = TYPE_MAP.get(TYPE_MODE)
    if keep_type is None:
        raise ValueError(f"Unknown TYPE_MODE: {TYPE_MODE}")
    stacked = stacked[stacked["insurance_type"] == keep_type].copy()

print(f"After type filter: {len(stacked):,} rows")
if stacked.empty:
    raise ValueError("No rows after type filtering.")

# ---------------------
# FEATURES / TARGET
# ---------------------
if RESPONSE not in stacked.columns:
    raise KeyError(f"Missing target column '{RESPONSE}'.")

existing_feats = [c for c in FEATURE_COLS if c in stacked.columns]
if not existing_feats:
    raise KeyError("None of the requested feature columns are present.")
FEATURE_COLS = existing_feats

# Coerce numeric
X_raw = stacked[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").astype("float64")
y = pd.to_numeric(stacked[RESPONSE], errors="coerce").astype("float64")

# ---------------------
# CLEANING (safe & minimal)
# ---------------------
# Replace inf with NaN (impute later)
X_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
y.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows only where y is NaN
keep = y.notna()
X_raw, y = X_raw.loc[keep], y.loc[keep]
print(f"Rows after y-drop: {len(y):,}")

# Drop columns with zero finite values
finite_counts = np.isfinite(X_raw.to_numpy()).sum(axis=0)
dead_cols = list(X_raw.columns[(finite_counts == 0)])
if dead_cols:
    X_raw.drop(columns=dead_cols, inplace=True)
    print("Dropped zero-finite columns:", dead_cols)

# Coverage-based feature keep (before impute)
coverage = X_raw.notna().mean().sort_values(ascending=False)
keep_cols = coverage[coverage >= MIN_FEATURE_COVERAGE].index.tolist()
if len(keep_cols) < MIN_FEATURES_FALLBACK:
    keep_cols = coverage.index[:MIN_FEATURES_FALLBACK].tolist()
dropped_lowcov = [c for c in X_raw.columns if c not in keep_cols]
if dropped_lowcov:
    print(f"Dropping low-coverage features (<{MIN_FEATURE_COVERAGE:.0%}): {dropped_lowcov}")
X_raw = X_raw[keep_cols]

# Winsorize tails per column (robust outlier control)
X_winz = winsorize_df(X_raw, lower_q=0.001, upper_q=0.999)

# Median impute remaining gaps (keep rows!)
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X_winz), columns=X_winz.columns, index=X_winz.index)

# Final conservative clamp (avoid float32 overflow deep in trees)
ABS_BOUND = 1e12
X = X.clip(lower=-ABS_BOUND, upper=ABS_BOUND)

# Final safety check
if not np.isfinite(X.to_numpy()).all():
    bad = X.columns[~np.isfinite(X).all()].tolist()
    raise ValueError(f"Non-finite values remain after cleaning: {bad}")

print(f"Modeling rows: {len(X):,}; features: {X.shape[1]}")

# ---------------------
# MODEL
# ---------------------
rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
rf.fit(X, y)

importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
pretty = importances.rename(index=lambda c: FEATURE_LABELS.get(c, c))

print("\nTop feature importances:")
print(pretty.head(min(25, len(pretty))).to_string())

# Importance plot
plt.figure(figsize=(10, 6), dpi=150)
pretty.head(min(25, len(pretty))).plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("TSR Insurance Drivers")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("AIA_Clean_TSR_Insurance_Drivers.png", dpi=150)
plt.show()

# ---------------------
# SHAP (guarded)
# ---------------------
try:
    import shap
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    if len(X) >= 2:
        X_pretty = X.rename(columns=lambda c: FEATURE_LABELS.get(c, c))
        plt.figure(figsize=(10, 7), dpi=150)
        shap.summary_plot(shap_values, X_pretty, plot_type="dot",
                          max_display=min(25, X.shape[1]), show=False)
        plt.title("TSR Drivers — SHAP Beeswarm")
        plt.tight_layout()
        plt.savefig("AIA_Insurance_all_TSR_SHAP_Beeswarm.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("Saved beeswarm: AIA_Insurance_all_TSR_SHAP_Beeswarm.png")
    else:
        # Single-row fallback: bar of |SHAP| values
        sv = np.array(shap_values).reshape(-1)
        abs_sv = pd.Series(np.abs(sv), index=X.columns).sort_values(ascending=True)
        plt.figure(figsize=(10, 7), dpi=150)
        abs_sv.plot(kind="barh")
        plt.title("Single Observation — |SHAP| values")
        plt.xlabel("|SHAP value|")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig("AIA_Insurance_single_obs_SHAP_bar.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("Saved single-obs SHAP bar: AIA_Insurance_single_obs_SHAP_bar.png")

except ImportError:
    print("SHAP not installed; skipping SHAP plots.")
