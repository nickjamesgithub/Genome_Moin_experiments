# --- Imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# ====== FILTER CONFIG AT THE VERY TOP ======
# Pick ONE mode or provide a custom list
UNIVERSE_MODE = "all"   # options: all | asia | non_asia | developed | emerging | asia_developed | asia_emerging | non_asia_developed | non_asia_emerging
UNIVERSE_CUSTOM = []          # e.g., ["USA", "Australia", "Japan"] — used only if UNIVERSE_MODE == "custom"

# Insurer Type filter
#   TYPE_MODE = "all"        -> no filter
#   TYPE_MODE = "pc"         -> keep only P&C
#   TYPE_MODE = "life"       -> keep only Life
#   TYPE_MODE = "multiline"  -> keep only Multiline
#   TYPE_MODE = "custom"     -> keep any types listed in TYPE_CUSTOM (e.g., ["P&C","Life"])
TYPE_MODE = "all"
TYPE_CUSTOM = []   # used only if TYPE_MODE == "custom"

# ====== Country groups ======
asia = [
    "Malaysia", "South_Korea", "India", "Saudi_Arabia",
    "Hong_Kong", "China", "Thailand", "Singapore", "Japan"
]

non_asia = [
    "Australia", "Italy", "Denmark", "Netherlands", "Belgium",
    "France", "United_Kingdom", "USA", "Switzerland", "Canada", "Germany"
]

developed = [
    "Australia", "Italy", "Denmark", "Netherlands", "Belgium",
    "France", "United_Kingdom", "USA", "Switzerland", "Canada", "Germany",
    "South_Korea", "Hong_Kong", "Singapore", "Japan"
]

emerging = [
    "Malaysia", "India", "Saudi_Arabia", "China", "Thailand"
]

def resolve_universe(mode: str):
    mode = (mode or "").lower()
    sets = {
        "asia": set(asia),
        "non_asia": set(non_asia),
        "developed": set(developed),
        "emerging": set(emerging),
    }
    if mode == "all":
        return None  # no filtering
    if mode == "custom":
        return set(UNIVERSE_CUSTOM)
    if mode in sets:
        return sets[mode]
    if mode == "asia_developed":
        return sets["asia"] & sets["developed"]
    if mode == "asia_emerging":
        return sets["asia"] & sets["emerging"]
    if mode == "non_asia_developed":
        return sets["non_asia"] & sets["developed"]
    if mode == "non_asia_emerging":
        return sets["non_asia"] & sets["emerging"]
    raise ValueError(f"Unknown UNIVERSE_MODE: {mode}")

def _norm_type_label(x: str) -> str:
    if not isinstance(x, str):
        return x
    s = x.strip().lower().replace("and", "&").replace(" ", "")
    # normalize variants to "P&C", "Life", "Multiline"
    if s in {"pc", "p&c", "pnc", "p+c"}:
        return "P&C"
    if s in {"life"}:
        return "Life"
    if s in {"multi", "multiline", "composite"}:
        return "Multiline"
    return x  # leave unchanged if not matched

def resolve_types(mode: str, df_columns=None):
    mode = (mode or "").lower()
    if df_columns is not None and "insurance_type" not in df_columns:
        raise KeyError("insurance_type column not found. Make sure you added it in the previous step.")
    valid = {"P&C", "Life", "Multiline"}
    if mode == "all":
        return None
    if mode == "pc":
        return {"P&C"}
    if mode == "life":
        return {"Life & Health"}
    if mode == "multiline":
        return {"Multiline"}
    if mode == "custom":
        custom_norm = {_norm_type_label(t) for t in TYPE_CUSTOM}
        if not custom_norm.issubset(valid):
            raise ValueError(f"TYPE_CUSTOM must be subset of {sorted(valid)} (after normalization). "
                             f"Got: {sorted(custom_norm)}")
        return custom_norm
    raise ValueError(f"Unknown TYPE_MODE: {mode}")

# --- Load data ---
df = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\Clean\Clean_insurance_data_250925.csv")
# To fix any IFRS issues
stacked = df.loc[df["Year"]<=2023]

# === Apply region filter ===
_universe = resolve_universe(UNIVERSE_MODE)
if _universe is None:  # "all"
    print("Using ALL countries (no filtering).")
else:
    before_n = len(stacked)
    stacked = stacked[stacked["Country_label"].isin(_universe)].copy()
    after_n = len(stacked)
    if after_n == 0:
        raise ValueError(
            f"No rows left after filtering for mode='{UNIVERSE_MODE}' with countries={sorted(_universe)}.\n"
            "Check Country spellings (e.g., 'United_Kingdom', 'Hong_Kong')."
        )
    kept_counts = stacked["Country_label"].value_counts().to_dict()
    print(f"Filtered mode='{UNIVERSE_MODE}': {after_n}/{before_n} rows. Breakdown: {kept_counts}")

# ====== Insurer Type filter ======
_type_set = resolve_types(TYPE_MODE, df_columns=stacked.columns)
if _type_set is None:
    print("Using ALL insurer types (no type filtering).")
else:
    before_n = len(stacked)
    # normalize a temp column and filter against normalized choices
    tmp = stacked["insurance_type"].map(_norm_type_label)
    stacked = stacked.loc[tmp.isin(_type_set)].copy()
    after_n = len(stacked)
    if after_n == 0:
        raise ValueError(
            f"No rows left after insurer type filtering for TYPE_MODE='{TYPE_MODE}' with types={sorted(_type_set)}.\n"
            "Check that 'insurance_type' values are present."
        )
    kept_counts = tmp.loc[tmp.isin(_type_set)].value_counts().to_dict()
    print(f"Filtered insurer types {sorted(_type_set)}: {after_n}/{before_n} rows. Breakdown: {kept_counts}")

# --- Modeling setup ---
RESPONSE = "TSR_CIQ_no_buybacks"

FEATURE_COLS = [
    "PBV",
    "Profit_margin",
    "ROE",
    "ROE_above_Cost_of_equity",
    "ROA",
    "BVE_per_share",
    "CROTE_TE",
    "EVA_Margin",
    "EVA_momentum",
    "EVA_shock",
    "EVA_Profitable_Growth",
    "EVA_Productivity_Gains",
    "Economic_profit_1_f",
    "EP_growth_2_f",
    "EP_growth_3_f",
    "Revenue_growth_1_f",
    "Revenue_growth_2_f",
    "Revenue_growth_3_f",
    "NAV_1_f",
    "NAV_growth_2_f",
    "NAV_growth_3_f",
    "profit_margin_1_f",
    "profit_margin_growth_2_f",
    "profit_margin_growth_3_f",
    "BVE_per_share_1_f",
    "Dividend_Yield",
    "Buyback_Yield"
]

FEATURE_LABELS = {
    "PBV": "P:BV",
    "Profit_margin": "Profit margin",
    "ROE": "ROE",
    "ROE_above_Cost_of_equity": "ROE - Cost of equity",
    "ROA": "ROA",
    "BVE_per_share": "BVE per share",
    "CROTE_TE": "CROTE - Cost of equity",
    "EVA_Margin": "EVA Margin",
    "EVA_momentum": "EVA momentum",
    "EVA_shock": "EVA shock",
    "EVA_Profitable_Growth": "EVA Profitable Growth",
    "EVA_Productivity_Gains": "EVA Productivity Gains",
    "Economic_profit_1_f": "EP (1-year)",
    "EP_growth_2_f": "EP growth (2-year)",
    "EP_growth_3_f": "EP growth (3-year)",
    "Revenue_growth_1_f": "Revenue growth (1-year)",
    "Revenue_growth_2_f": "Revenue growth (2-year)",
    "Revenue_growth_3_f": "Revenue growth (3-year)",
    "NAV_1_f": "NAV growth (1-year)",
    "NAV_growth_2_f": "NAV growth (2-year)",
    "NAV_growth_3_f": "NAV growth (3-year)",
    "profit_margin_1_f": "Profit margin (1-year)",
    "profit_margin_growth_2_f": "Profit margin growth (2-year)",
    "profit_margin_growth_3_f": "Profit margin growth (3-year)",
    "BVE_per_share_1_f": "BVE per share growth (1-year)",
    "Dividend_Yield": "Dividend Yield",
    "Buyback_Yield": "Buyback Yield"
}

# --- Safety: ensure required columns exist ---
missing = [c for c in FEATURE_COLS + [RESPONSE] if c not in stacked.columns]
if missing:
    raise KeyError(f"Missing columns in stacked: {missing}")

# --- Build X, y with numeric coercion & strict finite filtering ---
y = pd.to_numeric(stacked[RESPONSE], errors="coerce")
X = stacked[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")

# 1) Turn any +/-inf into NaN (common in Dividend_Yield / Buyback_Yield)
X = X.replace([np.inf, -np.inf], np.nan)
y = y.replace([np.inf, -np.inf], np.nan)

# 2) Guard against absurd magnitudes that can overflow float32 inside sklearn
#    (anything this large is almost certainly a bad data artifact)
X = X.where(X.abs() < 1e30)

# 3) Keep only rows fully finite in BOTH X and y
mask_finite = np.isfinite(y) & np.isfinite(X).all(axis=1)
X = X.loc[mask_finite].astype(np.float64)
y = y.loc[mask_finite].astype(np.float64)

print(f"After cleaning: rows={X.shape[0]}, features={X.shape[1]}")
print(f"y range: min={y.min():.6g}, max={y.max():.6g}")

# --- Fit Random Forest ---
rf = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)
rf.fit(X, y)

# --- Feature importances (pretty labels) ---
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
importances_pretty = importances.rename(index=lambda n: FEATURE_LABELS.get(n, n))

print("\nTop feature importances:")
print(importances_pretty.head(25))

# --- Plot ---
top_n = 25
top_features = importances_pretty.head(top_n)

plt.figure(figsize=(10, 6), dpi=150)
top_features.plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("TSR Insurance Drivers")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("AIA_Clean_TSR_Insurance_Drivers.png", dpi=150)
plt.show()

# ================== SHAP BEESWARM (append-only) ==================
# Optional dependency: pip install shap --upgrade
try:
    import shap
except ImportError:
    raise SystemExit(
        "SHAP is not installed. Please run: pip install shap --upgrade"
    )

# For tree models, TreeExplainer is fast/accurate
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X)

# Use pretty labels if available (for nicer axis names)
X_pretty = X.rename(columns=lambda c: FEATURE_LABELS.get(c, c))

# Beeswarm summary (direction + magnitude for each feature)
plt.figure(figsize=(10, 7), dpi=150)
shap.summary_plot(shap_values, X_pretty, plot_type="dot", max_display=25, show=False)
plt.title("TSR Drivers — SHAP Beeswarm")
plt.tight_layout()
plt.savefig("AIA_Insurance_all_TSR_SHAP_Beeswarm.png", dpi=150, bbox_inches="tight")
plt.show()

print("Saved SHAP beeswarm: TSR_SHAP_Beeswarm.png")

plt.figure(figsize=(10, 7), dpi=150)
shap.summary_plot(
    shap_values, X_pretty,
    plot_type="violin",
    max_display=25,
    show=False
)
plt.title("TSR Drivers — SHAP Beeswarm (Violin)")
plt.tight_layout()
plt.savefig("AIA_Insurance_all_TSR_SHAP_Beeswarm_Violin.png", dpi=150, bbox_inches="tight")
plt.show()
# ================================================================

