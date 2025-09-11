# --- Imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# --- Load data ---
stacked = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\Clean\Clean_insurance_data_genome_gpt.csv")

# ====== Country groups & universe selector ======

asia = [
    "Hong_Kong", "Japan", "South_Korea", "Singapore", "Taiwan",
    "China", "India", "Thailand", "Vietnam", "Malaysia", "Indonesia",
    "Saudi_Arabia"
]
non_asia = [
    "Australia", "Belgium", "Denmark", "France", "Germany",
    "Italy", "Netherlands", "Switzerland", "United_Kingdom", "USA",
    "Canada", "Spain", "Austria", "Poland"
]
developed = [
    "Australia", "Belgium", "Denmark", "France", "Germany",
    "Italy", "Netherlands", "Switzerland", "United_Kingdom", "USA",
    "Canada", "Spain", "Austria",
    "Hong_Kong", "Japan", "South_Korea", "Singapore", "Taiwan"
]
emerging = [
    "China", "India", "Thailand", "Vietnam", "Malaysia", "Indonesia",
    "Saudi_Arabia", "Poland"
]

# Pick ONE mode or provide a custom list
UNIVERSE_MODE = "all"   # options: all | asia | non_asia | developed | emerging | asia_developed | asia_emerging | non_asia_developed | non_asia_emerging
UNIVERSE_CUSTOM = []     # e.g., ["USA", "Australia", "Japan"] — used only if UNIVERSE_MODE == "custom"

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

# === Apply region filter ===
_universe = resolve_universe(UNIVERSE_MODE)
if _universe is None:  # "all"
    print("Using ALL countries (no filtering).")
else:
    before_n = len(stacked)
    stacked = stacked[stacked["Country"].isin(_universe)].copy()
    after_n = len(stacked)
    if after_n == 0:
        raise ValueError(
            f"No rows left after filtering for mode='{UNIVERSE_MODE}' with countries={sorted(_universe)}.\n"
            "Check Country spellings (e.g., 'United_Kingdom', 'Hong_Kong')."
        )
    kept_counts = stacked["Country"].value_counts().to_dict()
    print(f"Filtered mode='{UNIVERSE_MODE}': {after_n}/{before_n} rows. Breakdown: {kept_counts}")

# ====== NEW: Insurer Type filter (works like Universe Mode) ======

# Options:
#   TYPE_MODE = "all"        -> no filter
#   TYPE_MODE = "pc"         -> keep only P&C
#   TYPE_MODE = "life"       -> keep only Life
#   TYPE_MODE = "multiline"  -> keep only Multiline
#   TYPE_MODE = "custom"     -> keep any types listed in TYPE_CUSTOM (e.g., ["P&C","Life"])
TYPE_MODE = "pc"
TYPE_CUSTOM = []   # used only if TYPE_MODE == "custom"

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

def resolve_types(mode: str):
    mode = (mode or "").lower()
    if "Type_of_Insurer" not in stacked.columns:
        raise KeyError("Type_of_Insurer column not found. Make sure you added it in the previous step.")
    # normalized view of what's present
    present = stacked["Type_of_Insurer"].dropna().map(_norm_type_label)
    valid = {"P&C", "Life", "Multiline"}
    if mode == "all":
        return None
    if mode == "pc":
        return {"P&C"}
    if mode == "life":
        return {"Life"}
    if mode == "multiline":
        return {"Multiline"}
    if mode == "custom":
        custom_norm = {_norm_type_label(t) for t in TYPE_CUSTOM}
        if not custom_norm.issubset(valid):
            raise ValueError(f"TYPE_CUSTOM must be subset of {sorted(valid)} (after normalization). "
                             f"Got: {sorted(custom_norm)}")
        return custom_norm
    raise ValueError(f"Unknown TYPE_MODE: {mode}")

_type_set = resolve_types(TYPE_MODE)
if _type_set is None:
    print("Using ALL insurer types (no type filtering).")
else:
    before_n = len(stacked)
    # normalize a temp column and filter against normalized choices
    tmp = stacked["Type_of_Insurer"].map(_norm_type_label)
    stacked = stacked.loc[tmp.isin(_type_set)].copy()
    after_n = len(stacked)
    if after_n == 0:
        raise ValueError(
            f"No rows left after insurer type filtering for TYPE_MODE='{TYPE_MODE}' with types={sorted(_type_set)}.\n"
            "Check that 'Type_of_Insurer' values are present."
        )
    kept_counts = tmp.loc[tmp.isin(_type_set)].value_counts().to_dict()
    print(f"Filtered insurer types {sorted(_type_set)}: {after_n}/{before_n} rows. Breakdown: {kept_counts}")

# --- Modeling setup ---
RESPONSE = "TSR_CIQ_no_buybacks"

FEATURE_COLS = [
    "PBV",
    "Gross_margin",
    "Profit_margin",
    "ROE",
    "ROE_above_Cost_of_equity",
    "ROA",
    "BVE_per_share",
    "CAPEX/Revenue",
    "CROTE_TE",
    "EVA_Margin",
    "EVA_momentum",
    "EVA_shock",
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
    "BVE_per_share_growth_2_f",
    "BVE_per_share_growth_3_f",
]

FEATURE_LABELS = {
    "PBV": "P:BV",
    "Gross_margin": "Gross margin",
    "Profit_margin": "Profit margin",
    "ROE": "ROE",
    "ROE_above_Cost_of_equity": "ROE - Cost of equity",
    "ROA": "ROA",
    "BVE_per_share": "BVE per share",
    "CAPEX/Revenue": "CAPEX/Revenue",
    "CROTE_TE": "CROTE - Cost of equity",
    "EVA_Margin": "EVA Margin",
    "EVA_momentum": "EVA momentum",
    "EVA_shock": "EVA shock",
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
    "BVE_per_share_1_f": "BVE per share (1-year)",
    "BVE_per_share_growth_2_f": "BVE per share growth (2-year)",
    "BVE_per_share_growth_3_f": "BVE per share growth (3-year)",
}

# --- Safety: ensure required columns exist ---
missing = [c for c in FEATURE_COLS + [RESPONSE] if c not in stacked.columns]
if missing:
    raise KeyError(f"Missing columns in stacked: {missing}")

# --- Build X, y with numeric coercion & finite filtering ---
y = pd.to_numeric(stacked[RESPONSE], errors="coerce").replace([np.inf, -np.inf], np.nan)
X = stacked[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

mask_finite = y.notna() & np.isfinite(y) & X.notna().all(axis=1)
X = X.loc[mask_finite]
y = y.loc[mask_finite]

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
plt.savefig("TSR_Insurance_Drivers.png", dpi=150)
plt.show()
