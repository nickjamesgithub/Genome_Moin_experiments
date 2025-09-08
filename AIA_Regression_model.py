# --- Imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# --- Load data ---
stacked = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\Clean\Clean_insurance_data_genome_country.csv")

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
UNIVERSE_MODE = "developed"   # options: all | asia | non_asia | developed | emerging | asia_developed | asia_emerging | non_asia_developed | non_asia_emerging
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

# === Apply filter ===
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


#
# # --- Relabel: Company_name -> Country (preferred labels; else true domicile) ---
# country_map = {
#     'AIA Group': 'Hong_Kong',
#     'China Life Insurance Co.': 'China',
#     'Ping An Insurance (Group)': 'China',
#     'China Pacific Insurance (Group)': 'China',
#     'New China Life Insurance': 'China',
#     'China Taiping Insurance (Life)': 'China',
#     'Prudential Financial, Inc.': 'USA',
#     'Manulife Financial': 'Canada',
#     'Sun Life Financial': 'Canada',
#     'Great Eastern Holdings': 'Singapore',
#     'Dai-ichi Life Holdings': 'Japan',
#     'T&D Holdings': 'Japan',
#     'Japan Post Insurance': 'Japan',
#     'HDFC Life Insurance': 'India',
#     'ICICI Prudential Life': 'India',
#     'SBI Life Insurance': 'India',
#     'Samsung Life Insurance': 'South_Korea',
#     'Hanwha Life Insurance': 'South_Korea',
#     'Thai Life Insurance': 'Thailand',
#     'Bangkok Life Assurance': 'Thailand',
#     'Ageas': 'Belgium',
#     'Phoenix Group Holdings': 'United_Kingdom',
#     'Legal & General Group': 'United_Kingdom',
#     'Aflac Incorporated': 'USA',
#     'PICC Property & Casualty': 'China',
#     'ZhongAn Online P&C Insurance': 'China',
#     'AXA SA': 'France',
#     'Zurich Insurance Group': 'Switzerland',
#     'Chubb Limited': 'Switzerland',
#     'QBE Insurance Group': 'Australia',
#     'Insurance Australia Group': 'Australia',
#     'Suncorp Group': 'Australia',
#     'DB Insurance': 'South_Korea',
#     'Hyundai Marine & Fire': 'South_Korea',
#     'Mitsui Sumitomo (MS&AD)': 'Japan',
#     'AIG': 'USA',
#     'Travelers': 'USA',
#     'CNA Financial': 'USA',
#     'Admiral Group': 'United_Kingdom',
#     'Direct Line Group': 'United_Kingdom',
#     'Hiscox Ltd': 'United_Kingdom',  # Bermuda-domiciled; using UK per preferred set
#     'Beazley plc': 'United_Kingdom',
#     'Talanx AG (HDI)': 'Germany',
#     'Helvetia Holding': 'Switzerland',
#     'Baloise Holding': 'Switzerland',
#     'Allianz': 'Germany',
#     'Generali Group': 'Italy',
#     'Sompo Holdings': 'Japan',
#     'Tokio Marine Holdings': 'Japan',
#     'MetLife, Inc.': 'USA',
#     'Aegon N.V.': 'Netherlands',
#     'MAPFRE S.A.': 'Spain',
#     'Bao Viet Holdings': 'Vietnam',
#     'Bangkok Insurance PCL (composite)': 'Thailand',
#     'Dhipaya Group Holdings': 'Thailand',
#     'Syarikat Takaful Malaysia': 'Malaysia',
#     'Samsung Fire & Marine Insurance': 'South_Korea',
#     'Hanwha General Insurance': 'South_Korea',
#     'PT Asuransi Tugu Pratama Indonesia': 'Indonesia',
#     'Cincinnati Financial Corporation': 'USA',
#     'Lincoln National Corporation': 'USA',
#     'Principal Financial Group': 'USA',
#     'PZU SA': 'Poland',
#     'Vienna Insurance Group': 'Austria',
#     'Allstate Corporation': 'USA',
#     'Progressive Corporation': 'USA',
#     'Unum Group': 'USA',
#     'Hartford Financial Services': 'USA',
#     'LIC': 'India',
#     'Huaxia': 'China',
#     'Cathay Life': 'Taiwan',
#     'Fubon Financial': 'Taiwan',
#     'Medibank': 'Australia',
#     'Cigna': 'USA',
#     'United Health': 'USA',
#     'Humana': 'USA',
#     'CVS Health': 'USA',
# }
#
# mask_relabel = stacked['Country'].eq('insurance') & stacked['Company_name'].isin(country_map)
# stacked.loc[mask_relabel, 'Country'] = stacked.loc[mask_relabel, 'Company_name'].map(country_map)
#
# # === Optional: restrict to a country universe ===
# if UNIVERSE_COUNTRIES:
#     before_n = len(stacked)
#     stacked = stacked[stacked['Country'].isin(UNIVERSE_COUNTRIES)].copy()
#     after_n = len(stacked)
#     if after_n == 0:
#         raise ValueError(f"No rows left after filtering for {UNIVERSE_COUNTRIES}. "
#                          "Check spellings (e.g., 'United_Kingdom', 'Hong_Kong').")
#     print(f"Filtered to {UNIVERSE_COUNTRIES}: {after_n}/{before_n} rows.")

# --- Persist the relabeled data (optional) ---
# stacked.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\Clean\Clean_insurance_data_genome_country.csv", index=False)

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
