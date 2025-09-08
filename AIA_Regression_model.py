# --- Imports ---
import pandas as pd
import numpy as np

stacked = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\Clean\Clean_insurance_data_genome.csv")

# Map of Company_name -> Country (prefer your allowed set; otherwise true domicile)
country_map = {
    'AIA Group': 'Hong_Kong',
    'China Life Insurance Co.': 'China',
    'Ping An Insurance (Group)': 'China',
    'China Pacific Insurance (Group)': 'China',
    'New China Life Insurance': 'China',
    'China Taiping Insurance (Life)': 'China',
    'Prudential Financial, Inc.': 'USA',
    'Manulife Financial': 'Canada',
    'Sun Life Financial': 'Canada',
    'Great Eastern Holdings': 'Singapore',
    'Dai-ichi Life Holdings': 'Japan',
    'T&D Holdings': 'Japan',
    'Japan Post Insurance': 'Japan',
    'HDFC Life Insurance': 'India',
    'ICICI Prudential Life': 'India',
    'SBI Life Insurance': 'India',
    'Samsung Life Insurance': 'South_Korea',
    'Hanwha Life Insurance': 'South_Korea',
    'Thai Life Insurance': 'Thailand',
    'Bangkok Life Assurance': 'Thailand',
    'Ageas': 'Belgium',
    'Phoenix Group Holdings': 'United_Kingdom',
    'Legal & General Group': 'United_Kingdom',
    'Aflac Incorporated': 'USA',
    'PICC Property & Casualty': 'China',
    'ZhongAn Online P&C Insurance': 'China',
    'AXA SA': 'France',
    'Zurich Insurance Group': 'Switzerland',
    'Chubb Limited': 'Switzerland',
    'QBE Insurance Group': 'Australia',
    'Insurance Australia Group': 'Australia',
    'Suncorp Group': 'Australia',
    'DB Insurance': 'South_Korea',
    'Hyundai Marine & Fire': 'South_Korea',
    'Mitsui Sumitomo (MS&AD)': 'Japan',
    'AIG': 'USA',
    'Travelers': 'USA',
    'CNA Financial': 'USA',
    'Admiral Group': 'United_Kingdom',
    'Direct Line Group': 'United_Kingdom',
    'Hiscox Ltd': 'United_Kingdom',  # Bermuda-domiciled; using UK per your preferred set
    'Beazley plc': 'United_Kingdom',
    'Talanx AG (HDI)': 'Germany',
    'Helvetia Holding': 'Switzerland',
    'Baloise Holding': 'Switzerland',
    'Allianz': 'Germany',
    'Generali Group': 'Italy',
    'Sompo Holdings': 'Japan',
    'Tokio Marine Holdings': 'Japan',
    'MetLife, Inc.': 'USA',
    'Aegon N.V.': 'Netherlands',
    'MAPFRE S.A.': 'Spain',
    'Bao Viet Holdings': 'Vietnam',
    'Bangkok Insurance PCL (composite)': 'Thailand',
    'Dhipaya Group Holdings': 'Thailand',
    'Syarikat Takaful Malaysia': 'Malaysia',
    'Samsung Fire & Marine Insurance': 'South_Korea',
    'Hanwha General Insurance': 'South_Korea',
    'PT Asuransi Tugu Pratama Indonesia': 'Indonesia',
    'Cincinnati Financial Corporation': 'USA',
    'Lincoln National Corporation': 'USA',
    'Principal Financial Group': 'USA',
    'PZU SA': 'Poland',
    'Vienna Insurance Group': 'Austria',
    'Allstate Corporation': 'USA',
    'Progressive Corporation': 'USA',
    'Unum Group': 'USA',
    'Hartford Financial Services': 'USA',
    'LIC': 'India',
    'Huaxia': 'China',
    'Cathay Life': 'Taiwan',
    'Fubon Financial': 'Taiwan',
    'Medibank': 'Australia',
    'Cigna': 'USA',
    'United Health': 'USA',
    'Humana': 'USA',
    'CVS Health': 'USA',
}

# Overwrite only where current label is 'insurance' and company is in our map
mask = stacked['Country'].eq('insurance') & stacked['Company_name'].isin(country_map)
stacked.loc[mask, 'Country'] = stacked.loc[mask, 'Company_name'].map(country_map)

# (Optional) sanity check: which of the listed companies still show 'insurance' (e.g., naming mismatches)
still_unlabeled = sorted(set(country_map).intersection(stacked.loc[stacked['Country'].eq('insurance'), 'Company_name']))
print("Still labeled as 'insurance' (check name/spelling):", still_unlabeled)

# --- Feature set & response ---
RESPONSE = "TSR_CIQ_no_buybacks"
# NOTE: The original code listed 'profit_margin_growth_2_f' twice;
# this version assumes you intended to include the 3-year term as well.
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

# --- Human-readable labels for features (used for printing & plotting) ---
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

# --- Safety: ensure all columns exist ---
missing = [c for c in FEATURE_COLS + [RESPONSE] if c not in stacked.columns]
if missing:
    raise KeyError(f"Missing columns in stacked: {missing}")

# --- Build X, y with numeric coercion & finite filtering ---
y = pd.to_numeric(stacked[RESPONSE], errors="coerce").replace([np.inf, -np.inf], np.nan)
X = stacked[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

mask_finite = y.notna() & np.isfinite(y) & X.notna().all(axis=1)
X = X.loc[mask_finite]
y = y.loc[mask_finite]

print(f"\nAfter cleaning: rows={X.shape[0]}, features={X.shape[1]}")
print(f"y range: min={y.min():.6g}, max={y.max():.6g}")

# --- Fit Random Forest on all data ---
rf = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)
rf.fit(X, y)

# --- Feature importances (with readable labels) ---
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

# Map technical names -> human-readable labels for display
def pretty(name: str) -> str:
    return FEATURE_LABELS.get(name, name)

importances_pretty = importances.rename(index=pretty)

print("\nTop feature importances:")
print(importances_pretty.head(25))

# --- Plot top N with readable labels ---
top_n = 25
top_features = importances_pretty.head(top_n)

plt.figure(figsize=(10, 6), dpi=150)
top_features.plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("TSR Insurance Drivers")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("TSR_Insurance_Drivers")
plt.show()

