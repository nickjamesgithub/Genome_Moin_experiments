# --- Imports ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# use a non-interactive backend so saving works in any environment
import matplotlib
matplotlib.use("Agg")

# SHAP (install if needed: pip install shap)
try:
    import shap
except ImportError:
    raise SystemExit(
        "SHAP is not installed. Please run: pip install shap --upgrade"
    )

# --- Load data ---
df = pd.read_csv(
    r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\Clean\Clean_insurance_data_genome_gpt.csv"
)

# --- Focus only on AIA Group Limited ---
aia_df = df.loc[df["Company_name"] == "AIA Group Limited"].copy()

# --- Add Embedded Value and Price_to_Embedded_Value ---
embedded_value_by_year = {
    2010: 24748.0, 2011: 27239.0, 2012: 31408.0, 2013: 33818.0, 2014: 37153.0,
    2015: 38198.0, 2016: 42114.0, 2017: 50779.0, 2018: 54517.0, 2019: 61985.0,
    2020: 65247.0, 2021: 72987.0, 2022: 68865.0, 2023: 67447.0, 2024: 69035.0
}
aia_df["Embedded_Value"] = aia_df["Year"].map(embedded_value_by_year)
aia_df["Price_to_Embedded_Value"] = pd.to_numeric(
    aia_df["Market_Capitalisation"], errors="coerce"
) / aia_df["Embedded_Value"]

# --- Define response and features ---
RESPONSE = "TSR_CIQ_no_buybacks"
FEATURE_COLS = [
    # Core ratios & fundamentals
    "PE","PBV",
    "Gross_margin","Profit_margin","ROE","ROE_above_Cost_of_equity","ROA","BVE_per_share",
    "CAPEX/Revenue","CROTE_TE","EVA_Margin","EVA_momentum","EVA_shock","Economic_profit_1_f",
    "EP_growth_2_f","EP_growth_3_f","Revenue_growth_1_f","Revenue_growth_2_f","Revenue_growth_3_f",
    "NAV_1_f","NAV_growth_2_f","NAV_growth_3_f","profit_margin_1_f","profit_margin_growth_2_f",
    "profit_margin_growth_3_f","BVE_per_share_1_f","BVE_per_share_growth_2_f","BVE_per_share_growth_3_f",
    # New engineered features
    "Embedded_Value","Price_to_Embedded_Value"
]

# Keep only columns that exist
available_features = [c for c in FEATURE_COLS if c in aia_df.columns]

# --- Build X and y ---
y = pd.to_numeric(aia_df[RESPONSE], errors="coerce").replace([np.inf, -np.inf], np.nan)
X = aia_df[available_features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

# Drop missing values (rows with any NaNs in X or y)
mask = y.notna() & X.notna().all(axis=1)
X = X.loc[mask]
y = y.loc[mask]

print(f"Training data: {X.shape[0]} rows, {X.shape[1]} features")

if len(y) < 2:
    raise SystemExit("Not enough rows after cleaning to compute SHAP. Check missing values or filters.")

# --- Fit Random Forest ---
rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
rf.fit(X, y)

# --- SHAP values ---
# For tree models, TreeExplainer is fast and exact
explainer = shap.TreeExplainer(rf, feature_names=X.columns)
shap_values = explainer.shap_values(X)   # shape: (n_samples, n_features)
expected_value = explainer.expected_value

# --- SHAP summary (beeswarm) ---
plt.figure(figsize=(10, 7), dpi=150)
# max_display: how many top features to show (adjust if you want more/less)
shap.summary_plot(shap_values, X, plot_type="dot", max_display=20, show=False)
plt.title("AIA Group Limited — TSR Drivers (SHAP Beeswarm)")
plt.tight_layout()
plt.savefig("AIA_TSR_SHAP_Beeswarm.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Optional: SHAP bar of mean |SHAP| (global ranking) ---
plt.figure(figsize=(10, 7), dpi=150)
shap.summary_plot(shap_values, X, plot_type="bar", max_display=20, show=False)
plt.title("AIA Group Limited — Global Feature Importance (mean |SHAP|)")
plt.tight_layout()
plt.savefig("AIA_TSR_SHAP_Bar.png", dpi=150, bbox_inches="tight")
plt.show()

print("Saved plots:\n - AIA_TSR_SHAP_Beeswarm.png\n - AIA_TSR_SHAP_Bar.png")
