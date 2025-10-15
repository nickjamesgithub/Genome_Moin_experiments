import pandas as pd, numpy as np
from pathlib import Path

OUTDIR = Path(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\Woolworths\SVC_data")
GLOBAL = Path(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\grocery_data.csv")

# --- load ---
tsr   = pd.read_csv(OUTDIR / "grocery_company_tsr_1_3_5_10_years_with_counts.csv")
panel = pd.read_csv(GLOBAL)

# --- keys ---
bs = lambda t: str(t).upper().split('.')[0].split('-')[0].split('/')[0].split(' ')[0]
tsr["Ticker_base"] = tsr["Ticker"].map(bs)
panel["Ticker_base"] = panel["Ticker"].map(bs)

# --- window & fields ---
panel["Year"] = pd.to_numeric(panel["Year"], errors="coerce")
panel = panel[panel["Year"].between(2015, 2024, inclusive="both")].copy()
for c in ["Revenue_growth_3_f","EVA_ratio_bespoke","EBIT","Revenue","Funds_employed","NAV_1_f"]:
    if c in panel.columns:
        panel[c] = pd.to_numeric(panel[c], errors="coerce")

# --- derived metrics ---
panel["EBIT_margin"] = panel["EBIT"] / panel["Revenue"].replace(0, np.nan)
panel["Capital_efficiency"] = panel["EBIT"] / panel["Funds_employed"].replace(0, np.nan)

# NEW: ROFE = EBIT / Funds_employed (explicit field)
panel["ROFE"] = panel["EBIT"] / panel["Funds_employed"].replace(0, np.nan)

panel["EVA_pos"] = panel["EVA_ratio_bespoke"] > 0
panel["RG3_gt_2p7"] = panel["Revenue_growth_3_f"] > 0.027

# --- (1) per-company medians over time (+ counts) ---
per_co = (
    panel.groupby(["Ticker_base","Company_name"])
         .agg(
             Revenue_growth_3_f_med=("Revenue_growth_3_f","median"),
             EBIT_margin_med=("EBIT_margin","median"),
             EVA_ratio_bespoke_med=("EVA_ratio_bespoke","median"),
             Capital_efficiency_med=("Capital_efficiency","median"),
             NAV_1_f_med=("NAV_1_f","median"),
             # NEW: median ROFE
             ROFE_med=("ROFE","median"),
             Years_EVA_pos=("EVA_pos","sum"),
             Years_RG3_gt_2p7=("RG3_gt_2p7","sum"),
         )
         .reset_index()
)

# --- quartiles on TSR_10Y ---
tsr = tsr.dropna(subset=["TSR_10Y"]).copy()
tsr["TSR_10Y_qtile"] = pd.qcut(tsr["TSR_10Y"], 4, labels=["Q1_Lowest","Q2","Q3","Q4_Highest"])
per_co = per_co.merge(tsr[["Ticker_base","TSR_10Y","TSR_10Y_qtile"]], on="Ticker_base", how="inner")

# --- (2) median across companies within each quartile ---
features = [
    "Revenue_growth_3_f_med","EBIT_margin_med","EVA_ratio_bespoke_med",
    "Capital_efficiency_med","NAV_1_f_med","ROFE_med",
    "Years_EVA_pos","Years_RG3_gt_2p7","TSR_10Y"
]
quartile_medians = (
    per_co.groupby("TSR_10Y_qtile")[features]
          .median()
          .reset_index()
)
quartile_medians["n_companies"] = per_co.groupby("TSR_10Y_qtile").size().values

# --- save (optional) ---
per_co.to_csv(OUTDIR / "company_medians_2015_2024.csv", index=False)
quartile_medians.to_csv(OUTDIR / "quartile_medians_from_company_medians_2015_2024.csv", index=False)
print("Done.")

x=1
y=2
