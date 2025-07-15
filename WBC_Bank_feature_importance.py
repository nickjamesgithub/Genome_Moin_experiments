# Random Forest SHAP values — Banking TSR (annual rows)
# ----------------------------------------------------
# Builds company‑year predictors for 2014‑2024, fits a RandomForestRegressor on all
# available rows, and visualises mean absolute SHAP values by pillar.
# Figure width widened so labels and x‑axis text are fully visible.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

# ------------------ Parameters ------------------ #
START_YEAR, END_YEAR = 2014, 2024

# "Australia", "Denmark", "Hong_Kong", "India", "Malaysia", "Netherlands",
# "Singapore", "Sweden", "Switzerland", "Thailand", "USA", "United_Kingdom",
# "Bespoke",

COUNTRIES = [
    "Australia", "Denmark", "Hong_Kong", "India", "Malaysia", "Netherlands",
    "Singapore", "Sweden", "Switzerland", "Thailand", "USA", "United_Kingdom",
    "Bespoke",
]
GLOBAL_PATH  = r"C:/Users/60848/OneDrive - Bain/Desktop/Genome_code_250605/Genome-pipeline-code/Genome-pipeline-code/global_platform_data/Global_data.csv"
BESPOKE_PATH = r"C:/Users/60848/OneDrive - Bain/Desktop/Project_Genome/global_platform_data/bespoke_data.csv"

# ------------------ Load & Filter ------------------ #

df = (
    pd.concat([pd.read_csv(GLOBAL_PATH), pd.read_csv(BESPOKE_PATH)], ignore_index=True)
      .rename(columns=str.strip)
      .query("Sector == 'Banking' and @START_YEAR <= Year <= @END_YEAR and Country in @COUNTRIES")
)

# ------------------ Numeric coercion ------------------ #
NUM_COLS = [
    "NPAT", "Revenue", "Debt_to_equity", "Revenue_growth_3_f", "BVE_per_share",
    "Dividend_Buyback_Yield", "TSR_CIQ_no_buybacks",
]
for col in NUM_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)

# Company identifier -----------------------------------------------------------
COMPANY = "Ticker" if "Ticker" in df.columns else "Company"

# ------------------ Feature engineering ---------------------------------------
# NPAT margin
df["NPAT_margin"] = df["NPAT"] / df["Revenue"]

# YoY BVE growth
df = df.sort_values([COMPANY, "Year"])
df["BVE_growth_1_year"] = (
    df.groupby(COMPANY)["BVE_per_share"].pct_change().replace([np.inf, -np.inf], np.nan)
)

# Country‑year aggregates
country_year_growth = (
    df.groupby(["Country", "Year"])["Revenue_growth_3_f"].mean().rename("Country_growth")
)
df = df.merge(country_year_growth, on=["Country", "Year"], how="left")

peer_bve_median = (
    df.groupby(["Country", "Year"])["BVE_growth_1_year"].median().rename("Peer_BVE_median")
)
df = df.merge(peer_bve_median, on=["Country", "Year"], how="left")

df["BVE_VS_MARKET"] = df["BVE_growth_1_year"] - df["Peer_BVE_median"]

# ------------------ Predictor renaming ------------------ #
RENAME = {
    "NPAT_margin": "Operating productivity",
    "Debt_to_equity": "Balance sheet productivity",
    "Revenue_growth_3_f": "Growth story",
    "Country_growth": "Market characteristics",
    "BVE_growth_1_year": "Capital management",
    "BVE_VS_MARKET": "Growth vs market",
    "Dividend_Buyback_Yield": "Capital return",
}
PILLARS = list(RENAME.values())

df = df.rename(columns=RENAME)

# ------------------ Modelling matrix ------------------ #
mod_df = (
    df.replace([np.inf, -np.inf], np.nan)
      .dropna(subset=["TSR_CIQ_no_buybacks"])
      .copy()
)

imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(mod_df[PILLARS]), columns=PILLARS)
y = mod_df["TSR_CIQ_no_buybacks"].values

if len(X) < 10:
    raise ValueError("Too few observations after cleaning. Check data availability.")

# ------------------ Train model ------------------ #
rf = RandomForestRegressor(n_estimators=1000, random_state=42, n_jobs=-1, oob_score=True)
rf.fit(X, y)
print(f"R² on training data: {r2_score(y, rf.predict(X)):.3f}")
print(f"Out‑of‑bag R² estimate: {rf.oob_score_:.3f}\n")

# ------------------ SHAP values ------------------ #
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X)

# ----- Plot mean(|SHAP|) bar chart (wider) -----
shap.summary_plot(
    shap_values,
    X,
    plot_type="bar",
    feature_names=PILLARS,
    show=False,
    plot_size=(12, 6)  # width, height in inches
)
plt.gcf().set_size_inches(12, 6)  # ensure figure is wide
plt.title("Global banking system drivers")
plt.tight_layout()
plt.savefig("WBC_Global_banking_system_drivers")
plt.show()
