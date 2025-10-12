import pandas as pd
import numpy as np

# --- Config ---
CSV_PATH = r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\grocery_data.csv"
YEAR_START, YEAR_END = 2014, 2024
REQ_COLS = [
    "Company_name", "Year",
    "TSR_CIQ_no_buybacks", "NAV_1_f",
    "Revenue_growth_1_f", "EVA_ratio_bespoke"
]

# --- Load & validate ---
df = pd.read_csv(CSV_PATH)
missing = [c for c in REQ_COLS if c not in df.columns]
if missing:
    raise KeyError(f"Missing required columns in CSV: {missing}")

# --- Clean & types ---
df = df.replace([np.inf, -np.inf], np.nan)
df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
for c in ["NAV_1_f", "TSR_CIQ_no_buybacks", "Revenue_growth_1_f", "EVA_ratio_bespoke"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# --- Window & diffs ---
df = df[df["Year"].between(YEAR_START, YEAR_END)].copy()
df = df.sort_values(["Company_name", "Year"])
df["delta_NAV_1_f"] = df.groupby("Company_name")["NAV_1_f"].diff()

# --- Rank by largest ΔNAV and keep only requested columns ---
ranked_by_nav_increase = (
    df.dropna(subset=["delta_NAV_1_f"])
      .sort_values("delta_NAV_1_f", ascending=False)
      .loc[:, REQ_COLS]
      .reset_index(drop=True)
)

# --- Output ---
print(ranked_by_nav_increase.head(20))
ranked_by_nav_increase.to_csv("ranked_nav_increase_instances.csv", index=False)
x=1
y=2