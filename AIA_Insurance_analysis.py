# --- Imports ---
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor  # (unused below but keeping as in your script)
import matplotlib.pyplot as plt

# --- Paths (edit these if needed) ---
GLOBAL_DATA_PATH     = r"C:\Users\60848\OneDrive - Bain\Desktop\Genome_code_250605\Genome-pipeline-code\Genome-pipeline-code\global_platform_data\Global_data.csv"
ASIAN_INSURANCE_PATH = r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\Insurance_data\merged_insurance_data_with_genome.csv"
OUT_MERGED_PATH      = r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\Insurance_data\merge_asian_global.csv"
OUT_PNG_PATH         = r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\Insurance_data\AIA_Insurance_Drivers.png"
OUT_TSR_PATH         = r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\Insurance_data\AIA_Insurance_TSR10.csv"

# --- Load ---
global_data     = pd.read_csv(GLOBAL_DATA_PATH)
asian_insurance = pd.read_csv(ASIAN_INSURANCE_PATH)

# --- Keep Insurance rows from global data ---
insurance_global = global_data.loc[global_data["Sector"] == "Insurance"].copy()

# --- Stack on overlapping columns (row-wise union) ---
common_cols = asian_insurance.columns.intersection(insurance_global.columns)
stacked = (
    pd.concat(
        [asian_insurance[common_cols], insurance_global[common_cols]],
        axis=0,
        ignore_index=True
    )
    .drop_duplicates(subset=list(common_cols), keep="first")
)

# Save merged
stacked.to_csv(OUT_MERGED_PATH, index=False)

# -------------------------------------------------------------------------
# 10-year annualised TSR (2014 -> 2024) by Company_name
# TSR_10yr = (Adjusted_Stock_Price_2024 / Adjusted_Stock_Price_2014) ** (1/10) - 1
# -------------------------------------------------------------------------

df = stacked.copy()

# --- Ensure required columns exist / create Year if needed ---
required_cols = {"Company_name", "Adjusted_Stock_Price"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in merged data: {missing}")

# Create/clean Year column
if "Year" in df.columns:
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
elif "Date" in df.columns or "date" in df.columns:
    date_col = "Date" if "Date" in df.columns else "date"
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    df["Year"] = df[date_col].dt.year.astype("Int64")
else:
    raise ValueError("No 'Year' or 'Date' column found to extract years from.")

# --- Keep only 2014 and 2024, coerce prices numeric ---
df["Adjusted_Stock_Price"] = pd.to_numeric(df["Adjusted_Stock_Price"], errors="coerce")
df_tsr = df[df["Year"].isin([2014, 2024])].copy()

# If there are multiple observations per company-year, pick the *last* by date/time if available,
# otherwise use the mean of available prices to be robust.
if "Date" in df_tsr.columns or "date" in df_tsr.columns:
    date_col = "Date" if "Date" in df_tsr.columns else "date"
    df_tsr[date_col] = pd.to_datetime(df_tsr[date_col], errors="coerce", utc=True)
    df_tsr = df_tsr.sort_values([ "Company_name", "Year", date_col ])
    # last observation in the year
    yearly_prices = (
        df_tsr.groupby(["Company_name", "Year"], as_index=False)["Adjusted_Stock_Price"]
        .last()
    )
else:
    # fallback: average if no timestamp available
    yearly_prices = (
        df_tsr.groupby(["Company_name", "Year"], as_index=False)["Adjusted_Stock_Price"]
        .mean()
    )

# --- Pivot to get 2014 and 2024 side-by-side ---
pivot = yearly_prices.pivot(index="Company_name", columns="Year", values="Adjusted_Stock_Price").rename_axis(None, axis=1)
# Ensure both columns exist even if some companies are missing one year
for yr in (2014, 2024):
    if yr not in pivot.columns:
        pivot[yr] = np.nan

# --- Compute TSR (CAGR) with validity checks ---
P2014 = pivot[2014]
P2024 = pivot[2024]

# Valid only when both are positive numbers
valid = (P2014 > 0) & (P2024 > 0)
tsr_10yr = pd.Series(np.where(valid, (P2024 / P2014) ** (1/10) - 1, np.nan), index=pivot.index)

# --- Build results table ---
result = (
    pd.DataFrame({
        "Company_name": pivot.index,
        "Adjusted_Stock_Price_2014": P2014,
        "Adjusted_Stock_Price_2024": P2024,
        "TSR_10yr_annualised": tsr_10yr
    })
    .reset_index(drop=True)
)

# Rank (highest TSR first); dense so ties share a rank and next rank increments by 1
result["Rank_by_TSR_10yr"] = result["TSR_10yr_annualised"].rank(ascending=False, method="dense").astype("Int64")

# Sort for presentation
result_sorted = result.sort_values(["TSR_10yr_annualised", "Company_name"], ascending=[False, True]).reset_index(drop=True)

# --- Save & display ---
result_sorted.to_csv(OUT_TSR_PATH, index=False)
print("Top 10 companies by 10-year annualised TSR (2014->2024):")
print(
    result_sorted[["Rank_by_TSR_10yr", "Company_name", "TSR_10yr_annualised"]]
    .head(10)
    .to_string(index=False, justify="left", float_format=lambda x: f"{x:.4f}")
)

# quick sanity: how many had both years present?
have_both = result_sorted["TSR_10yr_annualised"].notna().sum()
total_cos = result_sorted.shape[0]
print(f"\nComputed TSR for {have_both} out of {total_cos} companies (requires valid prices in both 2014 and 2024).")

# Keep your original throwaway vars if needed
x = 1
y = 2
