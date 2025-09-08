import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------- Paths --------
in_path   = Path(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\Insurance_data\merged_insurance_data.csv")
out_path  = Path(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\Insurance_data\genome_embeddings_gwp3y_roe_hurdle.csv")
out_share = Path(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\Insurance_data\segment_share_gwp3y_roe_hurdle.csv")
out_tsr   = Path(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\Insurance_data\median_tsr_by_genome_gwp3y_roe_hurdle.csv")
plot_path = Path(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\Insurance_data\genome_embeddings_gwp3y_roe_hurdle.png")
out_full  = Path(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\Insurance_data\merged_insurance_data_with_genome.csv")

# -------- Country filter (configure here) --------
# Leave empty [] to disable. Example list shown; edit to your needs.
FILTERED_COUNTRIES = [
    'Hong Kong', 'China', 'Hong Kong/China',
    'Singapore', 'Japan', 'India', 'South Korea', 'Thailand',
    'Australia', 'Vietnam', 'Malaysia', 'Indonesia'
]

# -------- Load & year filter --------
df_ = pd.read_csv(in_path)

# Year numeric…
if "Year" in df_.columns:
    df_["Year"] = pd.to_numeric(df_["Year"], errors="coerce").astype("Int64")

# Coerce join keys
for k in ["Company_name", "Type", "Home_Country"]:
    if k in df_.columns:
        df_[k] = df_[k].astype("string")

# -------- Optional: restrict to specific Home_Country values --------
if FILTERED_COUNTRIES and "Home_Country" in df_.columns:
    keep = {c.strip().casefold() for c in FILTERED_COUNTRIES}
    hc_norm = df_["Home_Country"].astype("string").str.strip().str.casefold()
    before_n = len(df_)
    df_ = df_.loc[hc_norm.isin(keep)].copy()
    after_n = len(df_)
    print(f"Applied Home_Country filter: kept {after_n:,} of {before_n:,} rows "
          f"across {sorted(FILTERED_COUNTRIES)}")
elif "Home_Country" not in df_.columns:
    print("Warning: 'Home_Country' column not found — skipping country filter.")

# Filtered/engineered working frame
df = df_.loc[(df_["Year"] >= 2012) & (df_["Year"] <= 2023)].copy()

# -------- Helper to coerce %/decimal to decimal --------
def to_decimal(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    med = s.dropna().median()
    # If looks like percentage points (e.g., 12 = 12%), convert to decimal
    if pd.notna(med) and med > 1.5:
        s = s / 100.0
    return s

# -------- Build 3-year rolling arithmetic mean of GWP growth (per company) --------
# Requires GWP_growth_BCN (1y growth) to exist; if not, compute it from GWP_BCN:
if "GWP_growth_BCN" not in df.columns and "GWP_BCN" in df.columns:
    df = df.sort_values(["Company_name", "Year"]).reset_index(drop=True)
    df["GWP_growth_BCN"] = df.groupby("Company_name")["GWP_BCN"].pct_change(fill_method=None)
    df["GWP_growth_BCN"].replace([np.inf, -np.inf], np.nan, inplace=True)

df = df.sort_values(["Company_name", "Year"]).reset_index(drop=True)
if "GWP_growth_BCN" in df.columns:
    df["GWP_growth_3y"] = (
        df.groupby("Company_name")["GWP_growth_BCN"]
          .transform(lambda s: s.rolling(window=3, min_periods=3).mean())
    )
else:
    df["GWP_growth_3y"] = np.nan  # graceful fallback if growth not available

# -------- Normalize ROE & Cost of Equity; compute ROE_above_hurdle --------
if "ROE_BCN" in df.columns:
    df["ROE_BCN"] = to_decimal(df["ROE_BCN"])
else:
    df["ROE_BCN"] = np.nan

if "Cost of Equity" not in df.columns:
    raise KeyError("Expected 'Cost of Equity' column in the dataframe.")
df["Cost of Equity"] = to_decimal(df["Cost of Equity"])

df["ROE_above_hurdle"] = df["ROE_BCN"] - df["Cost of Equity"]

# Keep join key dtypes consistent on the engineered frame
for k in ["Company_name", "Type", "Home_Country"]:
    if k in df.columns:
        df[k] = df[k].astype("string")

# -------- Genome classification (ROE_above_hurdle vs GWP_growth_3y) --------
seg_col = "Genome_classification_gwp3y_roe_hurdle"
growth = df["GWP_growth_3y"]
roe_gap = df["ROE_above_hurdle"]

conditions = [
    (roe_gap < 0) & (growth < 0.00),
    (roe_gap < 0) & (growth.between(0.00, 0.10, inclusive="right")),
    (roe_gap < 0) & (growth.between(0.10, 0.20, inclusive="right")),
    (roe_gap < 0) & (growth >= 0.20),
    (roe_gap > 0) & (growth < 0.00),
    (roe_gap > 0) & (growth.between(0.00, 0.10, inclusive="right")),
    (roe_gap > 0) & (growth.between(0.10, 0.20, inclusive="right")),
    (roe_gap > 0) & (growth >= 0.20),
]
labels = ["UNTENABLE", "TRAPPED", "BRAVE", "FEARLESS",
          "CHALLENGED", "VIRTUOUS", "FAMOUS", "LEGENDARY"]

df[seg_col] = np.select(conditions, labels, default="UNKNOWN")

# -------- Segment percentages --------
segment_share = (df[seg_col].value_counts(normalize=True) * 100).round(2)
print("Segment share (% of rows):")
print(segment_share)
segment_share.rename("Percent").to_csv(out_share)
print(f"\nSaved segment share -> {out_share}")

# -------- Median TSR per segment --------
if "TSR_CIQ_no_buybacks" in df.columns:
    df["TSR_CIQ_no_buybacks"] = to_decimal(df["TSR_CIQ_no_buybacks"])
    median_tsr = df.groupby(seg_col)["TSR_CIQ_no_buybacks"].median().sort_index()
    print("\nMedian TSR_CIQ_no_buybacks by segment (3y growth):")
    print(median_tsr)
    median_tsr.rename("Median_TSR_CIQ_no_buybacks").to_csv(out_tsr)
    print(f"\nSaved median TSR by segment -> {out_tsr}")
else:
    print("\nWarning: 'TSR_CIQ_no_buybacks' not found. Skipping medians.")

# -------- Save enriched (slim) output --------
cols_to_keep = [
    "Company_name", "Type", "Home_Country", "Year", "Sector", "Ticker",
    "GWP_growth_BCN", "GWP_growth_3y",
    "ROE_BCN", "Cost of Equity", "ROE_above_hurdle",
    "TSR_CIQ_no_buybacks", seg_col
]
existing_cols = [c for c in cols_to_keep if c in df.columns]
df_out = df[existing_cols].copy()
df_out.to_csv(out_path, index=False)
print(f"\nSaved genome embeddings to: {out_path}")

# -------- Write full merged data + genome annotation (ALL original cols/rows) --------
# Columns from engineered df to append back to the raw merged dataset
anno_cols = ["Company_name", "Year", seg_col, "GWP_growth_3y", "ROE_above_hurdle", "ROE_BCN", "Cost of Equity"]
anno_cols = [c for c in anno_cols if c in df.columns]

# Prefer a more specific join if both frames have Type and Home_Country
join_keys = ["Company_name", "Year"]
for k in ["Type", "Home_Country"]:
    if (k in df_.columns) and (k in df.columns):
        join_keys.append(k)

# Ensure the right-side subset includes the join keys
right_cols = sorted(set(anno_cols + join_keys))
df_full = df_.merge(df[right_cols], on=join_keys, how="left")

# If a row couldn't be classified (e.g., insufficient lookback), mark as UNKNOWN
df_full[seg_col] = df_full[seg_col].fillna("UNKNOWN")

df_full.to_csv(out_full, index=False)
print(f"Saved full merged data with genome annotation -> {out_full}")

# -------- Scatter plot (x = GWP 3-year rolling mean) --------
plt.figure(figsize=(10,7))
class_colors = {
    "UNTENABLE": "#d62728",
    "TRAPPED": "#ff7f0e",
    "BRAVE": "#bcbd22",
    "FEARLESS": "#2ca02c",
    "CHALLENGED": "#9467bd",
    "VIRTUOUS": "#1f77b4",
    "FAMOUS": "#17becf",
    "LEGENDARY": "#8c564b",
    "UNKNOWN": "#7f7f7f"
}

for cls, sub in df.groupby(seg_col):
    plt.scatter(
        sub["GWP_growth_3y"], sub["ROE_above_hurdle"],
        s=25, alpha=0.7, label=cls,
        c=class_colors.get(cls, "#7f7f7f")
    )

# Guide lines
plt.axvline(0.00, color="k", linewidth=1, alpha=0.4)
plt.axvline(0.10, color="k", linewidth=0.8, alpha=0.3, linestyle="--")
plt.axvline(0.20, color="k", linewidth=0.8, alpha=0.3, linestyle="--")
plt.axhline(0.00, color="k", linewidth=1, alpha=0.4)

plt.title("Genome Embeddings: 3Y GWP Growth (rolling mean) vs (ROE − Cost of Equity)")
plt.xlabel("3-Year Rolling GWP Growth (arithmetic mean)")
plt.ylabel("ROE − Cost of Equity")
plt.xlim(-0.5, 0.5)
plt.legend(title="Genome", fontsize=8, frameon=True)
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(plot_path, dpi=200)
plt.show()
print(f"Saved plot to: {plot_path}")
