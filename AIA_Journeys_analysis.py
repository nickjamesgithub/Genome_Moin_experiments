import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Paths ---
in_path = Path(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\Insurance_data\journeys_asian_insurance\Journeys_summary_insurance.csv")
out_dir = Path(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\Insurance_data\journeys_asian_insurance\reports")
out_dir.mkdir(parents=True, exist_ok=True)

# --- Load ---
df = pd.read_csv(in_path).replace([np.inf, -np.inf], np.nan)

# --- Recreate genome labels from X/Y at begin & end (same thresholds you used) ---
def genome_label(x, y):
    if pd.isna(x) or pd.isna(y):
        return "UNKNOWN"
    if y < 0:
        if x < 0.00:            return "UNTENABLE"
        if 0.00 <= x <= 0.10:   return "TRAPPED"
        if 0.10 < x <= 0.20:    return "BRAVE"
        if x > 0.20:            return "FEARLESS"
    elif y > 0:
        if x < 0.00:            return "CHALLENGED"
        if 0.00 <= x <= 0.10:   return "VIRTUOUS"
        if 0.10 < x <= 0.20:    return "FAMOUS"
        if x > 0.20:            return "LEGENDARY"
    return "UNKNOWN"

# Ensure required columns exist
needed = {"X_beginning","Y_beginning","X_end","Y_end","Annualized_TSR_Capiq"}
missing = [c for c in needed if c not in df.columns]
if missing:
    raise KeyError(f"Journeys file missing columns: {missing}")

df["Genome_begin"] = [genome_label(x, y) for x, y in zip(df["X_beginning"], df["Y_beginning"])]
df["Genome_end"]   = [genome_label(x, y) for x, y in zip(df["X_end"],       df["Y_end"])]

# (Re)compute valuation % changes if not present
if "PBV_change_pct" not in df.columns:
    if {"Price_to_book_beginning","Price_to_book_end"}.issubset(df.columns):
        b, e = df["Price_to_book_beginning"], df["Price_to_book_end"]
        df["PBV_change_pct"] = np.where((b != 0) & ~b.isna() & ~e.isna(), e / b - 1.0, np.nan)

if "PE_change_pct" not in df.columns and {"PE_beginning","PE_end"}.issubset(df.columns):
    b, e = df["PE_beginning"], df["PE_end"]
    df["PE_change_pct"] = np.where((b != 0) & ~b.isna() & ~e.isna(), e / b - 1.0, np.nan)

tsr_col = "Annualized_TSR_Capiq"

# --- 1) Median TSR by starting Genome ---
med_tsr_by_genome_begin = (
    df.groupby("Genome_begin", dropna=False)[tsr_col]
      .median()
      .sort_values(ascending=False)
      .rename("Median_TSR")
)
med_tsr_by_genome_begin.to_csv(out_dir / "median_tsr_by_genome_begin.csv")

# (Optional) by ending genome too
med_tsr_by_genome_end = df.groupby("Genome_end", dropna=False)[tsr_col].median().sort_values(ascending=False)
med_tsr_by_genome_end.to_csv(out_dir / "median_tsr_by_genome_end.csv")

# --- 2) Median TSR by Journey ---
if "Journey" not in df.columns:
    raise KeyError("Missing 'Journey' column (Move_up/Move_down/Remain_*).")
med_tsr_by_journey = (
    df.groupby("Journey", dropna=False)[tsr_col]
      .median()
      .sort_values(ascending=False)
      .rename("Median_TSR")
)
med_tsr_by_journey.to_csv(out_dir / "median_tsr_by_journey.csv")

# --- 3) Avg % valuation change by Journey (PBV/PE) ---
agg_cols = [c for c in ["PBV_change_pct","PE_change_pct"] if c in df.columns]
if agg_cols:
    valn_change_by_journey = df.groupby("Journey", dropna=False)[agg_cols].mean()
    valn_change_by_journey.to_csv(out_dir / "avg_valuation_change_by_journey.csv")

# --- 4) ONE PLOT: Above-the-line starters (stay vs fall) ---
pos_start_mask = df["Y_beginning"] > 0
stay_mask = pos_start_mask & (df["Y_end"] >= 0)
fall_mask = pos_start_mask & (df["Y_end"] < 0)

stay = df.loc[stay_mask, tsr_col].dropna()
fall = df.loc[fall_mask, tsr_col].dropna()

print(f"Samples — start above line: stay={len(stay)}, fall={len(fall)}")

plt.figure(figsize=(9, 6))
n_bins = 15
# Use common bins so shapes are comparable
all_vals = pd.concat([stay, fall], ignore_index=True)
if not all_vals.empty:
    common_bins = np.histogram_bin_edges(all_vals, bins=n_bins)
else:
    common_bins = n_bins

stay_color = None
fall_color = None

# Plot histograms and capture facecolors for matching median lines
if not stay.empty:
    n_s, b_s, patches_s = plt.hist(stay, bins=common_bins, density=True, alpha=0.5,
                                   label=f"Stay above (n={len(stay)})")
    if patches_s:
        stay_color = patches_s[0].get_facecolor()  # RGBA
if not fall.empty:
    n_f, b_f, patches_f = plt.hist(fall, bins=common_bins, density=True, alpha=0.5,
                                   label=f"Fall below (n={len(fall)})")
    if patches_f:
        fall_color = patches_f[0].get_facecolor()  # RGBA

# Median lines in the SAME colors as their histograms
if not stay.empty:
    plt.axvline(stay.median(), linestyle="--", alpha=0.9, color=stay_color,
                label=f"Stay median: {stay.median():.2%}")
if not fall.empty:
    plt.axvline(fall.median(), linestyle="--", alpha=0.9, color=fall_color,
                label=f"Fall median: {fall.median():.2%}")

plt.title("Annualized TSR (CapIQ) — Above-the-line starters: Stay vs Fall")
plt.xlabel("Annualized TSR (CapIQ)")
plt.ylabel("Density")
plt.grid(True, alpha=0.25)
plt.legend()
plt.tight_layout()

plot_out = out_dir / "tsr_distribution_pos_starters_stay_vs_fall.png"
plt.savefig(plot_out, dpi=180)
plt.show()

# --- Compact summary CSV ---
summary = med_tsr_by_genome_begin.rename_axis("Genome_begin").to_frame()
summary = summary.join(med_tsr_by_journey.rename_axis("Journey").to_frame(), how="outer", rsuffix="_by_journey")
if agg_cols:
    summary = summary.join(valn_change_by_journey, how="outer")
summary.to_csv(out_dir / "summary_metrics.csv")

print("Wrote outputs to:")
print(" ", out_dir / "median_tsr_by_genome_begin.csv")
print(" ", out_dir / "median_tsr_by_genome_end.csv")
print(" ", out_dir / "median_tsr_by_journey.csv")
if agg_cols:
    print(" ", out_dir / "avg_valuation_change_by_journey.csv")
print(" ", out_dir / "tsr_distribution_pos_starters_stay_vs_fall.png")
print(" ", out_dir / "summary_metrics.csv")
