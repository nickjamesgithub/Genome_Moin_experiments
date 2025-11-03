import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# ---------------------
# Config
# ---------------------
CSV_PATH = r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\Nine\Media_global_Journeys_summary_Global_FE_Update.csv"
unique_sectors = ["media"]
desired_sectors = unique_sectors
start_year = 2014
end_year = 2025
plot_label = "Media_transitions"

# Colors (matplotlib defaults)
COLOR_STAY = '#1f77b4'   # blue
COLOR_FALL = '#ff7f0e'   # orange

# ---------------------
# Load & filter
# ---------------------
df = pd.read_csv(CSV_PATH)

# Filter by sector and date range
# df_ = df[df["Sector"].isin(desired_sectors)].copy()
df_filtered = df[(df['Year_beginning'] >= start_year) & (df['Year_final'] <= end_year)]

# Define segments and groups
above_line_segments = ["CHALLENGED", "VIRTUOUS", "FAMOUS", "LEGENDARY"]

stay_above = df_filtered[
    (df_filtered["Genome_classification_bespoke_beginning"].isin(above_line_segments)) &
    (df_filtered["Genome_classification_bespoke_end"].isin(above_line_segments))
]

fall_below = df_filtered[
    (df_filtered["Genome_classification_bespoke_beginning"].isin(above_line_segments)) &
    (~df_filtered["Genome_classification_bespoke_end"].isin(above_line_segments))
]

# ---------------------
# Plot helpers
# ---------------------
def pct_fmt(x, _):
    return f'{x * 100:.0f}%'

# Distribution with matching colors for hist & mean line
def plot_distribution(data1, data2, label1, label2, column, title, xlabel, outfile):
    if data1.empty and data2.empty:
        print(f"No data to plot for {column}.")
        return

    plt.figure(figsize=(10, 6))

    # Group 1
    vals1 = data1[column].dropna()
    plt.hist(vals1, bins=20, alpha=0.6, label=label1, density=True, color=COLOR_STAY)
    m1 = vals1.mean() if not vals1.empty else np.nan
    if pd.notna(m1):
        plt.axvline(m1, color=COLOR_STAY, linestyle='--', linewidth=2,
                    label=f'{label1} Mean: {m1 * 100:.2f}%')

    # Group 2
    vals2 = data2[column].dropna()
    plt.hist(vals2, bins=20, alpha=0.6, label=label2, density=True, color=COLOR_FALL)
    m2 = vals2.mean() if not vals2.empty else np.nan
    if pd.notna(m2):
        plt.axvline(m2, color=COLOR_FALL, linestyle='--', linewidth=2,
                    label=f'{label2} Mean: {m2 * 100:.2f}%')

    plt.gca().xaxis.set_major_formatter(FuncFormatter(pct_fmt))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, axis='y', linestyle=':', linewidth=0.8)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.show()

# ---------------------
# Plots
# ---------------------
# Step 1: TSR distribution
plot_distribution(
    stay_above, fall_below,
    "Stayed Above the Line", "Fell Below the Line",
    "Annualized_TSR_Capiq",
    "Distribution of Annualized TSR: Stay Above vs Fall Below",
    "Annualized TSR",
    outfile=f"{plot_label}_TSR_Distribution.png"
)

