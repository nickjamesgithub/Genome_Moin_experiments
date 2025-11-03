import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib

matplotlib.use('TkAgg')

# --- PARAMETERS ---
start_year = 2014
end_year = 2024
tsr_period = end_year - start_year
quantile_cutoff = 0.75
top_pct_label = round((1 - quantile_cutoff) * 100)
sector_list = ["Communication Services"]

# 'Australia', 'Belgium', 'Canada', 'Chile', 'China', 'Denmark', 'France', 'Germany', 'Hong_Kong', 'India', 'Italy',
# 'Japan', 'Luxembourg', 'Malaysia', 'Netherlands', 'Philippines','Saudi_Arabia', 'Singapore', 'South_Korea', 'Sweden','Switzerland', 'Thailand', 'UAE', 'United_Kingdom', 'USA'

# Optional: filter by country, or set to None for all
selected_countries = [
    'media'
]
selected_countries = None  # ← Uncomment to include all countries

# --- FIRELFY TICKERS TO PLOT ---
firefly_tickers = ["ASX:NEC"]
# firefly_tickers = []

# --- LOAD DATA ---
file_path = r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\Nine\media_data_global_mapped.csv"
data = pd.read_csv(file_path)

# --- FUNCTION TO COMPUTE 10-YEAR TSR ---
def compute_10yr_tsr(df):
    try:
        p0 = df.loc[df["Year"] == start_year, "Adjusted_Stock_Price"].values[0]
        p1 = df.loc[df["Year"] == end_year, "Adjusted_Stock_Price"].values[0]
        if p0 > 0 and p1 > 0:
            return (p1 / p0) ** (1 / tsr_period) - 1
    except:
        return np.nan

# --- GENOME GRID SETUP ---
x_segments = [(-0.3, 0), (0, 0.1), (0.1, 0.2), (0.2, 0.3)]
y_segments = [(-0.3, 0), (0, 0.3)]
zone_labels = [
    "Untenable", "Challenged", "Trapped", "Virtuous",
    "Brave", "Famous", "Fearless", "Legendary"
]

fig, ax = plt.subplots(figsize=(10, 8))
label_counter = 0
for x_range in x_segments:
    for y_range in y_segments:
        rect = Rectangle(
            (x_range[0], y_range[0]),
            x_range[1] - x_range[0],
            y_range[1] - y_range[0],
            linewidth=0.3,
            edgecolor='black',
            facecolor='red',
            alpha=0.2
        )
        ax.add_patch(rect)
        ax.text(
            (x_range[0] + x_range[1]) / 2,
            (y_range[0] + y_range[1]) / 2,
            zone_labels[label_counter],
            ha='center', va='center',
            fontsize=8,
            color='black',
            fontweight='bold',
            rotation=15
        )
        label_counter += 1

# --- COLOR MAP ---
cmap = plt.get_cmap('tab10')
included_all = []
excluded_all = []

# --- LOOP THROUGH SECTORS ---
for i, sector in enumerate(sector_list):
    sector_data = data[data["Sector"] == sector].copy()
    if selected_countries:
        sector_data = sector_data[sector_data["Country"].isin(selected_countries)]

    tsr_dict = {
        ticker: compute_10yr_tsr(sector_data[sector_data["Ticker_full"] == ticker])
        for ticker in sector_data["Ticker_full"].unique()
    }
    tsr_series = pd.Series(tsr_dict).dropna()

    tsr_threshold = tsr_series.quantile(quantile_cutoff)
    candidate_tickers = tsr_series[tsr_series >= tsr_threshold].index

    valid_tickers = []
    for ticker in candidate_tickers:
        df_ticker = sector_data[
            (sector_data["Ticker_full"] == ticker) &
            (sector_data["Year"] >= start_year) &
            (sector_data["Year"] <= end_year)
        ]
        invalid = (
            (df_ticker["EVA_ratio_bespoke"] < -1.0) | (df_ticker["EVA_ratio_bespoke"] > 2.0) |
            (df_ticker["Revenue_growth_3_f"] < -0.6) | (df_ticker["Revenue_growth_3_f"] > 2.5) |
            (df_ticker["PBV"] <= -200)
        ).any()
        tsr_val = tsr_series[ticker]
        if not invalid and (-0.4 <= tsr_val <= 1.0):
            valid_tickers.append(ticker)

    included_tickers = pd.Index(valid_tickers)
    excluded_tickers = tsr_series.index.difference(included_tickers)

    df_included = sector_data[
        sector_data["Ticker_full"].isin(included_tickers) &
        (sector_data["Year"] >= start_year) &
        (sector_data["Year"] <= end_year)
    ].dropna(subset=["Revenue_growth_3_f", "EVA_ratio_bespoke"])

    if df_included.empty:
        print(f"⚠️ No valid data for {sector} after filtering.")
        continue

    x_vals = df_included["Revenue_growth_3_f"].values
    y_vals = df_included["EVA_ratio_bespoke"].values
    mean_x = np.mean(x_vals)
    mean_y = np.mean(y_vals)
    std_x = np.std(x_vals)
    std_y = np.std(y_vals)

    print(f"\n📊 {sector}")
    print(f"   Expected Revenue_growth_3_f: {mean_x:.2%} ± {std_x:.2%}")
    print(f"   Expected EVA_ratio_bespoke: {mean_y:.2%} ± {std_y:.2%}")

    color = cmap(i % 10)
    ax.plot(mean_x, mean_y, 'o', color=color, label=f"{sector} (Top {top_pct_label}%)")
    ax.text(
        mean_x + 0.01, mean_y + 0.01,
        f"{sector}\nRev: {mean_x:.1%}\nEVA: {mean_y:.1%}",
        fontsize=8, color=color, fontweight='bold',
        ha='left', va='bottom'
    )
    std_box = Rectangle(
        (mean_x - std_x, mean_y - std_y),
        2 * std_x, 2 * std_y,
        linewidth=1.5,
        edgecolor=color,
        facecolor='none',
        linestyle='--'
    )
    ax.add_patch(std_box)

    included_metadata = sector_data[sector_data["Ticker_full"].isin(included_tickers)] \
        .groupby("Ticker_full").first()[["Company_name"]].copy()
    included_metadata["Sector"] = sector
    included_metadata["10yr_TSR"] = tsr_series[included_metadata.index]
    included_metadata["Mean_EVA"] = mean_y
    included_metadata["Mean_Revenue_growth"] = mean_x
    included_all.append(included_metadata)

    excluded_metadata = sector_data[sector_data["Ticker_full"].isin(excluded_tickers)] \
        .groupby("Ticker_full").first()[["Company_name"]].copy()
    excluded_metadata["Sector"] = sector
    excluded_metadata["10yr_TSR"] = tsr_series[excluded_metadata.index]
    excluded_all.append(excluded_metadata)

# --- OUTPUT DATAFRAMES ---
included_df = pd.concat(included_all).sort_values(["Sector", "10yr_TSR"], ascending=[True, False])
excluded_df = pd.concat(excluded_all).sort_values(["Sector", "10yr_TSR"], ascending=[True, False])

print("\n✔️ Included Companies:")
print(included_df.head(10))

print("\n❌ Excluded Companies:")
print(excluded_df.head(10))

x=1
y=2
