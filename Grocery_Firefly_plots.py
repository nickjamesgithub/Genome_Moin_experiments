import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
from matplotlib.patches import Rectangle
import matplotlib
matplotlib.use('TkAgg')

excel_output = True

# --- LOAD DATA ---
data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\grocery_data.csv")

# Full ticker list and corresponding start/end years
full_ticker_list = ["NASDAQGS:COST", "NASDAQGS:CASY", "NYSE:WM", "NASDAQGS:AMZN"]
start_years = [2019, 2019, 2019, 2019]
end_years = [2025, 2025, 2025, 2025]
plot_label = "Grocery_leaders"

# Extract company names before looping
company_name_list = [data.loc[data["Ticker_full"] == ticker, "Company_name"].iloc[0] for ticker in full_ticker_list]

x_axis_list = []
y_axis_list = []
labels_list = []

# (Optional pre-filter to just the tickers of interest — keeps your dataset small)
df = data.loc[data["Ticker_full"].isin(full_ticker_list)]

for i, full_ticker in enumerate(full_ticker_list):
    idx_i, company_i = full_ticker.split(":")

    # Retrieve the correct country for the ticker
    country_i = data.loc[data["Ticker_full"] == full_ticker, "Country"].iloc[0]

    # Load company data from the correct country subfolder
    # (You had this set to 'df = data.loc[...]' each iteration; leaving your df above)
    # df already filtered to the tickers of interest

    # >>> FIX: filter by THIS ticker AND the time window (sorting by Year helps)
    company_slice = (
        df[(df["Ticker_full"] == full_ticker) &
           (df["Year"] >= start_years[i]) &
           (df["Year"] <= end_years[i])]
        .sort_values("Year")
    )

    # Extract relevant data
    labels = company_slice["Year"].values
    x_axis = company_slice["Revenue_growth_3_f"].values
    y_axis = company_slice["EVA_ratio_bespoke"].values

    # Append results
    x_axis_list.append(x_axis.tolist())
    y_axis_list.append(y_axis.tolist())
    labels_list.append(labels.tolist())

# Generate the grid for each axis
x_pad = max(map(len, x_axis_list)) if x_axis_list else 0
y_pad = max(map(len, y_axis_list)) if y_axis_list else 0
labels_pad = max(map(len, labels_list)) if labels_list else 0

x_fill_list = np.array([i + [np.nan] * (x_pad - len(i)) for i in x_axis_list]) if x_pad > 0 else np.array([])
y_fill_list = np.array([i + [np.nan] * (y_pad - len(i)) for i in y_axis_list]) if y_pad > 0 else np.array([])
labels_fill_list = np.array([i + [np.nan] * (labels_pad - len(i)) for i in labels_list]) if labels_pad > 0 else np.array([])

# Handle case where arrays could be empty (defensive)
if x_fill_list.size > 0:
    x = np.linspace(np.nanmin(x_fill_list), np.nanmax(x_fill_list), 100)
else:
    x = np.array([-.3, .3])
if y_fill_list.size > 0:
    y = np.linspace(np.nanmin(y_fill_list), np.nanmax(y_fill_list), 100)
else:
    y = np.array([-.3, .3])

# Set automatic parameters for plotting
x_lb = min(-.3, np.nanmin(x))
x_ub = max(.3, np.nanmax(x))
y_lb = min(-.3, np.nanmin(y))
y_ub = max(.3, np.nanmax(y))

x_segment_ranges = [(x_lb, 0), (0, .1), (.1, .2), (.2, x_ub)]
y_segment_ranges = [(y_lb, 0), (0, y_ub)]
label_counter = 0
labels = ["Untenable", "Challenged", "Trapped", "Virtuous", "Brave", "Famous", "Fearless", "Legendary"]

fig, ax = plt.subplots()

# Plot each company with proper labels
for i in range(len(x_fill_list)):
    if i == 0:
        plt.plot(x_axis_list[i], y_axis_list[i], '-o', label=company_name_list[i], color="blue")
    else:
        plt.plot(x_axis_list[i], y_axis_list[i], '-o', label=company_name_list[i], alpha=0.4, linestyle='--')

    # >>> FIX: safe annotation only when coordinates are finite
    for j in range(len(labels_list[i])):
        xx = x_axis_list[i][j]
        yy = y_axis_list[i][j]
        if np.isfinite(xx) and np.isfinite(yy):
            plt.annotate(labels_list[i][j], (xx, yy), fontsize=6)

# Draw segmented rectangles and add labels
for x_range in x_segment_ranges:
    for y_range in y_segment_ranges:
        rect = Rectangle((x_range[0], y_range[0]), x_range[1] - x_range[0], y_range[1] - y_range[0],
                         linewidth=0.3, edgecolor='black', facecolor='red', alpha=0.5)
        ax.add_patch(rect)
        label = labels[label_counter]
        ax.text((x_range[0] + x_range[1]) / 2, (y_range[0] + y_range[1]) / 2, label,
                ha='center', va='center', color='black', fontsize=8, fontweight='bold', rotation=15)
        label_counter += 1

plt.title(plot_label)
plt.xlabel("Revenue growth (3 year moving average)")
plt.ylabel("EVA Ratio")
plt.legend()
plt.savefig(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\Woolworths\Firefly_plot_CAPIQ_" + plot_label)
plt.show()

# Export to CSV if enabled
if excel_output:
    x_stacked_array = np.vstack(x_fill_list)
    y_stacked_array = np.vstack(y_fill_list)
    labels_stacked_array = np.vstack(labels_fill_list)

    x_flat = x_stacked_array.flatten()
    y_flat = y_stacked_array.flatten()
    labels_flat = labels_stacked_array.flatten()

    def create_marker_array(rows, cols):
        marker_array = np.zeros((rows, cols), dtype=int)
        for i in range(rows):
            marker_array[i, :] = i + 1
        return marker_array.reshape(-1, 1)

    rows, cols = len(labels_fill_list), len(labels_fill_list[0])
    marker_array = create_marker_array(rows, cols).flatten()

    df = pd.DataFrame({
        'Series Labels': labels_flat,
        'X': x_flat,
        'Y': y_flat,
        'Marker and regression grouping': marker_array
    })

    df.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\Woolworths\Firefly_plot_CAPIQ_" + plot_label + ".csv")
