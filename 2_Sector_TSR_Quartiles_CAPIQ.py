import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
from matplotlib.patches import Rectangle
from sklearn.ensemble import RandomForestRegressor
import shap
from Utilities import compute_percentiles
import matplotlib
from Utilities import generate_market_cap_class, waterfall_value_plot, geometric_return
matplotlib.use('TkAgg')

"""
This is a script to compute the waterfall for an entire sector, or collection of sectors
"""

# Import data
data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\Global_data.csv")
# Define countries and sectors to include
countries_to_include = ["EURO"] # 'USA', 'AUS', 'INDIA', 'JAPAN', 'EURO', 'UK'
sectors_to_include = ['Technology', "Industrials"]
plot_label = "AUS Technology + Industrials"

# Filter data based on countries and sectors
filtered_data = data.loc[(data['Country'].isin(countries_to_include)) & (data['Sector'].isin(sectors_to_include))]

# Required tickers
tickers_ = np.unique(filtered_data["Ticker"].values)

# List of year
year_lb = 2014
year_ub = 2024
year_grid = np.linspace(year_lb, year_ub, year_ub-year_lb+1)

## Create TSR DataFrame with consistent alignment
tsr_list = []

for i in range(len(tickers_)):
    print("Processing ticker:", tickers_[i])
    company_i = tickers_[i]
    # Filter data for the current ticker
    df_i = filtered_data.loc[filtered_data["Ticker"] == company_i]

    # Create a dictionary mapping years to TSR values
    tsr_dict = dict(zip(df_i["Year"], df_i["TSR_CIQ_no_buybacks"]))

    # Align TSR with year_grid (fill missing years with NaN)
    aligned_tsr = [tsr_dict.get(year, np.nan) for year in year_grid]
    tsr_list.append(aligned_tsr)


# Convert TSR list to DataFrame with years as columns
tsr_df = pd.DataFrame(tsr_list, columns=year_grid, index=tickers_)

# Replace Inf with NaN
tsr_df = tsr_df.replace([np.inf, -np.inf], np.nan)

# Ensure column names are numeric
# Convert the columns (year_grid) to numeric, forcing invalid values to NaN
valid_columns = pd.to_numeric(tsr_df.columns, errors='coerce')  # Convert to numeric, invalid columns become NaN

# Filter the valid columns, retaining only numeric ones
tsr_df = tsr_df.loc[:, ~np.isnan(valid_columns)]  # Keep only columns where valid_columns is not NaN
tsr_df.columns = valid_columns[~np.isnan(valid_columns)]  # Update columns to numeric valid years

# Drop columns (years) with no valid data
tsr_df = tsr_df.dropna(axis=1, how='all')

# Plot the boxplots
fig, ax = plt.subplots(figsize=(10, 6))

# Create boxplot, automatically ignoring NaN values in computations
bp = ax.boxplot([tsr_df[col].dropna() for col in tsr_df.columns], showfliers=False)

# Set the x-tick labels to match the years
ax.set_xticks(range(1, len(tsr_df.columns) + 1))  # Boxplot positions start at 1
ax.set_xticklabels(tsr_df.columns.astype(int), rotation=45)

# Add labels and title
plt.ylabel("TSR")
plt.xlabel("Year")
plt.title(f"{plot_label} TSR Quartiles")
plt.tight_layout()

# Save and show the plot
plt.savefig(f"{plot_label}_TSR_Quartiles.png")
plt.show()
