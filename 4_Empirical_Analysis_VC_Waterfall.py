import numpy as np
import pandas as pd
from Utilities import compute_percentiles, firefly_plot, geometric_return
import matplotlib
from Utilities import generate_market_cap_class, waterfall_value_plot, geometric_return, dendrogram_plot, get_even_clusters
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import math
import glob
import os

make_plots = True

# Apply Genome Filter
genome_filtering = False
sp_500 = True

# Market capitalisation threshold
mcap_threshold = 500

def generate_genome_classification_df(df):
    # Conditions EP/FE
    conditions_genome = [
        (df["EP/FE"] < 0) & (df["Revenue_growth_3_f"] < 0),
        (df["EP/FE"] < 0) & (df["Revenue_growth_3_f"].between(0, 0.10, inclusive='right')),
        (df["EP/FE"] < 0) & (df["Revenue_growth_3_f"].between(0.10, 0.20, inclusive='right')),
        (df["EP/FE"] < 0) & (df["Revenue_growth_3_f"] >= 0.20),
        (df["EP/FE"] > 0) & (df["Revenue_growth_3_f"] < 0),
        (df["EP/FE"] > 0) & (df["Revenue_growth_3_f"].between(0, 0.10, inclusive='right')),
        (df["EP/FE"] > 0) & (df["Revenue_growth_3_f"].between(0.10, 0.20, inclusive='right')),
        (df["EP/FE"] > 0) & (df["Revenue_growth_3_f"] >= 0.20)
    ]

    # Values to display
    values_genome = ["UNTENABLE", "TRAPPED", "BRAVE", "FEARLESS", "CHALLENGED", "VIRTUOUS", "FAMOUS", "LEGENDARY"]

    df["Genome_classification"] = np.select(conditions_genome, values_genome)

    return df

matplotlib.use('TkAgg')

# Initialise years
beginning_year = 2011
end_year = 2024
# Generate grid of years
year_grid = np.linspace(beginning_year, end_year, end_year-beginning_year+1)
rolling_window = 3

# Import data
mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Company_list_GPT_SP500.csv")

# Get unique tickers
unique_tickers = mapping_data["Ticker"].unique()

# Choose sectors to include
sector = mapping_data["Sector_new"].values

# Required tickers
tickers_ = mapping_data.loc[mapping_data["Sector_new"].isin(sector)]["Ticker"].values

if sp_500:
    dfs_list = []
    for i in range(len(tickers_)):
        company_i = tickers_[i]
        try:
            df = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\USA_platform_data\_" + company_i + ".csv")
            dfs_list.append(df)
            print("Company data ", company_i)
        except:
            print("Error with company ", company_i)

# Merge dataframes
df_concat = pd.concat(dfs_list)
df_merge = generate_genome_classification_df(df_concat)
# Create feature for Price-to-book
df_merge["Price_to_Book"] = df_merge["Market_Capitalisation"]/df_merge["Book_Value_Equity"]
unique_tickers_100 = df_merge["Ticker"].unique()

# Collect data for 2023 and 2013
init_year = 2014
final_year = 2024
fy_23_market = []
fy_13_market = []

for ticker in unique_tickers_100:
    # Dataframe slice for ticker in year 2023
    df_slice_23 = df_merge.loc[(df_merge["Ticker"] == ticker) & (df_merge["Year"] == final_year)][
        ["Ticker", "Company_name", "Stock_Price", "Shares_outstanding", "Revenue", "NPAT", "Diluted_EPS", "Market_Capitalisation"]]
    # Infer P/E
    df_slice_23["P/E"] = df_slice_23["Stock_Price"] / df_slice_23["Diluted_EPS"]

    # Dataframe slice for ticker in year 2013
    df_slice_13 = df_merge.loc[(df_merge["Ticker"] == ticker) & (df_merge["Year"] == init_year)][
        ["Ticker", "Company_name", "Stock_Price", "Shares_outstanding", "Revenue", "NPAT", "Diluted_EPS", "Market_Capitalisation"]]
    # Infer P/E
    df_slice_13["P/E"] = df_slice_13["Stock_Price"] / df_slice_13["Diluted_EPS"]

    # Check if stock price is 0 in either year
    if not df_slice_23.empty and not df_slice_13.empty:
        if df_slice_23["Stock_Price"].iloc[0] != 0 and df_slice_13["Stock_Price"].iloc[0] != 0:
            fy_23_market.append(df_slice_23)
            fy_13_market.append(df_slice_13)

# Turn FY23 and FY13 into dataframes
fy_23_df = pd.concat(fy_23_market, axis=0).reset_index(drop=True)
fy_13_df = pd.concat(fy_13_market, axis=0).reset_index(drop=True)

# Calculate metrics for FY 2023
fy_23_market_cap = fy_23_df["Market_Capitalisation"].sum()
fy_23_revenue = fy_23_df["Revenue"].sum()
fy_23_npat = fy_23_df["NPAT"].sum()
fy_23_margin = fy_23_npat / fy_23_revenue
fy_23_market_pe = fy_23_market_cap / fy_23_npat

# Calculate metrics for FY 2013
fy_13_market_cap = fy_13_df["Market_Capitalisation"].sum()
fy_13_revenue = fy_13_df["Revenue"].sum()
fy_13_npat = fy_13_df["NPAT"].sum()
fy_13_margin = fy_13_npat / fy_13_revenue
fy_13_market_pe = fy_13_market_cap / fy_13_npat

# Calculate incremental values
incremental_revenue = (fy_23_revenue - fy_13_revenue) * fy_13_margin * fy_13_market_pe
incremental_margin = fy_23_revenue * (fy_23_margin - fy_13_margin) * fy_13_market_pe
incremental_pe = fy_23_revenue * fy_23_margin * (fy_23_market_pe - fy_13_market_pe)

# Total change in market value
total_change = fy_23_market_cap - fy_13_market_cap

# Verify that the total incremental values add up to the total change
assert round(incremental_revenue + incremental_margin + incremental_pe, 2) == round(total_change, 2), "Incremental values do not sum up to total change."

# Create waterfall chart
components = ['Initial Market Value', 'Revenue Growth', 'Margin Change', 'P/E Change', 'Final Market Value']
values = [fy_13_market_cap, incremental_revenue, incremental_margin, incremental_pe, fy_23_market_cap]
cumulative_values = [sum(values[:i+1]) for i in range(len(values))]

fig, ax = plt.subplots()
bar_width = 0.4

# Plot bars
for i in range(len(components)):
    if i == 0:
        ax.bar(i, values[i], bar_width, color='black')
    elif i == len(components) - 1:
        ax.bar(i, values[i], bar_width, color='black')
    else:
        color = 'green' if values[i] > 0 else 'red'
        ax.bar(i, values[i], bar_width, bottom=cumulative_values[i-1], color=color)

# Add labels
ax.set_xticks(range(len(components)))
ax.set_xticklabels(components)
ax.set_ylabel('Market Value')
ax.set_title('Waterfall Chart of Market Value Changes:'+str(init_year) + "-" + str(final_year))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Market_Value_Waterfall")
plt.show()