import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
matplotlib.use('TkAgg')

# Read the data
df = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Genome_code_250605\Genome-pipeline-code\Genome-pipeline-code\global_platform_data\Global_data.csv")

# Apply Genome Filter
genome_filtering = False
plot_label = "Australia"

# Desired sectors and date range
country_list = ["Australia"] # df["Country"].unique()  # "USA", 'AUS', 'INDIA', 'JAPAN', 'EURO', 'UK'
unique_sectors = df["Sector"].unique()
desired_sectors = unique_sectors
start_year = 2021
end_year = 2024
tsr_method = "capital_iq" # bain or capital_iq
make_plots = True

# 1. Filter by Country & Sector
df_merge = df[(df['Country'].isin(country_list)) & (df["Sector"]).isin(unique_sectors)]
# 2. Filter by date range
df_merge = df_merge[(df_merge['Year'] >= start_year) & (df_merge['Year'] <= end_year)]
# Create feature for Price-to-book
df_merge["Price_to_Book"] = df_merge["Market_Capitalisation"]/df_merge["Book_Value_Equity"]

# Get unique tickers
unique_tickers = df_merge["Ticker_full"].unique()


# Loop over unique tickers
dfs_concise_list = []
problem_ticker_list = []
for i in range(len(unique_tickers)):
    # Get unique ticker i
    ticker_i = unique_tickers[i]

    try:
        # Slice ticker_i within date range
        df_slice = df_merge.loc[(df_merge["Ticker_full"]==ticker_i) & (df_merge["Year"] >= start_year) & (df_merge["Year"] <= end_year)][["Year", "TSR", "Company_name", "Cost of Equity", "Stock_Price", "Adjusted_Stock_Price", "DPS", "BBPS", "DBBPS",
                                                                       "EVA_ratio_bespoke", "Revenue_growth_3_f", "Price_to_Book",  "Dividend_Yield", "Buyback_Yield"]]

        # Get Cumulative TSR components + Average yields
        company_name_i = df_slice["Company_name"].unique()[0]

        # Get Final year EP/FE and Revenue growth
        eva_final = df_slice.loc[df_slice["Year"]==end_year]["EVA_ratio_bespoke"].iloc[0]
        revenue_growth_final = df_slice.loc[df_slice["Year"] == end_year]["Revenue_growth_3_f"].iloc[0]
        price_to_book_final = df_slice.loc[df_slice["Year"] == end_year]["Price_to_Book"].iloc[0]

        # Get unadjusted stock prices
        beginning_price = df_slice.loc[df_slice["Year"]==start_year]["Stock_Price"].values[0]
        final_price = df_slice.loc[df_slice["Year"]==end_year]["Stock_Price"].values[0]

        # Get adjusted stock prices
        beginning_price_adjusted = df_slice.loc[df_slice["Year"]==start_year]["Adjusted_Stock_Price"].values[0]
        final_price_adjusted = df_slice.loc[df_slice["Year"]==end_year]["Adjusted_Stock_Price"].values[0]

        # Cumulative dividends and buybacks per share
        cumulative_dps = np.sum(np.nan_to_num(df_slice.loc[(df_slice["Year"]>start_year) & (df_slice["Year"]<=end_year)]["DPS"]))
        cumulative_bbps = np.sum(np.nan_to_num(df_slice.loc[(df_slice["Year"]>start_year) & (df_slice["Year"]<=end_year)]["BBPS"]))

        # Average dividend and buyback yield
        avg_dividend_yield = df_slice.loc[(df_slice["Year"]>start_year) & (df_slice["Year"]<=end_year)]["Dividend_Yield"].mean()
        avg_buyback_yield = df_slice.loc[(df_slice["Year"]>start_year) & (df_slice["Year"]<=end_year)]["Buyback_Yield"].mean()

        # Compute annualized TSR with Bain method
        cumulative_tsr_bain = (final_price - beginning_price + cumulative_bbps + cumulative_dps)/beginning_price
        n = len(df_slice)-1
        annualized_tsr_bain = (1+cumulative_tsr_bain)**(1/n) - 1

        # Compute annualized TSR with Capital IQ method
        cumulative_tsr_ciq = final_price_adjusted/beginning_price_adjusted - 1
        annualized_tsr_ciq = (1 + cumulative_tsr_ciq)**(1/n) - 1

        # Print unique ticker & annualized TSR
        print("Ticker is ", unique_tickers[i] + " with annualized Bain TSR of " + str(annualized_tsr_bain))

        # Create a DataFrame for each row
        df_concise = pd.DataFrame({
            "Company": [company_name_i],
            "Ticker": [ticker_i],
            "Annualized_TSR_CIQ": [annualized_tsr_ciq],
            "Avg_dividend_yield": [avg_dividend_yield],
            "Avg_buyback_yield": [avg_buyback_yield],
            "EVA_final": [eva_final],
            "Revenue_growth_final": [revenue_growth_final],
            "Price_to_Book": [price_to_book_final]
        })

        # Append dictionary to storing list
        dfs_concise_list.append(df_concise)
        print("Unique ticker ", unique_tickers[i])
    except:
        print("Issue with company ", ticker_i)
        problem_ticker_list.append(ticker_i)

# Collapse dataframes into flat file
df_flat = pd.concat(dfs_concise_list)
df_flat.loc[:, ["Avg_dividend_yield", "Avg_buyback_yield"]] = df_flat.loc[:, ["Avg_dividend_yield", "Avg_buyback_yield"]].replace([np.inf, -np.inf], np.nan)

# Remove N/A values
df_flat.loc[df_flat['Avg_dividend_yield'] > 1, 'Avg_dividend_yield'] = np.nan
df_flat.loc[df_flat['Avg_buyback_yield'] > 1, 'Avg_buyback_yield'] = np.nan

# # Apply standard Genome Filtering
# if genome_filtering:
#     df_flat = df_flat.loc[(df_flat["EVA_final"] >= -.3) & (df_flat["EVA_final"] <= .5) &
#                                      (df_flat["Revenue_growth_final"] >= -.3) & (df_flat["Revenue_growth_final"] <= 1.5) &
#                                      (df_flat["Annualized_TSR_CIQ"] >= -.4) & (df_flat["Annualized_TSR_CIQ"] <= 1) &
#                                      (df_flat["Price_to_Book"] > -200)]

if tsr_method == "capital_iq":
    # Remove all entries where TSR = -1
    df_flat = df_flat[df_flat["Annualized_TSR_CIQ"] != -1]
    # Sample data
    sorted_tsr = df_flat["Annualized_TSR_CIQ"].sort_values(ascending=False)

# Filter out non-finite values
sorted_tsr = sorted_tsr[np.isfinite(sorted_tsr)]
print("Number of companies ", len(sorted_tsr))

# Calculate percentiles
q_75 = np.percentile(sorted_tsr, 75)
q_50 = np.percentile(sorted_tsr, 50)
q_25 = np.percentile(sorted_tsr, 25)

# Plot ordered data as a line plot
plt.plot(sorted_tsr.values, np.arange(len(sorted_tsr)), marker='o')

# Annotate percentiles with vertical lines
plt.axvline(x=q_75, color='red', linestyle='--', label=f'75th percentile = {q_75 * 100:.2f}%')
plt.axvline(x=q_50, color='green', linestyle='--', label=f'50th percentile = {q_50 * 100:.2f}%')
plt.axvline(x=q_25, color='purple', linestyle='--', label=f'25th percentile = {q_25 * 100:.2f}%')

# Add legend
plt.legend()

# Add labels and title
plt.xlabel("Annualized TSR")
plt.ylabel("Company ranking")
plt.title(plot_label + "_" + str(start_year) + "-" + str(end_year))
# Show plot
plt.savefig(plot_label + "_" + str(start_year) + "-" + str(end_year))
plt.show()
