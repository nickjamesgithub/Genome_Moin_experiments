import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import linregress
matplotlib.use('TkAgg')

# Read the data
df_full = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\Global_data.csv")
company_ticker = "ASX:CGF"

# Desired sectors and date range
country_list = ['AUS'] # "USA", 'AUS', 'INDIA', 'JAPAN', 'EURO', 'UK'
unique_sectors = ['Investment and Wealth'] # df_full["Sector"].unique()
desired_sectors = unique_sectors

# 1. Filter by Country & Sector
df_slice = df_full[(df_full['Country'].isin(country_list)) & (df_full["Sector"].isin(unique_sectors))]
tickers = df_slice["Ticker_full"].unique()

# Company slice
target_company = df_slice.loc[(df_slice["Ticker_full"]==company_ticker)]
return_metric = "EVA_ratio_bespoke"
growth_metric = "Revenue_growth_3_f"
print("Ticker iteration ", company_ticker)

try:
    # Slice based on metric and company
    sector_i = df_slice.loc[df_slice["Ticker_full"]==company_ticker]["Sector"].unique()[0]
    company_name_i = df_slice.loc[df_slice["Ticker_full"]==company_ticker]["Company_name"].unique()[0]

    # Get sector return and growth
    df_sector = df_slice.loc[df_slice["Sector"]==sector_i]
    df_company = df_slice.loc[df_slice["Ticker_full"]==company_ticker]

    # Growth & Return average
    company_i_return = df_company[return_metric].values
    company_i_growth = df_company[growth_metric].values

    # Sector
    sector_return_avg = df_sector.groupby("Year")[return_metric].mean().values
    sector_growth_avg = df_sector.groupby("Year")[growth_metric].mean().values

    # Remove NaN values
    company_i_return = company_i_return[~np.isnan(company_i_return)]
    company_i_growth = company_i_growth[~np.isnan(company_i_growth)]
    sector_return_avg = sector_return_avg[~np.isnan(sector_return_avg)]
    sector_growth_avg = sector_growth_avg[~np.isnan(sector_growth_avg)]


    def company_decoupling(company_name_i, company_vector, peer_vector, peer_group, metric):
        # Convert to DataFrame and align based on Year
        df_company = pd.DataFrame({metric: company_vector}, index=range(len(company_vector)))
        df_peer = pd.DataFrame({metric: peer_vector}, index=range(len(peer_vector)))

        # Merge on index to align the data (ensuring same shape)
        df_merged = pd.merge(df_company, df_peer, left_index=True, right_index=True, how="inner",
                             suffixes=("_company", "_peer"))

        # Extract aligned data
        company_vector_aligned = df_merged[f"{metric}_company"].values
        peer_vector_aligned = df_merged[f"{metric}_peer"].values

        # Perform linear regression (only if we have data left after alignment)
        if len(company_vector_aligned) == len(peer_vector_aligned) and len(company_vector_aligned) > 1:
            slope, intercept, r_value, p_value, std_err = linregress(peer_vector_aligned, company_vector_aligned)

            # Create scatter plot
            plt.scatter(peer_vector_aligned, company_vector_aligned, label='Data')

            # Plot the line of best fit
            plt.plot(peer_vector_aligned, intercept + slope * peer_vector_aligned, color='red', label='Regression')

            # Add gray line at x=0 and y=0
            plt.axhline(0, color='gray', alpha=0.3)
            plt.axvline(0, color='gray', alpha=0.3)

            # Add labels and legend
            plt.xlabel(peer_group)
            plt.ylabel(company_name_i)
            plt.title(f"Decoupling {metric} vs {peer_group} {company_name_i}")
            legend_text = f'Regression\nBeta: {slope:.2f}, Alpha: {intercept:.2f}'
            plt.legend([legend_text])
            plt.savefig(f"Decoupling_{company_name_i}_{metric}_{peer_group}.png")
            plt.show()

            print(f"{metric} Alpha:", intercept)
            print(f"{metric} Beta:", slope)

            return intercept, slope
        else:
            print(f"Skipping regression for {company_name_i} due to insufficient data.")
            return np.nan, np.nan

    # Compute alpha & beta comparing Growth/Return to Sector
    alpha_growth_sector, beta_growth_sector = company_decoupling(company_name_i, company_i_growth, sector_growth_avg, sector_i, "Growth")
    alpha_return_sector, beta_return_sector = company_decoupling(company_name_i, company_i_return, sector_return_avg, sector_i, "Return")

except Exception as e:
    print(f"Issue with company {company_ticker}: {e}")