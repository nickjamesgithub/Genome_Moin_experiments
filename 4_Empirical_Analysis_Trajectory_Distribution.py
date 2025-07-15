import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import math
import glob
import os
import matplotlib
matplotlib.use('TkAgg')
# Import specific utilities (these imports assume the presence of `Utilities`)
from Utilities import compute_percentiles, firefly_plot, geometric_return
from Utilities import generate_market_cap_class, waterfall_value_plot, geometric_return, dendrogram_plot, get_even_clusters

# Global parameters
make_plots = False
genome_filtering = True
# Initialise years
beginning_year = 2011
end_year = 2024
# Generate grid of years
year_grid = np.linspace(beginning_year, end_year, end_year - beginning_year + 1)
rolling_window = 3

# Import data
data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Genome_code_250605\Genome-pipeline-code\Genome-pipeline-code\global_platform_data\Global_data.csv")

# Define countries and sectors to include
# Define countries and sectors to include
countries_to_include = [
    "Australia",
    "Belgium",
    "Canada",
    "Chile",
    "China",
    "Denmark",
    "France",
    "Germany",
    "Hong_Kong",
    "India",
    "Italy",
    "Japan",
    "Luxembourg",
    "Malaysia",
    "Netherlands",
    "Philippines",
    "Saudi_Arabia",
    "Singapore",
    "South_Korea",
    "Switzerland",
    "Sweden",
    "Thailand",
    "UAE",
    "USA",
    "United_Kingdom"
]

sectors_to_include =  [
    "Banking",
    "Communication Services",
    "Consumer Discretionary",
    "Consumer Staples",
    "Diversified",
    "Energy",
    "Financials",
    "Health Care",
    "Healthcare",
    "Industrials",
    "Insurance",
    "Investment and Wealth",
    "Materials",
    "Real Estate",
    "Technology",
    "Telecommunication",
    "Transportation",
    "Utilities"
]

# Filter data based on countries and sectors
filtered_data = data.loc[(data['Country'].isin(countries_to_include)) & (data['Sector'].isin(sectors_to_include))]

# Define functions
def sector_functions_mobility_matrix(df):
    unique_companies = df["Company_name"].unique()
    features_list = []
    for i in range(len(unique_companies)):
        # Slice for each company within sector slice
        slice_i = df.loc[df["Company_name"] == unique_companies[i]]
        # Various averages and calculations for the sector mobility matrix
        revenue_avg = slice_i["Revenue"].iloc[1:].mean()
        if len(slice_i) > 1 and slice_i["Revenue"].iloc[0] != 0:
            revenue_cagr = (slice_i["Revenue"].iloc[-1] / slice_i["Revenue"].iloc[0]) ** (1 / (len(slice_i) - 1)) - 1
        else:
            revenue_cagr = np.nan
        if len(slice_i) > 1:
            tsr = (slice_i["Adjusted_Stock_Price"].iloc[-1] / slice_i["Adjusted_Stock_Price"].iloc[0]) ** (1 / (len(slice_i) - 1)) - 1
        else:
            tsr = np.nan
        leverage = slice_i["Debt_to_equity"].iloc[1:].mean()
        investment = slice_i["RD/Revenue"].iloc[1:].mean()
        eva_ratio_avg = slice_i["EVA_ratio_bespoke"].iloc[1:].mean()
        acquisition_propensity = abs(slice_i["Cash_acquisitions"].iloc[1:]).sum() / slice_i["Market_Capitalisation"].iloc[1:].mean()
        capex_per_revenue_avg = slice_i["CAPEX/Revenue"].iloc[1:].mean()
        npat_per_employee_avg = slice_i["NPAT_per_employee"].iloc[1:].mean()
        gross_margin_avg = slice_i["Gross_margin"].iloc[1:].mean()
        features_list.append([revenue_avg, revenue_cagr, leverage, investment, eva_ratio_avg, acquisition_propensity, capex_per_revenue_avg, npat_per_employee_avg, gross_margin_avg, tsr])

    features_df = pd.DataFrame(features_list)
    features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    mean_values = features_df.mean(skipna=True)
    return mean_values

def company_functions_mobility_matrix(df):
    revenue_avg = df["Revenue"].iloc[1:].mean()
    revenue_cagr = (df["Revenue"].iloc[-1] / df["Revenue"].iloc[0]) ** (1 / (len(df) - 1)) - 1
    tsr = (df["Adjusted_Stock_Price"].iloc[-1] / df["Adjusted_Stock_Price"].iloc[0]) ** (1 / (len(df) - 1)) - 1
    leverage = df["Debt_to_equity"].iloc[1:].mean()
    investment = df["RD/Revenue"].iloc[1:].mean()
    eva_ratio_avg = df["EVA_ratio_bespoke"].iloc[1:].mean()
    acquisition_propensity = abs(df["Cash_acquisitions"].iloc[1:]).sum() / df["Market_Capitalisation"].iloc[1:].mean()
    capex_per_revenue_avg = df["CAPEX/Revenue"].iloc[1:].mean()
    npat_per_employee_avg = df["NPAT_per_employee"].iloc[1:].mean()
    gross_margin_avg = df["Gross_margin"].iloc[1:].mean()

    return revenue_avg, revenue_cagr, leverage, investment, eva_ratio_avg, acquisition_propensity, capex_per_revenue_avg, npat_per_employee_avg, gross_margin_avg, tsr

# Get unique tickers
unique_tickers = filtered_data["Ticker_full"].unique()
# Choose sectors to include
sector = filtered_data["Sector"].values
# Merge dataframes
# Create feature for Price-to-book
filtered_data["Price_to_Book"] = filtered_data["Market_Capitalisation"] / filtered_data["Book_Value_Equity"]
# Features to slice
features = ["Company_name", "Country", "Sector", "Year", "TSR", "Revenue_growth_3_f", "Stock_Price", "Adjusted_Stock_Price",
            "DPS", "BBPS", "DBBPS", "Genome_classification_bespoke", "Price_to_Book", "PE_Implied", "Market_Capitalisation",
            "Revenue", "Debt_to_equity", "RD/Revenue", "EVA_ratio_bespoke", "ROE_above_Cost_of_equity", "CROTE_TE",
            "Cash_acquisitions", "NPAT_per_employee", "CAPEX/Revenue", "Gross_margin"]

# Loop over all years
df_list = []
df_issue_list = []
for i in range(len(year_grid) - rolling_window):
    # Loop over all companies
    for j in range(len(unique_tickers)):
        try:
            # Get years... think of indexing as Years 0, 1 and 2
            year_i = year_grid[i]
            year_i_2 = year_grid[i + rolling_window]

            # Slice dataframes based on company and year
            df_slice_i = filtered_data.loc[(filtered_data["Year"] == year_i) & (filtered_data["Ticker_full"] == unique_tickers[j])][features]
            df_slice_i_2 = filtered_data.loc[(filtered_data["Year"] == year_i_2) & (filtered_data["Ticker_full"] == unique_tickers[j])][features]

            # Slice company name
            company_info = filtered_data.loc[
                (filtered_data["Year"] == year_i_2) &
                (filtered_data["Ticker_full"] == unique_tickers[j])
            ]

            if company_info.empty:
                print(f"Issue: No data for ticker {unique_tickers[j]} in year {year_i_2}")
                df_issue_list.append([unique_tickers[j], year_i, year_i_2])
                continue

            company_name_j = company_info["Company_name"].values[0]

            unique_ticker_j = unique_tickers[j]
            sector_j = df_slice_i["Sector"].values[0]
            country_j = df_slice_i["Country"].values[0]

            # Slice dataframe based on entire date range for sector and company
            df_slice_company_range_i = filtered_data.loc[(filtered_data["Year"] >= year_i) & (filtered_data["Year"] <= year_i_2) & (filtered_data["Ticker_full"] == unique_tickers[j])][features]
            df_slice_sector_range_i = filtered_data.loc[(filtered_data["Year"] >= year_i) & (filtered_data["Year"] <= year_i_2) & (filtered_data["Sector"] == sector_j)][features]

            # Compute sector mobility matrix
            sector_revenue_avg, sector_revenue_cagr, sector_leverage, sector_investment, sector_eva_avg, \
            sector_acquisition_propensity, sector_capex_per_revenue, sector_npat_per_employee, sector_gross_margin, sector_tsr = sector_functions_mobility_matrix(df_slice_sector_range_i)

            # Compute Company mobility matrix
            company_revenue_avg, company_revenue_cagr, company_leverage, company_investment, company_eva_avg,  \
            company_acquisition_propensity, company_capex_per_revenue, company_npat_per_employee, company_gross_margin, company_tsr = company_functions_mobility_matrix(df_slice_company_range_i)

            # Compute mobility matrix delta
            delta_revenue_avg = company_revenue_avg - sector_revenue_avg
            delta_revenue_cagr = company_revenue_cagr - sector_revenue_cagr
            delta_leverage = company_leverage - sector_leverage
            delta_investment = company_investment - sector_investment
            delta_eva_avg = company_eva_avg - sector_eva_avg
            delta_acquisition_propensity = company_acquisition_propensity - sector_acquisition_propensity
            delta_capex_per_revenue = company_capex_per_revenue - sector_capex_per_revenue
            delta_npat_per_employee = company_npat_per_employee - sector_npat_per_employee
            delta_gross_margin = company_gross_margin - sector_gross_margin

            # Get Genome classification (standard) and bespoke
            genome_classification_bespoke_beginning = df_slice_i["Genome_classification_bespoke"].values[0]
            genome_classification_bespoke_end = df_slice_i_2["Genome_classification_bespoke"].values[0]

            # Get the P/E
            pe_implied_beginning = df_slice_i["PE_Implied"].values[0]
            pe_implied_end = df_slice_i_2["PE_Implied"].values[0]

            # Get Market Capitalisation
            market_capitalisation_beginning = df_slice_i["Market_Capitalisation"].values[0]
            market_capitalisation_end = df_slice_i_2["Market_Capitalisation"].values[0]

            # Get coordinates of firefly with X & Y
            firefly_y_beginning = df_slice_i["EVA_ratio_bespoke"].values[0]
            firefly_y_end = df_slice_i_2["EVA_ratio_bespoke"].values[0]
            firefly_x_beginning = df_slice_i["Revenue_growth_3_f"].values[0]
            firefly_x_end = df_slice_i_2["Revenue_growth_3_f"].values[0]

            # Compute angle between X & Y - first in radians and then in degrees
            radians = math.atan2(firefly_y_end - firefly_y_beginning,
                                 firefly_x_end - firefly_x_beginning)
            degree_angle = math.degrees(radians)
            degrees = (degree_angle + 360) % 360

            # Get all Dividends and buybacks in that range
            dbbps_slice = filtered_data.loc[(filtered_data["Year"] >= year_i) &
                                            (filtered_data["Year"] <= year_i_2) &
                                            (filtered_data["Ticker_full"] == unique_tickers[j])]["DBBPS"]
            dbbps_total = dbbps_slice.values.sum()

            ### Scenario 1 - Remain Negative ###
            if df_slice_i["EVA_ratio_bespoke"].values[0] < 0 and df_slice_i_2["EVA_ratio_bespoke"].values[0] < 0:
                print("Company " + unique_ticker_j + " Remain negative scenario " + "Years " + str(year_i) + "-" + str(year_i_2))

                cumulative_tsr = (df_slice_i_2["Stock_Price"].values[0] - df_slice_i["Stock_Price"].values[0] + dbbps_total) / df_slice_i["Stock_Price"].values[0]
                annualized_tsr = (1 + cumulative_tsr) ** (1 / rolling_window) - 1

                cumulative_tsr_capiq = df_slice_i_2["Adjusted_Stock_Price"].values[0] / df_slice_i["Adjusted_Stock_Price"].values[0] - 1
                annualized_tsr_capiq = (1 + cumulative_tsr_capiq) ** (1 / rolling_window) - 1

                df_list.append(["Remain_negative", genome_classification_bespoke_beginning, genome_classification_bespoke_end, company_name_j, unique_ticker_j, country_j, sector_j, int(year_i), int(year_i_2), df_slice_i["Revenue_growth_3_f"].values[0], df_slice_i_2["Revenue_growth_3_f"].values[0],
                                df_slice_i["EVA_ratio_bespoke"].values[0], df_slice_i_2["EVA_ratio_bespoke"].values[0],
                                df_slice_i["ROE_above_Cost_of_equity"].values[0], df_slice_i_2["ROE_above_Cost_of_equity"].values[0],
                                df_slice_i["CROTE_TE"].values[0], df_slice_i_2["CROTE_TE"].values[0],
                                df_slice_i["Stock_Price"].values[0], df_slice_i_2["Stock_Price"].values[0], df_slice_i["Price_to_Book"].values[0], dbbps_total, degrees, radians,
                                annualized_tsr, annualized_tsr_capiq, pe_implied_beginning, pe_implied_end, market_capitalisation_beginning, market_capitalisation_end,
                                company_revenue_avg, sector_revenue_avg, delta_revenue_avg,
                                company_revenue_cagr, sector_revenue_cagr, delta_revenue_cagr,
                                company_leverage, sector_leverage, delta_leverage,
                                company_investment, sector_investment, delta_investment,
                                company_eva_avg, sector_eva_avg, delta_eva_avg,
                                company_acquisition_propensity, sector_acquisition_propensity, delta_acquisition_propensity,
                                company_capex_per_revenue, sector_capex_per_revenue, delta_capex_per_revenue,
                                company_npat_per_employee, sector_npat_per_employee, delta_npat_per_employee,
                                company_gross_margin, sector_gross_margin, delta_gross_margin, sector_tsr, company_tsr])

            ### Scenario 2 - Move Up ###
            if df_slice_i["EVA_ratio_bespoke"].values[0] < 0 and df_slice_i_2["EVA_ratio_bespoke"].values[0] >= 0:
                print("Company " + unique_ticker_j + " Move up scenario " + "Years " + str(year_i) + "-" + str(year_i_2))

                cumulative_tsr = (df_slice_i_2["Stock_Price"].values[0] - df_slice_i["Stock_Price"].values[0] + dbbps_total) / df_slice_i["Stock_Price"].values[0]
                annualized_tsr = (1 + cumulative_tsr) ** (1 / rolling_window) - 1

                cumulative_tsr_capiq = df_slice_i_2["Adjusted_Stock_Price"].values[0] / df_slice_i["Adjusted_Stock_Price"].values[0] - 1
                annualized_tsr_capiq = (1 + cumulative_tsr_capiq) ** (1 / rolling_window) - 1

                df_list.append(["Move_up", genome_classification_bespoke_beginning, genome_classification_bespoke_end, company_name_j, unique_ticker_j, country_j, sector_j, int(year_i), int(year_i_2), df_slice_i["Revenue_growth_3_f"].values[0], df_slice_i_2["Revenue_growth_3_f"].values[0],
                                df_slice_i["EVA_ratio_bespoke"].values[0], df_slice_i_2["EVA_ratio_bespoke"].values[0],
                                df_slice_i["ROE_above_Cost_of_equity"].values[0], df_slice_i_2["ROE_above_Cost_of_equity"].values[0],
                                df_slice_i["CROTE_TE"].values[0], df_slice_i_2["CROTE_TE"].values[0],
                                df_slice_i["Stock_Price"].values[0], df_slice_i_2["Stock_Price"].values[0], df_slice_i["Price_to_Book"].values[0], dbbps_total, degrees, radians,
                                annualized_tsr, annualized_tsr_capiq, pe_implied_beginning, pe_implied_end, market_capitalisation_beginning, market_capitalisation_end,
                                company_revenue_avg, sector_revenue_avg, delta_revenue_avg,
                                company_revenue_cagr, sector_revenue_cagr, delta_revenue_cagr,
                                company_leverage, sector_leverage, delta_leverage,
                                company_investment, sector_investment, delta_investment,
                                company_eva_avg, sector_eva_avg, delta_eva_avg,
                                company_acquisition_propensity, sector_acquisition_propensity, delta_acquisition_propensity,
                                company_capex_per_revenue, sector_capex_per_revenue, delta_capex_per_revenue,
                                company_npat_per_employee, sector_npat_per_employee, delta_npat_per_employee,
                                company_gross_margin, sector_gross_margin, delta_gross_margin, sector_tsr, company_tsr])

            ### Scenario 3 - Remain Positive ###
            if df_slice_i["EVA_ratio_bespoke"].values[0] >= 0 and df_slice_i_2["EVA_ratio_bespoke"].values[0] >= 0:
                print("Company " + unique_ticker_j + " Remain positive scenario " + "Years " + str(year_i) + "-" + str(year_i_2))

                cumulative_tsr = (df_slice_i_2["Stock_Price"].values[0] - df_slice_i["Stock_Price"].values[0] + dbbps_total) / df_slice_i["Stock_Price"].values[0]
                annualized_tsr = (1 + cumulative_tsr) ** (1 / rolling_window) - 1

                cumulative_tsr_capiq = df_slice_i_2["Adjusted_Stock_Price"].values[0] / df_slice_i["Adjusted_Stock_Price"].values[0] - 1
                annualized_tsr_capiq = (1 + cumulative_tsr_capiq) ** (1 / rolling_window) - 1

                df_list.append(["Remain_positive", genome_classification_bespoke_beginning, genome_classification_bespoke_end, company_name_j, unique_ticker_j, country_j, sector_j, int(year_i), int(year_i_2), df_slice_i["Revenue_growth_3_f"].values[0], df_slice_i_2["Revenue_growth_3_f"].values[0],
                                df_slice_i["EVA_ratio_bespoke"].values[0], df_slice_i_2["EVA_ratio_bespoke"].values[0],
                                df_slice_i["ROE_above_Cost_of_equity"].values[0], df_slice_i_2["ROE_above_Cost_of_equity"].values[0],
                                df_slice_i["CROTE_TE"].values[0], df_slice_i_2["CROTE_TE"].values[0],
                                df_slice_i["Stock_Price"].values[0], df_slice_i_2["Stock_Price"].values[0], df_slice_i["Price_to_Book"].values[0], dbbps_total, degrees, radians,
                                annualized_tsr, annualized_tsr_capiq, pe_implied_beginning, pe_implied_end, market_capitalisation_beginning, market_capitalisation_end,
                                company_revenue_avg, sector_revenue_avg, delta_revenue_avg,
                                company_revenue_cagr, sector_revenue_cagr, delta_revenue_cagr,
                                company_leverage, sector_leverage, delta_leverage,
                                company_investment, sector_investment, delta_investment,
                                company_eva_avg, sector_eva_avg, delta_eva_avg,
                                company_acquisition_propensity, sector_acquisition_propensity, delta_acquisition_propensity,
                                company_capex_per_revenue, sector_capex_per_revenue, delta_capex_per_revenue,
                                company_npat_per_employee, sector_npat_per_employee, delta_npat_per_employee,
                                company_gross_margin, sector_gross_margin, delta_gross_margin, sector_tsr, company_tsr])

            ### Scenario 4 - Move Down ###
            if df_slice_i["EVA_ratio_bespoke"].values[0] > 0 and df_slice_i_2["EVA_ratio_bespoke"].values[0] < 0:
                print("Company " + unique_ticker_j + " Move down scenario " + "Years " + str(year_i) + "-" + str(year_i_2))

                cumulative_tsr = (df_slice_i_2["Stock_Price"].values[0] - df_slice_i["Stock_Price"].values[0] + dbbps_total) / df_slice_i["Stock_Price"].values[0]
                annualized_tsr = (1 + cumulative_tsr) ** (1 / rolling_window) - 1

                cumulative_tsr_capiq = df_slice_i_2["Adjusted_Stock_Price"].values[0] / df_slice_i["Adjusted_Stock_Price"].values[0] - 1
                annualized_tsr_capiq = (1 + cumulative_tsr_capiq) ** (1 / rolling_window) - 1

                df_list.append(["Move_down", genome_classification_bespoke_beginning, genome_classification_bespoke_end, company_name_j, unique_ticker_j, country_j, sector_j, int(year_i), int(year_i_2), df_slice_i["Revenue_growth_3_f"].values[0], df_slice_i_2["Revenue_growth_3_f"].values[0],
                                df_slice_i["EVA_ratio_bespoke"].values[0], df_slice_i_2["EVA_ratio_bespoke"].values[0],
                                df_slice_i["ROE_above_Cost_of_equity"].values[0], df_slice_i_2["ROE_above_Cost_of_equity"].values[0],
                                df_slice_i["CROTE_TE"].values[0], df_slice_i_2["CROTE_TE"].values[0],
                                df_slice_i["Stock_Price"].values[0], df_slice_i_2["Stock_Price"].values[0], df_slice_i["Price_to_Book"].values[0], dbbps_total, degrees, radians,
                                annualized_tsr, annualized_tsr_capiq, pe_implied_beginning, pe_implied_end, market_capitalisation_beginning, market_capitalisation_end,
                                company_revenue_avg, sector_revenue_avg, delta_revenue_avg,
                                company_revenue_cagr, sector_revenue_cagr, delta_revenue_cagr,
                                company_leverage, sector_leverage, delta_leverage,
                                company_investment, sector_investment, delta_investment,
                                company_eva_avg, sector_eva_avg, delta_eva_avg,
                                company_acquisition_propensity, sector_acquisition_propensity, delta_acquisition_propensity,
                                company_capex_per_revenue, sector_capex_per_revenue, delta_capex_per_revenue,
                                company_npat_per_employee, sector_npat_per_employee, delta_npat_per_employee,
                                company_gross_margin, sector_gross_margin, delta_gross_margin, sector_tsr, company_tsr])
        except:
            print(f"Issue during processing: Ticker {unique_tickers[j]} Years: {year_i}-{year_i_2}")

            df_issue_list.append([company_name_j, year_i, year_i_2])

# Create collapsed journeys dataframe
df_journey_collapsed = pd.DataFrame(df_list)

# Create columns
df_journey_collapsed.columns = ["Journey", "Genome_classification_bespoke_beginning", "Genome_classification_bespoke_end",
                                "Company_name", "Ticker", "Country", "Sector", "Year_beginning", "Year_final",
                                "Revenue_growth_beginning", "Revenue_growth_end", "EVA_beginning", "EVA_end",
                                "ROE_above_Cost_of_equity_beginning", "ROE_above_Cost_of_equity_end",
                                "CROTE_TE_beginning", "CROTE_TE_end",
                                "Stock_price_beginning", "Stock_price_final", "Price_to_book", "DBBPS_total", "Angle", "Radians",
                                "Annualized_TSR", "Annualized_TSR_Capiq", "PE_beginning", "PE_end", "Market_Capitalisation_beginning", "Market_Capitalisation_end",
                                "Company_revenue_avg", "Sector_revenue_avg", "Delta_revenue_avg",
                                "Company_revenue_cagr", "Sector_revenue_cagr", "Delta_revenue_cagr",
                                "Company_leverage", "Sector_leverage", "Delta_leverage",
                                "Company_investment", "Sector_investment", "Delta_investment",
                                "Company_eva_avg", "Sector_eva_avg", "delta_eva_avg",
                                "Company_acquisition_propensity", "Sector_acquisition_propensity", "delta_acquisition_propensity",
                                "Company_capex/revenue", "Sector_capex/revenue", "Delta_capex/revenue",
                                "Company_npat_per_employee", "Sector_npat_per_employee", "Delta_npat_per_employee",
                                "Company_gross_margin", "Sector_gross_margin", "Delta_gross_margin", "Sector_TSR", "Company_TSR"]

# Replace all inf values with nan
df_journey_collapsed = df_journey_collapsed.replace([np.inf, -np.inf], np.nan)

# Remove rows with NaN values in 'Annualized_TSR'
df_journey_collapsed = df_journey_collapsed.dropna(subset=['Annualized_TSR_Capiq'], axis=0)

# Add change in X axis and Y axis
df_journey_collapsed["X_change"] = df_journey_collapsed["Revenue_growth_end"] - df_journey_collapsed["Revenue_growth_beginning"]
df_journey_collapsed["Y_change"] = df_journey_collapsed["EVA_end"] - df_journey_collapsed["EVA_beginning"]

# Write out csv file locally
df_journey_collapsed.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Journeys\Journeys_summary_Global_FE_Update.csv")

if genome_filtering:
    df_journey_collapsed = df_journey_collapsed.loc[(df_journey_collapsed["EVA_end"] >= -0.3) & (df_journey_collapsed["EVA_end"] <= 0.5) &
                                     (df_journey_collapsed["Revenue_growth_end"] >= -0.3) & (df_journey_collapsed["Revenue_growth_end"] <= 1.5) &
                                     (df_journey_collapsed["Annualized_TSR_Capiq"] >= -0.4) & (df_journey_collapsed["Annualized_TSR_Capiq"] <= 1) &
                                     (df_journey_collapsed["Price_to_book"] > -200)]

# Write out csv file locally
df_journey_collapsed.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Journeys\Journeys_summary_Global_FE_Update_filtered.csv")

# Filter out infinite and NaN values before plotting
df_journey_collapsed = df_journey_collapsed.replace([np.inf, -np.inf], np.nan)

# Drop rows with NaN values in 'Annualized_TSR' column
df_journey_collapsed = df_journey_collapsed.dropna(subset=['Annualized_TSR_Capiq'], axis=0)

# Ensure that 'Angle' and 'Annualized_TSR' columns contain only numeric values
df_journey_collapsed['Angle'] = pd.to_numeric(df_journey_collapsed['Angle'], errors='coerce')
df_journey_collapsed['Annualized_TSR_Capiq'] = pd.to_numeric(df_journey_collapsed['Annualized_TSR_Capiq'], errors='coerce')

if make_plots:
    # Loop over unique values in "Journey" column
    for journey in df_journey_collapsed["Journey"].unique():
        # Filter dataframe for the current journey
        df_filtered = df_journey_collapsed[df_journey_collapsed["Journey"] == journey]
        df_filtered = df_filtered.replace([np.inf, -np.inf], np.nan).dropna(subset=['Angle', 'Annualized_TSR_Capiq'], how='any')
        plt.figure(figsize=(8, 6))
        plt.scatter(df_filtered["Angle"], df_filtered["Annualized_TSR_Capiq"], alpha=0.5)
        kde = gaussian_kde(df_filtered[["Angle", "Annualized_TSR_Capiq"]].T, bw_method=0.3)
        xi, yi = np.mgrid[df_filtered["Angle"].min():df_filtered["Angle"].max():100j,
                          df_filtered["Annualized_TSR_Capiq"].min():df_filtered["Annualized_TSR_Capiq"].max():100j]
        zi = kde(np.vstack([xi.flatten(), yi.flatten()]))
        plt.contour(xi, yi, zi.reshape(xi.shape), colors='k')
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Annualized TSR")
        plt.title("Annualized TSR vs Angle for Journey: " + journey)
        plt.grid(True)
        plt.savefig("Distribution_joint_angles_Journey_" + journey)
        plt.show()

if make_plots:
    plt.figure(figsize=(10, 8))
    journey_colors = {}
    journey_medians = {}
    for journey in df_journey_collapsed["Journey"].unique():
        df_filtered = df_journey_collapsed[df_journey_collapsed["Journey"] == journey]
        df_filtered = df_filtered.replace([np.inf, -np.inf], np.nan).dropna(subset=['Annualized_TSR_Capiq'], how='any')
        tsr_vector = df_filtered["Annualized_TSR_Capiq"] * 100
        kde = gaussian_kde(tsr_vector, bw_method=0.3)
        x = np.linspace(tsr_vector.min(), tsr_vector.max(), 100)
        y = kde(x)
        plt.plot(x, y, label=journey)
        line_color = plt.gca().lines[-1].get_color()
        journey_colors[journey] = line_color
        median_tsr = np.median(tsr_vector)
        journey_medians[journey] = median_tsr

    plt.xlabel("Annualized TSR %")
    plt.ylabel("Density")
    plt.title("Distribution of Annualized TSR by Journey")
    plt.legend()
    for journey, color in journey_colors.items():
        median_tsr = journey_medians[journey]
        plt.axvline(median_tsr, linestyle='--', color=color, alpha=0.5, label=f"{journey} - Median: {median_tsr:.2f}%")
    plt.grid(True)
    plt.legend()
    plt.savefig("Distribution_Annualized_TSR_ByJourney")
    plt.show()
