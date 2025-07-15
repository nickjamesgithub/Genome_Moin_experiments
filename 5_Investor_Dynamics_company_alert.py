import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import savgol_filter

matplotlib.use('TkAgg')

# Global parameters
window = 250

# Define a dictionary to map specific sectors to the y-axis metric, with others defaulting to "EP/FE"
sector_metric_mapping = {
    "Banking": "CROTE_TE",
    "Investment and Wealth": "ROE_above_Cost_of_equity",
    "Insurance": "ROE_above_Cost_of_equity",
    "Financials - other": "ROE_above_Cost_of_equity",
    # Other sectors will use "EP/FE" by default
}

# Function to process genome classification
def generate_bespoke_genome_classification_df(df):
    for sector, metric in sector_metric_mapping.items():
        if metric not in df.columns:
            raise ValueError(f"Missing required metric '{metric}' in DataFrame for sector '{sector}'.")
    classified_dfs = []
    for sector in df["Sector"].unique():
        metric = sector_metric_mapping.get(sector, "EP/FE")
        if metric not in df.columns:
            continue
        sector_df = df[df["Sector"] == sector].copy()
        conditions_genome = [
            (sector_df[metric] < 0) & (sector_df["Revenue_growth_3_f"] < 0),
            (sector_df[metric] < 0) & (sector_df["Revenue_growth_3_f"].between(0, 0.10, inclusive='right')),
            (sector_df[metric] < 0) & (sector_df["Revenue_growth_3_f"].between(0.10, 0.20, inclusive='right')),
            (sector_df[metric] < 0) & (sector_df["Revenue_growth_3_f"] >= 0.20),
            (sector_df[metric] > 0) & (sector_df["Revenue_growth_3_f"] < 0),
            (sector_df[metric] > 0) & (sector_df["Revenue_growth_3_f"].between(0, 0.10, inclusive='right')),
            (sector_df[metric] > 0) & (sector_df["Revenue_growth_3_f"].between(0.10, 0.20, inclusive='right')),
            (sector_df[metric] > 0) & (sector_df["Revenue_growth_3_f"] >= 0.20)
        ]
        values_genome = ["UNTENABLE", "TRAPPED", "BRAVE", "FEARLESS", "CHALLENGED", "VIRTUOUS", "FAMOUS", "LEGENDARY"]
        sector_df["Genome_classification_bespoke"] = np.select(conditions_genome, values_genome, default="UNKNOWN")
        classified_dfs.append(sector_df)
    return pd.concat(classified_dfs)

# Define autocorrelation function and financial metrics functions
def autocorrelation_function(returns, window):
    up_days = len([x for x in returns[-window:] if x > 0])
    down_days = len([x for x in returns[-window:] if x <= 0])
    return down_days, up_days

def downside_risk(returns, risk_free=0):
    adj_returns = returns - risk_free
    sqr_downside = np.square(np.clip(adj_returns, np.NINF, 0))
    return np.sqrt(np.nanmean(sqr_downside) * 252)

def sortino(returns, risk_free=0):
    adj_returns = returns - risk_free
    drisk = downside_risk(adj_returns)
    if drisk == 0:
        return np.nan
    return (np.nanmean(adj_returns) * np.sqrt(252)) / drisk

def compute_sharpe_ratio(log_returns, risk_free_rate=0.01):
    mean_log_return_annual = np.nanmean(log_returns) * 252
    std_dev_log_return_annual = np.nanstd(log_returns) * np.sqrt(252)
    if std_dev_log_return_annual > 0:
        sharpe_ratio = (mean_log_return_annual - risk_free_rate) / std_dev_log_return_annual
    else:
        sharpe_ratio = np.nan
    return sharpe_ratio

# Import data
data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Genome_code_250605\Genome-pipeline-code\Genome-pipeline-code\global_platform_data\Global_data.csv")

# Define countries and sectors to include
# countries_to_include = ['USA', 'AUS', 'INDIA', 'JAPAN', 'EURO', 'UK']
# sectors_to_include = data["Sector"].unique()

# Filter data based on countries and sectors
filtered_data = data # data.loc[(data['Country'].isin(countries_to_include)) & (data['Sector'].isin(sectors_to_include))]
tickers_ = np.unique(filtered_data["Ticker"].values)

# Store features
features_list = []
for i in range(len(tickers_)):
    company_i = tickers_[i]
    try:
        country_i = filtered_data.loc[filtered_data["Ticker"] == company_i, "Country"].values[0]
        sector_i = filtered_data.loc[filtered_data["Ticker"] == company_i, "Sector"].values[0]
        name_i = filtered_data.loc[filtered_data["Ticker"] == company_i, "Company_name"].values[0]
        df_path = fr"\Users\60848\OneDrive - Bain\Desktop\Genome_code_250605\Genome-pipeline-code\Genome-pipeline-code\global_platform_data\{country_i}\_{company_i}.csv"
        price_path = fr"\Users\60848\OneDrive - Bain\Desktop\Genome_code_250605\Genome-pipeline-code\Genome-pipeline-code\global_platform_data\share_price\{country_i}\_{company_i}_price.csv"
        print("Iteration ", name_i)

        # Read the data files
        df = pd.read_csv(df_path)
        price_df = pd.read_csv(price_path)

        df_g = generate_bespoke_genome_classification_df(df)

        genome_class_i_current = df_g.loc[df_g['Year'] == 2024, 'Genome_classification_bespoke'].values[0] if not df_g.loc[df_g['Year'] == 2024, 'Genome_classification_bespoke'].empty else 'N/A'
        genome_class_i_trailing = df_g.loc[df_g['Year'] == 2023, 'Genome_classification_bespoke'].values[0] if not df_g.loc[df_g['Year'] == 2023, 'Genome_classification_bespoke'].empty else 'N/A'
        genome_class_i_trailing_2 = df_g.loc[df_g['Year'] == 2022, 'Genome_classification_bespoke'].values[0] if not df_g.loc[df_g['Year'] == 2022, 'Genome_classification_bespoke'].empty else 'N/A'

        # Compute price information
        price = price_df["Price"].iloc[-window:]
        log_returns = np.log(price) - np.log(price.shift(1))

        # Apply Savitzky-Golay filter for smoothing
        if len(price) >= 31:  # Check if the data has enough points to apply the filter
            smoothed_prices = savgol_filter(price, window_length=31, polyorder=2)
            # Assuming 'smoothed_prices' has already been calculated using the Savitzky-Golay filter
            # Function to compute velocity as the rate of change over a specified number of days
            def compute_velocity(prices, days):
                if len(prices) >= days:
                    return (prices[-1] - prices[-days]) / prices[-days]
                else:
                    return np.nan  # Return NaN if there aren't enough days of data

            # Compute velocities
            velocity_10 = compute_velocity(smoothed_prices, 10)
            velocity_30 = compute_velocity(smoothed_prices, 30)
            velocity_60 = compute_velocity(smoothed_prices, 60)

        # Compute financial metrics
        downs, ups = autocorrelation_function(log_returns, window)
        down_ratio = np.nan_to_num(downs / ups)
        total_returns_window = np.nan_to_num(np.sum(log_returns))
        volatility_window = np.nanstd(log_returns) * np.sqrt(252)
        sharpe = compute_sharpe_ratio(log_returns.dropna(), risk_free_rate=0.0)
        sortino_r = sortino(log_returns, 0)

        # Append features to feature list
        features_list.append([name_i, country_i, sector_i, genome_class_i_current, genome_class_i_trailing, genome_class_i_trailing_2,
                              total_returns_window, volatility_window, down_ratio, sharpe, sortino_r, velocity_10, velocity_30, velocity_60])

    except Exception as e:
        print(f"Error with company {company_i}: {e}")

# Generate features DataFrame
features_df = pd.DataFrame(features_list)
features_df.columns = ["Company_name", "Country", "Sector", "Genome_class_2024", "Genome_class_2023", "Genome_class_2022", "Total_returns", "Volatility", "Down_ratio", "Sharpe", "Sortino", "Velocity_10", "Velocity_30", "Velocity_60"]
features_df.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Genome_code_250605\Genome-pipeline-code\Genome-pipeline-code\global_platform_data\Global_dynamics_alert_flag_250606.csv")
