import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

# If you need an interactive backend:
matplotlib.use('TkAgg')

# Global error list for logging files with errors
error_files = []

# -----------------------------------------------------------
# 1. Sector-Metric Mapping
# -----------------------------------------------------------
sector_metric_mapping = {
    "Banking": "CROTE_TE",
    "Investment and Wealth": "ROE_above_Cost_of_equity",
    "Insurance": "ROE_above_Cost_of_equity",
    "Financials - other": "ROE_above_Cost_of_equity",
    # Other sectors will default to "EVA_ratio_bespoke"
}

# -----------------------------------------------------------
# 2. Bespoke Genome Classification Function
# -----------------------------------------------------------
def generate_bespoke_genome_classification_df(df):
    for sector, metric in sector_metric_mapping.items():
        if sector in df["Sector"].unique() and metric not in df.columns:
            raise ValueError(f"Missing required metric '{metric}' in DataFrame for sector '{sector}'.")

    classified_dfs = []
    for sector in df["Sector"].unique():
        metric = sector_metric_mapping.get(sector, "EVA_ratio_bespoke")
        if metric not in df.columns:
            continue

        sector_df = df[df["Sector"] == sector].copy()

        conditions_genome = [
            (sector_df["EVA_ratio_bespoke"] < 0) & (sector_df["Revenue_growth_3_f"] < 0),
            (sector_df["EVA_ratio_bespoke"] < 0) & (sector_df["Revenue_growth_3_f"].between(0, 0.10, inclusive='right')),
            (sector_df["EVA_ratio_bespoke"] < 0) & (sector_df["Revenue_growth_3_f"].between(0.10, 0.20, inclusive='right')),
            (sector_df["EVA_ratio_bespoke"] < 0) & (sector_df["Revenue_growth_3_f"] >= 0.20),
            (sector_df["EVA_ratio_bespoke"] > 0) & (sector_df["Revenue_growth_3_f"] < 0),
            (sector_df["EVA_ratio_bespoke"] > 0) & (sector_df["Revenue_growth_3_f"].between(0, 0.10, inclusive='right')),
            (sector_df["EVA_ratio_bespoke"] > 0) & (sector_df["Revenue_growth_3_f"].between(0.10, 0.20, inclusive='right')),
            (sector_df["EVA_ratio_bespoke"] > 0) & (sector_df["Revenue_growth_3_f"] >= 0.20)
        ]
        values_genome = [
            "UNTENABLE", "TRAPPED", "BRAVE", "FEARLESS",
            "CHALLENGED", "VIRTUOUS", "FAMOUS", "LEGENDARY"
        ]

        sector_df["Genome_classification_bespoke"] = np.select(
            conditions_genome, values_genome, default="UNKNOWN"
        )
        classified_dfs.append(sector_df)

    return pd.concat(classified_dfs) if classified_dfs else df

# -----------------------------------------------------------
# 3. Directory Setup
# -----------------------------------------------------------
BASE_DIR = r"C:\Users\60848\OneDrive - Bain\Desktop\Genome_code_250605\Genome-pipeline-code\Genome-pipeline-code"
GLOBAL_MARKET_DIR = os.path.join(BASE_DIR, "Global_market_constituents")
GLOBAL_DATA_DIR = os.path.join(BASE_DIR, "global_platform_data")

# -----------------------------------------------------------
# 4. Helper Function to Load Country Data
# -----------------------------------------------------------
def df_country_creator(country, tickers):
    dfs_list = []
    for ticker in tickers:
        print(f"Processing {country}: {ticker}")
        file_path = os.path.join(GLOBAL_DATA_DIR, country, f"_{ticker}.csv")
        try:
            df = pd.read_csv(file_path, encoding='cp1252')
            df["Country"] = country
            df["File_Path"] = file_path
            df["Ticker"] = ticker
            dfs_list.append(df)
        except Exception as e:
            error_message = f"Error with {ticker} in {country}: {e}"
            print(error_message)
            error_files.append({
                "Country": country,
                "Ticker": ticker,
                "File_Path": file_path,
                "Error_Message": str(e)
            })

    return pd.concat(dfs_list, ignore_index=True) if dfs_list else pd.DataFrame()

# -----------------------------------------------------------
# 5. Main Routine to Dynamically Process All Countries
# -----------------------------------------------------------
def main():
    all_dataframes = []

    for file_name in os.listdir(GLOBAL_MARKET_DIR):
        if file_name.startswith("Company_list_GPT_") and file_name.endswith(".csv"):
            country = file_name.replace("Company_list_GPT_", "").replace(".csv", "")

            print(f"---\nLoading tickers for {country} from {file_name}")
            mapping_df = pd.read_csv(os.path.join(GLOBAL_MARKET_DIR, file_name), encoding='cp1252', dtype={'Ticker': str})

            if "Ticker" not in mapping_df.columns:
                print(f"Skipping {file_name}: No 'Ticker' column found.")
                continue

            tickers = mapping_df["Ticker"].unique()
            country_df = df_country_creator(country, tickers)

            if not country_df.empty:
                country_df_classified = generate_bespoke_genome_classification_df(country_df)
                all_dataframes.append(country_df_classified)
            else:
                print(f"No data loaded for {country}.")

    if all_dataframes:
        df_merge_global = pd.concat(all_dataframes, axis=0)

        df_merge_global["Sector"] = df_merge_global["Sector"].replace("Consumer staples", "Consumer Staples")

        def extract_ticker_row(row):
            ticker_full = row.get("Ticker_full", "")
            file_path = row.get("File_Path", "")

            if isinstance(ticker_full, str) and ":" in ticker_full:
                return ticker_full.split(":")[1]
            elif isinstance(ticker_full, str) and "." in ticker_full:
                return ticker_full.split(".")[0]
            elif isinstance(file_path, str):
                filename = os.path.basename(file_path)
                return filename.replace(".csv", "").lstrip("_")
            return None

        if "Ticker_full" in df_merge_global.columns or "File_Path" in df_merge_global.columns:
            df_merge_global["Ticker"] = df_merge_global.apply(extract_ticker_row, axis=1)

        if "File_Path" in df_merge_global.columns:
            df_merge_global.drop(columns=["File_Path"], inplace=True)

        output_path = os.path.join(GLOBAL_DATA_DIR, "Global_data.csv")
        df_merge_global.to_csv(output_path, index=False)
        print(f"\nGlobal data saved to: {output_path}")
    else:
        print("No country dataframes were created. Please check your file paths.")

    if error_files:
        error_df = pd.DataFrame(error_files)
        error_output_path = os.path.join(GLOBAL_DATA_DIR, "Error_log.csv")
        error_df.to_csv(error_output_path, index=False)
        print(f"\nError log saved to: {error_output_path}")
    else:
        print("\nNo errors encountered during file loading.")

# -----------------------------------------------------------
# 6. Entry Point
# -----------------------------------------------------------
if __name__ == "__main__":
    main()
