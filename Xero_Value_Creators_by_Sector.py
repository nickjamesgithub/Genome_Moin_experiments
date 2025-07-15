import pandas as pd
import numpy as np

# ------------------ Load and Prepare Data ------------------ #
df = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Genome_code_250605\Genome-pipeline-code\Genome-pipeline-code\global_platform_data\Global_data.csv")

start_year = 2014
end_year = 2024
tsr_method = "capital_iq"  # or "bain"

# Define selected countries
selected_countries = [
    'Australia', 'Belgium', 'Canada', 'Denmark', 'France', 'Germany', 'Hong_Kong',
    'Italy', 'Japan', 'Luxembourg', 'Netherlands', 'Singapore', 'South_Korea',
    'Sweden', 'Switzerland', 'United_Kingdom', 'USA'
]

# Filter to valid years and countries
df = df[(df["Year"] >= start_year) & (df["Year"] <= end_year)].copy()
df = df[df["Country"].isin(selected_countries)].copy()

# Calculate Price-to-Book ratio
df["Price_to_Book"] = df["Market_Capitalisation"] / df["Book_Value_Equity"]


# ------------------ TSR Analysis Function ------------------ #
def compute_top_companies(df_input, top_n=5, label="Global"):
    top_df_list = []
    sectors = df_input["Sector"].dropna().unique()

    print(f"\n--- Starting {label} Top {top_n} Analysis ---")
    print(f"Total sectors to process: {len(sectors)}\n")

    for idx, sector in enumerate(sectors):
        print(f"Processing sector {idx + 1}/{len(sectors)}: {sector}")
        df_sector = df_input[df_input["Sector"] == sector]
        tickers = df_sector["Ticker_full"].dropna().unique()
        rows = []

        for ticker in tickers:
            try:
                df_ticker = df_sector[df_sector["Ticker_full"] == ticker].copy()
                if df_ticker["Year"].nunique() < (end_year - start_year + 1):
                    continue

                # Get key values
                begin_price = df_ticker.loc[df_ticker["Year"] == start_year, "Stock_Price"].values[0]
                end_price = df_ticker.loc[df_ticker["Year"] == end_year, "Stock_Price"].values[0]
                begin_adj_price = df_ticker.loc[df_ticker["Year"] == start_year, "Adjusted_Stock_Price"].values[0]
                end_adj_price = df_ticker.loc[df_ticker["Year"] == end_year, "Adjusted_Stock_Price"].values[0]

                if begin_price == 0 or begin_adj_price == 0:
                    continue

                dps_total = df_ticker[df_ticker["Year"] > start_year]["DPS"].sum()
                bbps_total = df_ticker[df_ticker["Year"] > start_year]["BBPS"].sum()

                if tsr_method == "bain":
                    cum_tsr = (end_price - begin_price + dps_total + bbps_total) / begin_price
                else:
                    cum_tsr = (end_adj_price / begin_adj_price) - 1

                n_years = end_year - start_year
                ann_tsr = (1 + cum_tsr) ** (1 / n_years) - 1

                rows.append({
                    "Sector": sector,
                    "Company": df_ticker["Company_name"].iloc[0],
                    "Ticker": ticker,
                    "Country": df_ticker["Country"].iloc[0],
                    "Annualized_TSR": ann_tsr,
                    "Avg_EVA": df_ticker["EVA_ratio_bespoke"].mean(),
                    "Avg_Revenue_Growth": df_ticker["Revenue_growth_3_f"].mean()
                })

            except Exception:
                continue

        df_summary = pd.DataFrame(rows)

        if not df_summary.empty and len(df_summary) >= top_n:
            df_sorted = df_summary.sort_values(by="Annualized_TSR", ascending=False).reset_index(drop=True)
            top_df = df_sorted.head(top_n).copy()
            top_df["Rank"] = [f"Top {i+1}" for i in range(len(top_df))]
            top_df_list.append(top_df)
        else:
            print(f"  ⚠ Skipping sector '{sector}' due to insufficient valid companies.\n")

    final_top = pd.concat(top_df_list, ignore_index=True) if top_df_list else pd.DataFrame()
    return final_top


# ------------------ Run Global and Australian Analysis ------------------ #
# Global top 5 per sector
global_top5_df = compute_top_companies(df, top_n=6, label="Global")

# Australia top 5 per sector
df_aus = df[df["Country"] == "Australia"].copy()
aus_top5_df = compute_top_companies(df_aus, top_n=6, label="Australia")

# ------------------ Export & Preview ------------------ #
global_top5_df.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\Xero\Global_top_5_by_sector.csv", index=False)
aus_top5_df.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\Xero\Australia_top_5_by_sector.csv", index=False)

print("\nSample: Australia Top 5 TSR by Sector:\n", aus_top5_df.head())

x=1
y=2