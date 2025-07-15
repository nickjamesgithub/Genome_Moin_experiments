import pandas as pd
import numpy as np

# ------------------ Load and Prepare Data ------------------ #
df = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Genome_code_250605\Genome-pipeline-code\Genome-pipeline-code\global_platform_data\Global_data.csv")

start_year = 2014
end_year = 2024
tsr_method = "capital_iq"  # or "bain"

# Filter to relevant years and Technology sector
df = df[(df["Year"] >= start_year) & (df["Year"] <= end_year)]
df = df[df["Sector"] == "Materials"].copy()

# Compute Price-to-Book
df["Price_to_Book"] = df["Market_Capitalisation"] / df["Book_Value_Equity"]

# ------------------ Compute Annualized TSR ------------------ #
results = []

for ticker in df["Ticker"].unique():
    df_ticker = df[df["Ticker"] == ticker].sort_values("Year")

    if df_ticker.empty or df_ticker["Year"].nunique() < (end_year - start_year + 1):
        continue  # Skip incomplete time series

    try:
        company_name = df_ticker["Company_name"].iloc[0]
        country = df_ticker["Country"].iloc[0]
        begin_price = df_ticker.loc[df_ticker["Year"] == start_year, "Stock_Price"].values[0]
        end_price = df_ticker.loc[df_ticker["Year"] == end_year, "Stock_Price"].values[0]
        begin_adj_price = df_ticker.loc[df_ticker["Year"] == start_year, "Adjusted_Stock_Price"].values[0]
        end_adj_price = df_ticker.loc[df_ticker["Year"] == end_year, "Adjusted_Stock_Price"].values[0]

        if begin_price == 0 or begin_adj_price == 0:
            continue

        dps_total = df_ticker[df_ticker["Year"] > start_year]["DPS"].sum()
        bbps_total = df_ticker[df_ticker["Year"] > start_year]["BBPS"].sum()
        n_years = end_year - start_year

        if tsr_method == "bain":
            cum_tsr = (end_price - begin_price + dps_total + bbps_total) / begin_price
        else:
            cum_tsr = (end_adj_price / begin_adj_price) - 1

        ann_tsr = (1 + cum_tsr) ** (1 / n_years) - 1

        avg_eva = df_ticker["EVA_ratio_bespoke"].mean()
        avg_rev_growth = df_ticker["Revenue_growth_3_f"].mean()

        results.append({
            "Company_name": company_name,
            "Country": country,
            "Avg_EVA_ratio_bespoke": avg_eva,
            "Avg_Revenue_growth_3_f": avg_rev_growth,
            "Annualized_TSR": ann_tsr
        })

    except (IndexError, KeyError, ValueError):
        continue  # Skip rows with missing or corrupt data

# ------------------ Create and Display Results Table ------------------ #
results_df = pd.DataFrame(results)
results_df.sort_values(by="Annualized_TSR", ascending=False, inplace=True)
results_df.reset_index(drop=True, inplace=True)

# Show top N (optional)
print(results_df.head(20))

x=1
y=2
