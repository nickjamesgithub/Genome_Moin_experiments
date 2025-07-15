import pandas as pd
import numpy as np

filter = False

# ------------------ Load and Prepare Global Data ------------------ #
df = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Genome_code_250605\Genome-pipeline-code\Genome-pipeline-code\global_platform_data\Global_data.csv")

start_year = 2015
end_year = 2024
tsr_method = "capital_iq"  # or "bain"

# Filter to relevant years and Banking sector
df = df[(df["Year"] >= start_year) & (df["Year"] <= end_year)]
df = df[df["Sector"] == "Banking"].copy()

if filter:
    # Filter by selected countries
    selected_countries = [
        "Australia", "Denmark", "Hong_Kong", "Italy", "Malaysia", "Netherlands",
        "Singapore", "Sweden", "Switzerland", "Thailand", "USA", "United_Kingdom"
    ]
    df = df[df["Country"].isin(selected_countries)].copy()

# Compute Price-to-Book
df["Price_to_Book"] = df["Market_Capitalisation"] / df["Book_Value_Equity"]

# ------------------ Compute TSR for Global Data ------------------ #
results = []

for ticker in df["Ticker"].unique():
    df_ticker = df[df["Ticker"] == ticker].sort_values("Year")

    if df_ticker.empty or df_ticker["Year"].nunique() < (end_year - start_year + 1):
        continue

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
        eva_positive_years = (df_ticker["EVA_ratio_bespoke"] >= 0).sum()
        growth_positive_years = (df_ticker["Revenue_growth_3_f"] >= 0.03).sum()

        results.append({
            "Company_name": company_name,
            "Country": country,
            "Avg_EVA_ratio_bespoke": avg_eva,
            "Avg_Revenue_growth_3_f": avg_rev_growth,
            "Annualized_TSR": ann_tsr,
            "Years_EVA_Positive": eva_positive_years,
            "Years_growth_Positive": growth_positive_years,
        })

    except (IndexError, KeyError, ValueError):
        continue

# Convert to DataFrame
results_df = pd.DataFrame(results)

# ------------------ Add Companies from bespoke_data.csv ------------------ #
bespoke_df = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\bespoke_data.csv")

# Filter for Banking sector and valid years
bespoke_df = bespoke_df[(bespoke_df["Sector"] == "Banking") &
                        (bespoke_df["Year"] >= start_year) &
                        (bespoke_df["Year"] <= end_year)].copy()

bespoke_results = []

for ticker in bespoke_df["Ticker"].unique():
    df_ticker = bespoke_df[bespoke_df["Ticker"] == ticker].sort_values("Year")

    if df_ticker.empty or df_ticker["Year"].nunique() < (end_year - start_year + 1):
        continue

    try:
        company_name = df_ticker["Company_name"].iloc[0]
        country = df_ticker["Country"].iloc[0]
        begin_adj_price = df_ticker.loc[df_ticker["Year"] == start_year, "Adjusted_Stock_Price"].values[0]
        end_adj_price = df_ticker.loc[df_ticker["Year"] == end_year, "Adjusted_Stock_Price"].values[0]

        if begin_adj_price == 0 or pd.isnull(begin_adj_price) or pd.isnull(end_adj_price):
            continue

        cum_tsr = (end_adj_price / begin_adj_price) - 1
        ann_tsr = (1 + cum_tsr) ** (1 / (end_year - start_year)) - 1

        avg_eva = df_ticker["EVA_ratio_bespoke"].mean()
        avg_rev_growth = df_ticker["Revenue_growth_3_f"].mean()
        eva_positive_years = (df_ticker["EVA_ratio_bespoke"] >= 0).sum()
        growth_positive_years = (df_ticker["Revenue_growth_3_f"] >= 0.03).sum()

        bespoke_results.append({
            "Company_name": company_name,
            "Country": country,
            "Avg_EVA_ratio_bespoke": avg_eva,
            "Avg_Revenue_growth_3_f": avg_rev_growth,
            "Annualized_TSR": ann_tsr,
            "Years_EVA_Positive": eva_positive_years,
            "Years_growth_Positive": growth_positive_years
        })

    except (IndexError, KeyError, ValueError):
        continue

# Append bespoke results to main results
bespoke_df_final = pd.DataFrame(bespoke_results)
results_df = pd.concat([results_df, bespoke_df_final], ignore_index=True)

# ------------------ Sort and Add Average Row ------------------ #
results_df.sort_values(by="Annualized_TSR", ascending=False, inplace=True)
results_df.reset_index(drop=True, inplace=True)

average_row = {
    "Company_name": "Average",
    "Country": "",
    "Avg_EVA_ratio_bespoke": results_df["Avg_EVA_ratio_bespoke"].mean(),
    "Avg_Revenue_growth_3_f": results_df["Avg_Revenue_growth_3_f"].mean(),
    "Annualized_TSR": results_df["Annualized_TSR"].mean(),
    "Years_EVA_Positive": results_df["Years_EVA_Positive"].mean(),
    "Years_growth_Positive": results_df["Years_growth_Positive"].mean()
}

results_df = pd.concat([results_df, pd.DataFrame([average_row])], ignore_index=True)
results_df.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\WBC\Capital_markets_review\Banking_Value_Creators.csv")

# ------------------ Display Final Table ------------------ #
print(results_df.tail(21))  # Shows bottom 20 + average row

x=1
y=2