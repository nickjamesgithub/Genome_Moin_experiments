import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ------------------ Setup ------------------ #
start_year = 2015
end_year = 2024
n_years = end_year - start_year

included_countries = [
    "Australia", "Denmark", "Hong_Kong", "India", "Malaysia", "Netherlands",
    "Singapore", "Sweden", "Switzerland", "Thailand", "USA", "United_Kingdom"
]

# ------------------ Define Processing Function ------------------ #
def process_dataset(df, source_label, is_bespoke=False):
    df = df.copy()
    df['Source'] = source_label

    # ✅ Fix: Conditional country filtering
    if not is_bespoke:
        df = df[
            (df["Sector"] == "Banking") &
            (df["Country"].isin(included_countries)) &
            (df["Year"].between(start_year, end_year))
        ].copy()
    else:
        df = df[
            (df["Sector"] == "Banking") &
            (df["Year"].between(start_year, end_year))
        ].copy()
        df["Country"] = "Bespoke"  # override

    # ✅ Filter for tickers with full coverage
    complete_tickers = (
        df.groupby("Ticker_full")["Year"]
        .nunique()
        .reset_index()
        .query(f"Year == {n_years + 1}")["Ticker_full"]
        .tolist()
    )
    df = df[df["Ticker_full"].isin(complete_tickers)].copy()

    df["BVE_per_Share"] = df["Book_Value_Equity"] / df["Shares_outstanding"]
    df = df.sort_values(["Company_name", "Year"])

    decomp_list = []
    metrics_list = []

    for company, group in df.groupby("Company_name"):
        try:
            group = group.sort_values("Year")

            price_start = group.loc[group["Year"] == start_year, "Adjusted_Stock_Price"].values[0]
            price_end = group.loc[group["Year"] == end_year, "Adjusted_Stock_Price"].values[0]
            pbv_start = group.loc[group["Year"] == start_year, "PBV"].values[0]
            pbv_end = group.loc[group["Year"] == end_year, "PBV"].values[0]
            bveps_start = group.loc[group["Year"] == start_year, "BVE_per_Share"].values[0]
            bveps_end = group.loc[group["Year"] == end_year, "BVE_per_Share"].values[0]
            cash_return = group["Dividend_Buyback_Yield"].mean()

            if price_start == 0 or pd.isnull(price_start) or pd.isnull(price_end):
                continue

            ann_tsr = ((price_end / price_start) ** (1 / n_years) - 1) * 100
            pbv_cagr = ((pbv_end / pbv_start) ** (1 / n_years) - 1) * 100
            bveps_cagr = ((bveps_end / bveps_start) ** (1 / n_years) - 1) * 100
            cash_return_pct = cash_return * 100

            decomp_list.append({
                "Company_name": company,
                "Country": group["Country"].iloc[0],
                "Source": source_label,
                "Annualized_TSR_%": ann_tsr,
                "PBV_CAGR_%": pbv_cagr,
                "BVEps_CAGR_%": bveps_cagr,
                "Cash_Return_%": cash_return_pct
            })

            avg_eva = group["EVA_ratio_bespoke"].mean()
            avg_growth = group["Revenue_growth_3_f"].mean()
            eva_pos = (group["EVA_ratio_bespoke"] >= 0).sum()
            growth_pos = (group["Revenue_growth_3_f"] >= 0.03).sum()

            metrics_list.append({
                "Company_name": company,
                "Avg_EVA_ratio_bespoke": avg_eva,
                "Avg_Revenue_growth_3_f": avg_growth,
                "Years_EVA_Positive": eva_pos,
                "Years_growth_Positive": growth_pos
            })

        except Exception:
            continue

    decomp_df = pd.DataFrame(decomp_list)
    metrics_df = pd.DataFrame(metrics_list)

    if not metrics_df.empty and 'Company_name' in metrics_df.columns:
        final_df = pd.merge(decomp_df, metrics_df, on="Company_name", how="left")
    else:
        # fallback if EVA/growth metrics were empty
        decomp_df["Avg_EVA_ratio_bespoke"] = np.nan
        decomp_df["Avg_Revenue_growth_3_f"] = np.nan
        decomp_df["Years_EVA_Positive"] = np.nan
        decomp_df["Years_growth_Positive"] = np.nan
        final_df = decomp_df

    return final_df

# ------------------ Load Data ------------------ #
global_path = r"C:\Users\60848\OneDrive - Bain\Desktop\Genome_code_250605\Genome-pipeline-code\Genome-pipeline-code\global_platform_data\Global_data.csv"
bespoke_path = r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\bespoke_data.csv"

df_global = pd.read_csv(global_path)
df_bespoke = pd.read_csv(bespoke_path)

# ------------------ Process Separately ------------------ #
global_final = process_dataset(df_global, "Global", is_bespoke=False)
bespoke_final = process_dataset(df_bespoke, "Bespoke", is_bespoke=True)

# ------------------ Combine ------------------ #
final_df = pd.concat([global_final, bespoke_final], ignore_index=True)

# ------------------ Add Average Row ------------------ #
average_row = {
    "Company_name": "Average",
    "Country": "",
    "Source": "",
    "Annualized_TSR_%": final_df["Annualized_TSR_%"].mean(),
    "PBV_CAGR_%": final_df["PBV_CAGR_%"].mean(),
    "BVEps_CAGR_%": final_df["BVEps_CAGR_%"].mean(),
    "Cash_Return_%": final_df["Cash_Return_%"].mean(),
    "Avg_EVA_ratio_bespoke": final_df["Avg_EVA_ratio_bespoke"].mean(),
    "Avg_Revenue_growth_3_f": final_df["Avg_Revenue_growth_3_f"].mean(),
    "Years_EVA_Positive": final_df["Years_EVA_Positive"].mean(),
    "Years_growth_Positive": final_df["Years_growth_Positive"].mean()
}

final_df = pd.concat([final_df, pd.DataFrame([average_row])], ignore_index=True)


x=1
y=2