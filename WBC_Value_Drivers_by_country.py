import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# ------------------ Parameters ------------------ #
start_year = 2015
end_year = 2024
n_years = end_year - start_year + 1

# ------------------ Country Classifications ------------------ #
emerging_markets = [
    "Saudi_Arabia", "India", "Thailand", "China",
    "Philippines", "Malaysia", "Chile"
]

developed_markets = [
    "USA", "Japan", "Singapore", "Hong_Kong", "South_Korea",
    "Italy", "Sweden", "Australia", "Switzerland", "France",
    "United_Kingdom", "Netherlands", "Denmark", "Belgium", "Germany"
]

# ------------------ Load and Combine Data ------------------ #
global_path = r"C:\Users\60848\OneDrive - Bain\Desktop\Genome_code_250605\Genome-pipeline-code\Genome-pipeline-code\global_platform_data\Global_data.csv"
bespoke_path = r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\bespoke_data.csv"

df_global = pd.read_csv(global_path)
df_bespoke = pd.read_csv(bespoke_path)
df_all = pd.concat([df_global, df_bespoke], ignore_index=True)

# ------------------ Preprocessing ------------------ #
# Filter for Banking sector
df_banking = df_all[df_all["Sector"] == "Banking"]

# Map bespoke tickers to real countries
bespoke_country_map = {
    "D05": "Singapore",
    "SAN": "Spain",
    "MQG": "Australia",
    "RY": "Canada",
    "ITUB4": "Brazil"
}

# Replace 'bespoke' country using ticker mapping
df_banking["Country"] = df_banking.apply(
    lambda row: bespoke_country_map.get(row["Ticker"], row["Country"]) if row["Country"] == "bespoke" else row["Country"],
    axis=1
)

# ------------------ Filter Australia to Only 4 Banks ------------------ #
australia_banks = [
    "Commonwealth Bank of Australia",
    "Westpac Banking Corporation",
    "National Australia Bank",
    "ANZ Group Holdings Limited"
]

df_banking = df_banking[
    ~((df_banking["Country"] == "Australia") & (~df_banking["Company_name"].isin(australia_banks)))
]

# ------------------ Feature Engineering ------------------ #
df_banking["TE_per_share_"] = df_banking["Tangible_equity"] / df_banking["Shares_outstanding"]
df_banking["CROTE_"] = df_banking["NPAT"] / df_banking["Tangible_equity"]
df_banking["P:CR"] = df_banking["Market_Capitalisation"] / df_banking["NPAT"]

# ------------------ Assign Market Group ------------------ #
def classify_market(country):
    if country == "Australia":
        return "Big 4 (Australia)"
    elif country in emerging_markets:
        return "Emerging"
    elif country in developed_markets:
        return "Developed"
    else:
        return None

df_banking["Market_Group"] = df_banking["Country"].apply(classify_market)

# ------------------ Filter by Year and Valid Groups ------------------ #
df_grouped = df_banking[df_banking["Market_Group"].notna() & df_banking["Year"].between(start_year, end_year)]

# ------------------ Plot Variables (Median) ------------------ #
variables = {
    "TE_per_share_": "Tangible Equity / Share",
    "CROTE_": "CROTE",
    "P:CR": "Price to Cash Earnings",
    "Dividend_Buyback_Yield": "Dividend + Buyback Yield"
}

# ------------------ Plot Variables (Median) with Custom Formatting & Safe Filenames ------------------ #
for var, label in variables.items():
    plt.figure(figsize=(10, 6))
    median_df = (
        df_grouped.groupby(["Year", "Market_Group"])[var]
        .median()
        .reset_index()
    )
    sns.lineplot(data=median_df, x="Year", y=var, hue="Market_Group")
    plt.title(f"Median {label} by Market Group Over Time")
    plt.ylabel(label)
    plt.xlabel("Year")

    # Custom y-axis formatting
    if var in ["CROTE_", "Dividend_Buyback_Yield"]:
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1%}"))
    elif var in ["TE_per_share_", "P:CR"]:
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1f}x"))

    # Safe filename by removing or replacing invalid characters
    safe_var = var.replace(":", "_")
    filename = f"median_{safe_var}_by_market_group.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()