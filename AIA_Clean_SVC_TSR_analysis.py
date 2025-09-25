import pandas as pd
import numpy as np

# df = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\Clean\Clean_insurance_data_genome_country.csv")
df = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\insurance_data_clean_250925.csv")

# Filter to insurance only
stacked = df.loc[df["Sector"] == "Insurance"].copy()

# --- Normalize column name just in case ---
if "Ticker_full" in stacked.columns and "TICKER_FULL" not in stacked.columns:
    stacked = stacked.rename(columns={"Ticker_full": "TICKER_FULL"})

# --- Dedupe to keep full time series: one row per company-year ---
# If you have a preferred data source, sort to prioritize it *before* dropping duplicates.
# Example (optional): prefer rows with a price present, then by a Source priority if available.
sort_cols = []
if "Adjusted_Stock_Price" in stacked.columns:
    sort_cols.append(stacked["Adjusted_Stock_Price"].notna().astype(int))  # True (1) first
# If you have a 'Source' or 'Data_Source' column, you can add a priority map here:
# priority = {"PreferredSourceA": 0, "PreferredSourceB": 1}
# if "Source" in stacked.columns:
#     sort_cols.append(stacked["Source"].map(priority).fillna(9))

# Build a stable sort index if any preference defined
if sort_cols:
    stacked = stacked.assign(_pref=0)
    # Combine preferences into a tuple to sort descending (put best first)
    # Convert to DataFrame then to list of columns for sort_values
    stacked["_pref"] = pd.concat(sort_cols, axis=1).apply(tuple, axis=1)
    stacked = stacked.sort_values(by=["TICKER_FULL", "Year", "_pref"], ascending=[True, True, False])
else:
    # Fall back to a stable sort
    stacked = stacked.sort_values(by=["TICKER_FULL", "Year"])

# Now drop duplicates for (company, year)
stacked_panel = stacked.drop_duplicates(subset=["TICKER_FULL", "Year"], keep="first").drop(columns=["_pref"], errors="ignore")

# ----- Your TSR logic on the cleaned panel -----
END_YEAR = 2024
HORIZONS = [1, 3, 5, 10]

def compute_tsr(p_start, p_end, years):
    if pd.notna(p_start) and pd.notna(p_end) and p_start > 0:
        return (p_end / p_start) ** (1 / years) - 1
    return np.nan

results = []

# Iterate by company (use TICKER_FULL as the key; you can add Company_name alongside)
for ticker in sorted(stacked_panel["TICKER_FULL"].dropna().unique()):
    g = stacked_panel.loc[stacked_panel["TICKER_FULL"] == ticker].copy()

    # Optional descriptors (take first non-null)
    company = g["Company_name"].dropna().iloc[0] if "Company_name" in g and not g["Company_name"].dropna().empty else np.nan
    country = g["Country"].dropna().iloc[0] if "Country" in g and not g["Country"].dropna().empty else np.nan
    insurer_type = g["Type_of_Insurer"].dropna().iloc[0] if "Type_of_Insurer" in g and not g["Type_of_Insurer"].dropna().empty else np.nan

    # Year -> Adjusted_Stock_Price
    price_by_year = g.set_index("Year")["Adjusted_Stock_Price"].to_dict() if "Adjusted_Stock_Price" in g else {}

    p_end = price_by_year.get(END_YEAR, np.nan)

    row = {
        "TICKER_FULL": ticker,
        "Company_name": company,
        "Country": country,
        "Type_of_Insurer": insurer_type
    }

    for h in HORIZONS:
        start_year = END_YEAR - h
        p_start = price_by_year.get(start_year, np.nan)
        row[f"TSR_{h}yr"] = compute_tsr(p_start, p_end, h)

    results.append(row)

tsr_df = pd.DataFrame(results, columns=[
    "TICKER_FULL", "Company_name", "Country", "Type_of_Insurer",
    "TSR_1yr", "TSR_3yr", "TSR_5yr", "TSR_10yr"
])

# Quick sanity checks
print("Original stacked rows:", len(stacked))
print("Company-year rows after dedupe:", len(stacked_panel))
print(tsr_df.head())

tsr_df.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\SVC_analysis\TSR_insurers_messy.csv")

x=1
y=2
