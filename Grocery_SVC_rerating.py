import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path


matplotlib.use('TkAgg')

# Load the main data
df = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\grocery_data.csv")

# Filter data for the required years
start_year = 2014
end_year = 2024
df = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]

# Drop duplicates by considering only the first occurrence of each company for each year
df = df.drop_duplicates(subset=['Company_name', 'Year'])

# Apply the criteria and create a new column for whether criteria are met each year
df['SVC_Criteria_Met'] = (df['EVA_ratio_bespoke'] > 0) & (df['Revenue_growth_1_f'] > 0.027)

# Group by company and count the number of years criteria is met
criteria_count = df.groupby('Company_name')['SVC_Criteria_Met'].sum()

# Create a dictionary to collect companies by the number of years they meet the criteria
svc_summary_dict = {i: [] for i in range(0, end_year - start_year + 2)}  # from 0 to 11 years

for company, count in criteria_count.items():
    svc_summary_dict[count].append(company)

# Convert dictionary into a DataFrame for display
svc_summary = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in svc_summary_dict.items()])).fillna('')
svc_summary.columns = ["SVC_0", "SVC_1", "SVC_2", "SVC_3", "SVC_4", "SVC_5", "SVC_6", "SVC_7", "SVC_8", "SVC_9", "SVC_10", "SVC_11"]
print(svc_summary)

# Write out to local file
svc_summary.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\svc_summary_global.csv")

### Test if valuation is associated with SVC ###
# Initialize dictionary to store median PBV values for each SVC category
pbv_medians = {}
# Iterate through SVC categories (columns) in svc_summary
for column in svc_summary.columns:
    # Get the list of companies for the current SVC category
    companies = svc_summary[column].dropna().tolist()
    # Filter df for these companies
    filtered_df = df[df["Company_name"].isin(companies)]
    # Compute the median PBV and store it in the dictionary
    pbv_medians[column] = filtered_df["PBV"].median()
# Convert the results dictionary to a DataFrame
pbv_summary = pd.DataFrame(list(pbv_medians.items()), columns=['SVC_Category', 'Median_PBV'])
# Print the resulting DataFrame
print(pbv_summary)

# Plot SVC vs Median PBV
plt.plot(pbv_summary["SVC_Category"][:-1], pbv_summary["Median_PBV"][:-1])
plt.ylabel("Average P:BV")
plt.xlabel("SVC Years")
plt.title("Grocery SVC Criteria met vs P:BV")
plt.show()

OUTDIR = Path(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\Woolworths\SVC_data")
GLOBAL = Path(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\grocery_data.csv")

# --- load ---
tsr   = pd.read_csv(OUTDIR / "grocery_company_tsr_1_3_5_10_years_with_counts.csv")
panel = pd.read_csv(GLOBAL)

# -----------------------------
# SVC vs Non-SVC: Median 10Y TSR
# -----------------------------

# Define the 10-year SVC window explicitly (last 10 years up to end_year)
svc_window_years = 10
svc_start = end_year - svc_window_years + 1  # e.g., 2015 if end_year=2024

df_window = df[(df['Year'] >= svc_start) & (df['Year'] <= end_year)].copy()

# Recompute the criteria on the window (defensive copy already above)
df_window['SVC_Criteria_Met'] = (df_window['EVA_ratio_bespoke'] > 0) & (df_window['Revenue_growth_1_f'] > 0.027)

# Count "met" years in the 10-year window per company
svc_counts = (
    df_window.groupby('Company_name', as_index=False)['SVC_Criteria_Met']
             .sum()
             .rename(columns={'SVC_Criteria_Met': 'SVC_years_met_10y'})
)

# Build SVC flag: >=8/10
svc_counts['SVC_flag'] = svc_counts['SVC_years_met_10y'] >= 8

# --- Join to TSR data and pick the 10Y TSR column ---
# Try to detect the 10-year TSR column automatically
tsr_cols_lower = {c: c.lower() for c in tsr.columns}
teny_candidates = [c for c in tsr.columns if ('10' in tsr_cols_lower[c]) and ('tsr' in tsr_cols_lower[c])]
if len(teny_candidates) == 0:
    # Fallback common names if auto-detect fails; adjust if your file uses a different header
    fallback_candidates = ['TSR_10Y', 'TSR_10yr', 'TSR_10_year', 'TSR_10y', 'Ten_Year_TSR', 'tsr_10y']
    teny_candidates = [c for c in tsr.columns if c in fallback_candidates]

if len(teny_candidates) == 0:
    raise ValueError(
        "Couldn't auto-detect the 10-year TSR column. "
        "Please update `teny_col` to the correct column name in the TSR file."
    )

teny_col = teny_candidates[0]

# Keep only company + 10Y TSR
tsr_10y = tsr[['Company_name', teny_col]].copy()

# Merge SVC flags onto TSR
tsr_svc = tsr_10y.merge(svc_counts[['Company_name', 'SVC_flag', 'SVC_years_met_10y']],
                        on='Company_name', how='left')

# If some companies in TSR aren't in the panel window (unlikely but possible), mark as Non-SVC by default
tsr_svc['SVC_flag'] = tsr_svc['SVC_flag'].fillna(False).astype(bool)
tsr_svc['SVC_label'] = tsr_svc['SVC_flag'].map({True: 'SVC', False: 'Non-SVC'})

# -----------------------------
# 1) Summary table: group median 10Y TSR
# -----------------------------
summary_tbl = (
    tsr_svc.groupby('SVC_label', as_index=False)
           .agg(Median_TSR_10Y=(teny_col, 'median'),
                Count=('Company_name', 'nunique'))
           .sort_values('SVC_label', ascending=False)
)

print("\n=== Median 10-Year TSR by SVC status ===")
print(summary_tbl)

# -----------------------------
# 2) Company-level table
# -----------------------------
company_tbl = (
    tsr_svc[['Company_name', 'SVC_label', 'SVC_years_met_10y', teny_col]]
        .rename(columns={
            'SVC_label': 'SVC_Status',
            'SVC_years_met_10y': 'SVC_Years_Met_in_Last_10',
            teny_col: 'TSR_10Y'
        })
        .sort_values(['SVC_Status', 'Company_name'])
        .reset_index(drop=True)
)

print("\n=== Company 10-Year TSR with SVC flag ===")
print(company_tbl)

# -----------------------------
# Save outputs
# -----------------------------
summary_out = OUTDIR / "svc_vs_non_svc_median_10y_tsr.csv"
company_out = OUTDIR / "company_10y_tsr_with_svc_flag.csv"

summary_tbl.to_csv(summary_out, index=False)
company_tbl.to_csv(company_out, index=False)

print(f"\nSaved summary to: {summary_out}")
print(f"Saved company-level table to: {company_out}")

x=1
y=2