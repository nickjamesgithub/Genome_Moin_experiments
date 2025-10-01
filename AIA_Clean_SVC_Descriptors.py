import numpy as np
import pandas as pd

# --- Load data ---
df = pd.read_csv(
    r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\SVC_analysis\Insurance_SVC_data__CLEAN.csv"
)

# --- Parameters (edit as needed) ---
COST_OF_EQUITY = 0.078   # 7.8%  (set to 0.085 for 8.5%)
INFLATION = 0.027      # 3.2%  (set to 0.027 for 2.7%)

# --- Prep: ensure clean company names & numeric "Clean" columns ---
df2 = df.copy()
df2['Company'] = df2['Company'].astype(str).str.strip()

for col in ['Operating ROE Clean', 'GWP Growth Clean']:
    df2[col] = pd.to_numeric(df2[col], errors='coerce')

# --- Flags for the comparisons ---
df2['roe_gt_coe'] = df2['Operating ROE Clean'] > COST_OF_EQUITY
df2['gwp_gt_infl'] = df2['GWP Growth Clean'] > INFLATION

# --- Combined condition: BOTH criteria met (positive EP + real growth) ---
df2['svc'] = df2['roe_gt_coe'] & df2['gwp_gt_infl']

# --- Aggregate per company ---
out = (
    df2.groupby('Company', dropna=False)
       .agg(
           years_roe_gt_coe   = ('roe_gt_coe', 'sum'),
           years_gwp_gt_infl  = ('gwp_gt_infl', 'sum'),
           SVC                = ('svc', 'sum'),      # years both criteria satisfied
           total_years        = ('Year', 'nunique')
       )
       .reset_index()
)

# Optional: add shares
out['share_roe_gt_coe']  = (out['years_roe_gt_coe']  / out['total_years']).round(3)
out['share_gwp_gt_infl'] = (out['years_gwp_gt_infl'] / out['total_years']).round(3)
out['share_SVC']         = (out['SVC']               / out['total_years']).round(3)

# Sort for readability (top SVC first, then company)
out = out.sort_values(['SVC', 'Company'], ascending=[False, True], kind='stable')

# --- Results ---
print(out)

# Optional: save to CSV next to your input file
out.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\SVC_analysis\SVC_insurer_count_summary.csv", index=False)


x=1
y=2