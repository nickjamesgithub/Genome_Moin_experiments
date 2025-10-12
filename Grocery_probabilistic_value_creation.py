# 10-year TSR top-quartile medians + ±1 SD bands for EVA_ratio_bespoke and Revenue_growth_3_f

import pandas as pd
import numpy as np

CSV_PATH = r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\grocery_data.csv"
YEAR_START, YEAR_END = 2014, 2024
SPAN_YEARS = YEAR_END - YEAR_START

df = pd.read_csv(CSV_PATH)

# --- Columns ---
company_col = 'Company_name'
year_col = 'Year'
eva_col = 'EVA_ratio_bespoke'
rev_col = 'Revenue_growth_3_f'

if 'Adjusted_Stock_Price' in df.columns:
    price_col = 'Adjusted_Stock_Price'
elif 'Adjusted_stock_price' in df.columns:
    price_col = 'Adjusted_stock_price'
else:
    raise KeyError("Price column not found.")

df = df[[company_col, year_col, price_col, eva_col, rev_col]].copy()
df.columns = ['company', 'year', 'price', 'eva', 'rev3f']
df = df[df['year'].between(YEAR_START, YEAR_END)]

# --- Aggregate if multiple rows per company-year ---
df_agg = (
    df.groupby(['company', 'year'], as_index=False)
      .agg(price=('price', 'mean'),
           eva=('eva', 'mean'),
           rev3f=('rev3f', 'mean'))
)

# --- Endpoint prices and TSR ---
p2014 = df_agg[df_agg['year'] == YEAR_START][['company', 'price']].rename(columns={'price': 'price_2014'})
p2024 = df_agg[df_agg['year'] == YEAR_END][['company', 'price']].rename(columns={'price': 'price_2024'})

prices = p2014.merge(p2024, on='company', how='inner')
prices = prices[(prices['price_2014'] > 0) & (prices['price_2024'] > 0)]
prices['tsr_10y'] = (prices['price_2024'] / prices['price_2014']) ** (1 / SPAN_YEARS) - 1

# --- Top quartile companies ---
q75 = prices['tsr_10y'].quantile(0.75)
top_companies = prices.loc[prices['tsr_10y'] >= q75, 'company'].unique()
top_panel = df_agg[df_agg['company'].isin(top_companies)]

# --- Medians and ±1 std dev ---
eva_median = top_panel['eva'].median(skipna=True)
eva_std = top_panel['eva'].std(skipna=True)
rev3f_median = top_panel['rev3f'].median(skipna=True)
rev3f_std = top_panel['rev3f'].std(skipna=True)

result = {
    'companies_with_2014_and_2024_prices': int(prices['company'].nunique()),
    'top_quartile_threshold_tsr': float(q75),
    'top_quartile_company_count': int(len(top_companies)),
    'EVA_ratio_bespoke': {
        'median': float(eva_median),
        'lower_bound': float(eva_median - eva_std),
        'upper_bound': float(eva_median + eva_std)
    },
    'Revenue_growth_3_f': {
        'median': float(rev3f_median),
        'lower_bound': float(rev3f_median - rev3f_std),
        'upper_bound': float(rev3f_median + rev3f_std)
    }
}

print(result)

x=1
y=2