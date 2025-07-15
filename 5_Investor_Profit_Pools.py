import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io as pio
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress
import matplotlib
matplotlib.use('TkAgg')

# Set plotting backend
pio.renderers.default = 'browser'

# Read the data
df_full = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\Global_data.csv")

# Define variables for analysis
return_metric = "EVA_ratio_bespoke"
growth_metric = "TSR_CIQ_no_buybacks"

# Desired sectors and date range
country_list = ["USA", 'AUS', 'INDIA', 'JAPAN', 'EURO', 'UK'] # "USA", 'AUS', 'INDIA', 'JAPAN', 'EURO', 'UK'
unique_sectors = df_full["Sector"].unique()
desired_sectors = unique_sectors

# 1. Filter by Country & Sector
df_slice = df_full[(df_full['Country'].isin(country_list)) & (df_full["Sector"].isin(unique_sectors))]
tickers_ = df_slice["Ticker_full"].unique()

# Get unique tickers
unique_tickers = df_slice["Ticker"].unique()

# Define the year range for filtering
start_year = 2019
end_year = 2024

# Remove "UNKNOWN" from Genome_classification_bespoke
df_slice = df_slice[df_slice['Genome_classification_bespoke'] != 'UNKNOWN']

# Filter the dataframe by the year range
df_filtered = df_slice[(df_slice['Year'] >= start_year) & (df_slice['Year'] <= end_year)]

# Group by 'Sector' and calculate the median values
df_grouped_return_sector = df_filtered.groupby('Sector')[return_metric].median().reset_index()
df_grouped_growth_sector = df_filtered.groupby('Sector')[growth_metric].median().reset_index()

# Group by 'Genome_classification' and calculate the median values
df_grouped_genome = df_filtered.groupby('Genome_classification_bespoke')[[return_metric, growth_metric]].median().reset_index()

# Create a figure with three subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: Average return metric per Sector
axes[0, 0].bar(df_grouped_return_sector['Sector'], df_grouped_return_sector[return_metric], color='skyblue')
axes[0, 0].set_title(f'Median {return_metric} per Sector ({start_year} - {end_year})')
axes[0, 0].set_xlabel('Sector')
axes[0, 0].set_ylabel(f'Median {return_metric}')
axes[0, 0].tick_params(axis='x', rotation=90)

# Subplot 2: Median growth metric per Sector
axes[0, 1].bar(df_grouped_growth_sector['Sector'], df_grouped_growth_sector[growth_metric], color='lightgreen')
axes[0, 1].set_title(f'Median {growth_metric} per Sector ({start_year} - {end_year})')
axes[0, 1].set_xlabel('Sector')
axes[0, 1].set_ylabel(f'Median {growth_metric}')
axes[0, 1].tick_params(axis='x', rotation=90)

# Subplot 3: Median return and growth metric per Genome Classification
bar_width = 0.35
index = range(len(df_grouped_genome['Genome_classification_bespoke']))
axes[1, 0].bar(index, df_grouped_genome[return_metric], bar_width, label=return_metric, color='skyblue')
axes[1, 0].bar([i + bar_width for i in index], df_grouped_genome[growth_metric], bar_width, label=growth_metric, color='lightgreen')
axes[1, 0].set_title(f'Median {return_metric} and {growth_metric} per Genome Classification ({start_year} - {end_year})')
axes[1, 0].set_xlabel('Genome Classification')
axes[1, 0].set_ylabel('Median Value')
axes[1, 0].set_xticks([i + bar_width / 2 for i in index])
axes[1, 0].set_xticklabels(df_grouped_genome['Genome_classification_bespoke'])
axes[1, 0].legend()

# Subplot 4: Scatter plot with regression line of growth metric vs return metric (5th - 95th Percentile)
lower_percentile_return = df_filtered[return_metric].quantile(0.05)
upper_percentile_return = df_filtered[return_metric].quantile(0.95)
lower_percentile_growth = df_filtered[growth_metric].quantile(0.05)
upper_percentile_growth = df_filtered[growth_metric].quantile(0.95)

# Filter data within percentiles
df_filtered_percentile = df_filtered[
    (df_filtered[return_metric] >= lower_percentile_return) & (df_filtered[return_metric] <= upper_percentile_return) &
    (df_filtered[growth_metric] >= lower_percentile_growth) & (df_filtered[growth_metric] <= upper_percentile_growth)
]

# Scatter plot with regression line using Seaborn
model = LinearRegression()
m_ = model.fit(np.array(df_filtered_percentile[growth_metric]).reshape(-1,1), np.array(df_filtered_percentile[return_metric]).reshape(-1,1))
intercept = m_.intercept_[0]

# Add vertical line at 0
plt.axvline(x=0, color='red', linestyle='--')
sns.regplot(x=growth_metric, y=return_metric, data=df_filtered_percentile, scatter_kws={'s': 10, 'alpha':0.2, 'color':'orange'}, ax=axes[1, 1])

# Add labels and title
axes[1, 1].set_title(f'{growth_metric} vs {return_metric} (5th - 95th Percentile)')
axes[1, 1].set_xlabel(growth_metric)
axes[1, 1].set_ylabel(return_metric)

# Annotate expected TSR for companies breaking above the line
axes[1, 1].text(0.5, 0.9, f'Expected {return_metric} for companies at the line:\n{intercept * 100 :.2f}% (intercept)', ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)

# Adjust layout to make room for the labels
plt.tight_layout()

# Display the plots
plt.savefig("Market_Revenue_breakdown")
plt.show()
