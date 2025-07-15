import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib

matplotlib.use('TkAgg')

"""
This is a tool to compute TSR Quartiles and generate the Firefly comparison at the market level
"""

# Import data
data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\Global_data.csv")

# Define countries and sectors to include
countries_to_include = ["AUS"] # 'USA', 'AUS', 'INDIA', 'JAPAN', 'EURO', 'UK'
sectors_to_include = ['Industrials', 'Materials', 'Healthcare', 'Technology',
                      'Insurance', 'Gaming/alcohol', 'Media', 'REIT', 'Utilities',
                      'Consumer staples', 'Consumer Discretionary',
                      'Investment and Wealth', 'Telecommunications', 'Energy', 'Banking',
                      'Metals', 'Financials - other', 'Mining', 'Consumer Staples',
                      'Diversified', 'Rail Transportation', 'Transportation']

# Filter data based on countries and sectors
filtered_data = data.loc[(data['Country'].isin(countries_to_include)) & (data['Sector'].isin(sectors_to_include))]

# Ensure relevant columns are numeric
filtered_data['TSR_CIQ_no_buybacks'] = pd.to_numeric(filtered_data['TSR_CIQ_no_buybacks'], errors='coerce')
filtered_data['Year'] = pd.to_numeric(filtered_data['Year'], errors='coerce')

# Drop rows with NaN or invalid values
filtered_data = filtered_data.dropna(subset=['TSR_CIQ_no_buybacks', 'Year'])

# Filter data for years between 2014 and 2024
filtered_data = filtered_data[(filtered_data['Year'] >= 2014) & (filtered_data['Year'] <= 2024)]

# Group data by Year and create a list of TSR values for each year
tsrs_by_year = [filtered_data.loc[filtered_data['Year'] == year, 'TSR_CIQ_no_buybacks'].values for year in range(2014, 2025)]

# Create a boxplot
fig, ax = plt.subplots(figsize=(12, 8))
ax.boxplot(tsrs_by_year, patch_artist=True, showfliers=False,
           boxprops=dict(facecolor='blue', color='black'),
           medianprops=dict(color='red'))

# Set axis labels and title
ax.set_xticks(range(1, 12))  # 2014 to 2024 corresponds to 11 years
ax.set_xticklabels(range(2014, 2025), rotation=45, ha='right')
ax.set_xlabel('Year')
ax.set_ylabel('TSR (Total Shareholder Return)')
plt.title('Yearly Distribution of TSR by Country and Sector')
plt.tight_layout()

# Save and display the plot
plt.savefig(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\TSR_Distribution_Boxplot.png")
plt.show()
