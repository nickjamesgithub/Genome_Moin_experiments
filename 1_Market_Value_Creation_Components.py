import pandas as pd
from Utilities import generate_market_cap_class, waterfall_value_plot, geometric_return
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

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

# Infer metrics
filtered_data["Market_Capitalisation_inferred"] = filtered_data["Stock_Price"] * filtered_data["Shares_outstanding"]
filtered_data["PE_inferred"] = filtered_data["Stock_Price"]/filtered_data["Diluted_EPS"]

# Drop rows with missing values in key columns
features = ["Company_name", "Year", "Market_Capitalisation",  "Book_Value_Equity", "Shares_outstanding", "Dividends_Paid", "Stock_repurchased",
            "Dividend_Yield", "Buyback_Yield", "Dividend_Buyback_Yield", "ROTE", "PE", "PE_Implied",  "Diluted_EPS","NPAT", "Stock_Price"]

df_slice_ = filtered_data[features]
df_slice = df_slice_.dropna()
df_slice["BVE_per_share"] = df_slice["Book_Value_Equity"] / df_slice["Shares_outstanding"]

# Unique companies
unique_companies = df_slice["Company_name"].unique()

# Get initial & final years
beginning_year = 2011
end_year = 2023
n = end_year-beginning_year+1

# Loop over unique companies and compute differences
tsr_driver_list = []
for i in range(len(unique_companies)):

    # Get company i data
    company_i_data = df_slice.loc[df_slice["Company_name"]==unique_companies[i]]

    # Return on equity (productivity)
    try:
        roe_beginning = company_i_data.loc[company_i_data["Year"]==beginning_year]["ROTE"].values[0]
        roe_end = company_i_data.loc[company_i_data["Year"]==end_year]["ROTE"].values[0]
        roe_growth = geometric_return(roe_end, roe_beginning, n) * 100
    except:
        roe_difference = "N/A"

    # Book value of equity (Growth)
    try:
        bve_beginning = company_i_data.loc[company_i_data["Year"]==beginning_year]["BVE_per_share"].values[0]
        bve_end = company_i_data.loc[company_i_data["Year"]==end_year]["BVE_per_share"].values[0]
        bve_growth = geometric_return(bve_end, bve_beginning, n) * 100
    except:
       bve_difference = "N/A"

    # P/E Multiple change
    try:
        pe_beginning = company_i_data.loc[company_i_data["Year"] == beginning_year]["PE_Implied"].values[0]
        pe_end = company_i_data.loc[company_i_data["Year"] == end_year]["PE_Implied"].values[0]

        if pe_beginning <= 0 or pe_end <= 0 or np.isinf(pe_beginning) or np.isinf(pe_end):
            pe_growth = np.nan
        else:
            pe_growth = geometric_return(pe_end, pe_beginning, n) * 100 # Assuming n is defined elsewhere
    except:
        pe_growth = np.nan

    # Yields for dividends and buybacks
    try:
        dividend_yield_avg = company_i_data.loc[(company_i_data["Year"] >= beginning_year) & (company_i_data["Year"] <= end_year)]["Dividend_Yield"].mean() * 100
        buyback_yield_avg = company_i_data.loc[(company_i_data["Year"] >= beginning_year) & (company_i_data["Year"] <= end_year)]["Buyback_Yield"].mean() * 100
    except:
        dividend_yield_avg = np.nan
        buyback_yield_avg = np.nan

    # Append TSR drivers to list
    tsr_driver_list.append([unique_companies[i], roe_growth, bve_growth, pe_growth, dividend_yield_avg, buyback_yield_avg])
    print("TSR drivers for ", unique_companies[i])

# Make dataframe
tsr_driver_df = pd.DataFrame(tsr_driver_list)
tsr_driver_df.columns = ["Company", "ROE Change (productivity)", "BVE Change (growth)", "P/E Change (expectations)", "Dividend_yield*", "Buyback_yield*"]

# Calculate median values for each feature
# Convert non-numeric values to NaN
tsr_driver_df_numeric = tsr_driver_df.apply(pd.to_numeric, errors='coerce')
# Replace infinite values with NaN
tsr_driver_df_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)

# Calculate mean values for each feature
median_values = tsr_driver_df_numeric.median()

# Separate positive and negative values
positive_values = median_values[median_values >= 0]
negative_values = median_values[median_values < 0]

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))

# Plot positive values in green
bars_pos = ax.barh(positive_values.index, positive_values, color='green', label='Positive')

# Plot negative values in red
bars_neg = ax.barh(negative_values.index, negative_values, color='red', label='Negative')

# Add labels and title
ax.set_xlabel('Median Company change (%)')  # Adjusted x-axis label
ax.set_ylabel('TSR Driver CAGR')
ax.set_title('TSR Driver change in median company')

# Add legend
ax.legend()

# Add percentage values over the bars
for bars in [bars_pos, bars_neg]:
    for bar in bars:
        width = bar.get_width()
        ax.annotate(f'{width:.2f}%',                      # Display percentage value with 2 decimal places
                    xy=(width, bar.get_y() + bar.get_height() / 2),  # Positioning the label at the center of the bar
                    xytext=(3, 0),                          # Offset of label from the bar
                    textcoords="offset points",
                    ha='left', va='center')

# Tight layout to fit everything in
plt.tight_layout()

# Rotate x-axis labels for better readability
plt.xticks(rotation=90, ha='right')

# Show plot
plt.show()