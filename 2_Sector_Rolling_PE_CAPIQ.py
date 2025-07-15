import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib
import string
matplotlib.use('TkAgg')

"""
This is a tool to compute an evolutionary Firefly at a country/sector level
"""

# Import data
data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\Global_data.csv")

# Define countries and sectors to include
countries_to_include = ['AUS'] # 'USA', 'AUS', 'INDIA', 'JAPAN', 'EURO', 'UK'
sectors_to_include = ["Banking"]
plot_label = "AUS_Banking"

# 'Industrials', 'Materials', 'Healthcare', 'Technology',
#        'Insurance', 'Gaming/alcohol', 'Media', 'REIT', 'Utilities',
#        'Consumer staples', 'Consumer Discretionary',
#        'Investment and Wealth', 'Telecommunications', 'Energy', 'Banking',
#        'Metals', 'Financials - other', 'Mining', 'Consumer Staples',
#        'Diversified', 'Rail Transportation', 'Transportation'

# Filter the data based on the selected countries and sectors
filtered_data = data.copy()
filtered_data = filtered_data[filtered_data["Country"].isin(countries_to_include)]
filtered_data = filtered_data[filtered_data["Sector"].isin(sectors_to_include)]

tickers_ = filtered_data["Ticker"].unique()

# Store Black/Grey/White values
pe_list = []
dates_list = []

for i in range(len(tickers_)):
    # Company ticker
    company_i = tickers_[i]
    try:
        # Determine the country for the current ticker
        country_i = filtered_data.loc[filtered_data["Ticker"] == company_i, "Country"].values[0]

        # Update paths for raw data and share prices based on the country
        df = pd.read_csv(fr"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\{country_i}\_{company_i}.csv")
        price_df = pd.read_csv(fr"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\share_price\{country_i}\_{company_i}_price.csv")
        price_df["Year"] = pd.DatetimeIndex(price_df["Date"]).year

        # Set conditions to conditionally populate P/E
        conditions = [
            (price_df['Year'] == 2010),
            (price_df['Year'] == 2021),
            (price_df['Year'] == 2022),
            (price_df['Year'] == 2023),
            (price_df['Year'] == 2024),
        ]

        # Create a list of the values we want to assign for each condition
        values = [
            df.loc[df["Year"] == 2010, "Diluted_EPS"],
            df.loc[df["Year"] == 2021, "Diluted_EPS"],
            df.loc[df["Year"] == 2022, "Diluted_EPS"],
            df.loc[df["Year"] == 2023, "Diluted_EPS"],
            df.loc[df["Year"] == 2024, "Diluted_EPS"],
        ]

        # Create a new column and use np.select to assign values to it using our lists as arguments
        price_df["Diluted_EPS_Year"] = np.select(conditions, values, default=np.nan)

        # Filter out rows where Diluted_EPS_Year is NaN or <= 0
        price_df = price_df[(price_df["Diluted_EPS_Year"].notna()) & (price_df["Diluted_EPS_Year"] > 0)]

        # Add a column for rolling P/E based on rolling Price/EPS
        price_df["PE_rolling"] = price_df["Price"] / price_df["Diluted_EPS_Year"]

        # Append to lists
        pe_list.append(price_df["PE_rolling"])
        dates_list.append(price_df["Date"])
    except Exception as e:
        print(f"Issue with company {tickers_[i]}: {e}")

fig, ax = plt.subplots()
for i in range(len(pe_list)):
    date_grid = pd.date_range(min(dates_list[i]), max(dates_list[i]), len(pe_list[i]))
    plt.plot(date_grid, pe_list[i], label=tickers_[i])
    plt.title(plot_label)
    company_flat = plot_label.translate(str.maketrans('', '', string.punctuation))

ax.xaxis.set_major_locator(plt.MaxNLocator(5))
ax.yaxis.set_major_locator(plt.MaxNLocator(5))
plt.legend()
plt.savefig("PE_" + str(plot_label))
plt.show()