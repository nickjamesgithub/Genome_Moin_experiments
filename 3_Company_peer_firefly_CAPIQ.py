import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
from matplotlib.patches import Rectangle
import matplotlib
import itertools
matplotlib.use('TkAgg')

# Import data
data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\Global_data.csv")

# Full ticker list
full_ticker_list = ["XTRA:MBG", "BIT:RACE", "XTRA:VOW3"]
ticker_list = data.loc[data["Ticker_full"].isin(full_ticker_list)]["Ticker"].unique()
company_name_list = data.loc[data["Ticker_full"].isin(full_ticker_list)]["Company_name"].unique()
plot_label = "Peer_firefly"

# Store Economic profit and revenue growth
company_f1_list = []
company_f2_list = []
labels_year = []

for i in range(len(full_ticker_list)):
    company, ticker = full_ticker_list[i].split(":")
    # Loop over companies & countries
    country_i = data.loc[data["Ticker_full"]==full_ticker_list[i]]["Country"].values[0]
    ticker_i = ticker_list[i]
    # Update paths for raw data and share prices based on the country
    company_slice = pd.read_csv(fr"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\{country_i}\_{ticker_i}.csv")

    feature_1 = "Revenue_growth_3_f"
    feature_2 = "EVA_ratio_bespoke"

    # Get key genome criteria
    outputs = [feature_1, feature_2, "Year"]

    # Store values
    company_f1_list.append(company_slice[outputs[0]].values)
    company_f2_list.append(company_slice[outputs[1]].values)
    labels_year.append(company_slice[outputs[2]].values)

# Merged scripts
merged_f1 = list(itertools.chain(*company_f1_list))
merged_f2 = list(itertools.chain(*company_f2_list))

# Plot average metric for each company over the years
fig, ax = plt.subplots()
for i in range(len(company_f1_list)):
    if i == 0:
        plt.scatter(company_f1_list[i], company_f2_list[i], label=company_name_list[i], alpha=0.4, color='red')
        for j, txt in enumerate(labels_year[i]):
            plt.annotate(f"{labels_year[i][j]}" + " " + ticker_list[i][:3], (company_f1_list[i][j], company_f2_list[i][j]), fontsize=6)
    if i != 0:
        plt.scatter(company_f1_list[i], company_f2_list[i], label=company_name_list[i], alpha=0.4, color='blue')
        for j, txt in enumerate(labels_year[i]):
            plt.annotate(f"{labels_year[i][j]}" + " " + ticker_list[i][:3], (company_f1_list[i][j], company_f2_list[i][j]), fontsize=6)
    plt.xlabel(feature_1)
    plt.title(plot_label)
    plt.ylabel(feature_2)
    plt.legend()
plt.savefig("USA_Technology")
plt.show()
