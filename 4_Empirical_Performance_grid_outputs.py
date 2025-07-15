import numpy as np
import pandas as pd
from Utilities import compute_percentiles, firefly_plot, geometric_return
import matplotlib
from Utilities import generate_market_cap_class, waterfall_value_plot, geometric_return, dendrogram_plot, get_even_clusters
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import math

# Read the data
df_full = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Journeys\Journeys_summary_Global_FE_Update_filtered.csv")
# df_full = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Journeys\Journeys_summary_Global_filtered.csv")

# Journeys_summary_Global_filtered

# Desired sectors and date range
country_list = ["USA", 'AUS', 'INDIA', 'JAPAN', 'EURO', 'UK'] # "USA", 'AUS', 'INDIA', 'JAPAN', 'EURO', 'UK'
unique_sectors = df_full["Sector"].unique()
desired_sectors = unique_sectors
start_year = 2014
end_year = 2024

# 1. Filter by Country & Sector
df_slice = df_full # df_full[(df_full['Country'].isin(country_list)) & (df_full["Sector"]).isin(unique_sectors)]
# Get unique Genome classes
genome_class_unique = df_slice["Genome_classification_bespoke_beginning"].unique()
results_list = []

for i in range(len(genome_class_unique)):
    # Print Genome class
    print("Genome class ", genome_class_unique[i])

    ### NOTE - DATA CAPS BASED ON PREVIOUS ANALYSIS ###
    # Slice genome class
    df_genome_class_i = df_slice.loc[(df_slice["Genome_classification_bespoke_end"]==genome_class_unique[i]) &
                               (df_slice["EVA_end"] >= -.3) & (df_slice["EVA_end"] <= .5) &
                               (df_slice["Revenue_growth_end"] >= -.3) & (df_slice["Revenue_growth_end"] <= 1.5) &
                               (df_slice["Annualized_TSR_Capiq"] >= -.4) & (df_slice["Annualized_TSR_Capiq"] <= 1) &
                               (df_slice["Price_to_book"] > -200)]

    df_market_return_proxy = df_slice.loc[(df_slice["EVA_end"] >= -.3) & (df_slice["EVA_end"] <= .5) &
                               (df_slice["Revenue_growth_end"] >= -.3) & (df_slice["Revenue_growth_end"] <= 1.5) &
                               (df_slice["Annualized_TSR_Capiq"] >= -.4) & (df_slice["Annualized_TSR_Capiq"] <= 1) &
                               (df_slice["Price_to_book"] > -200)]
    # Market return proxy
    market_return_median = np.median(df_market_return_proxy["Annualized_TSR_Capiq"].values)

    # Compute annualized TSR values
    annualised_tsr_capiq_i = df_genome_class_i["Annualized_TSR_Capiq"].mean()
    # Compute standard deviation
    annualised_tsr_capiq_i_std = df_genome_class_i["Annualized_TSR_Capiq"].std()
    annualised_tsr_capiq_i_min = df_genome_class_i["Annualized_TSR_Capiq"].min()
    annualised_tsr_capiq_i_max = df_genome_class_i["Annualized_TSR_Capiq"].max()
    normalized_annualized_tsr = annualised_tsr_capiq_i - market_return_median
    price_to_book = df_genome_class_i["Price_to_book"].median()
    price_to_earnings = df_genome_class_i["PE_end"].median()

    # Get length
    n = len(df_genome_class_i)
    results_list.append([genome_class_unique[i], n, annualised_tsr_capiq_i, annualised_tsr_capiq_i_std, annualised_tsr_capiq_i_min,
                         annualised_tsr_capiq_i_max, normalized_annualized_tsr, price_to_book])

# Make results dataframe
results_df = pd.DataFrame(results_list)
results_df.columns = ["Genome_class", "N", "Annualized_TSR", "Standard_deviation_TSR", "Minimum_TSR", "Maximum_TSR", "Normalized_TSR", "Price_to_book"]
results_df.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Global_Genome_grid_output_FE_updated.csv")

x=1
y=2

