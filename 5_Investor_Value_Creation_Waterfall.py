import numpy as np
import pandas as pd
from Utilities import compute_percentiles, firefly_plot, geometric_return
import matplotlib
from Utilities import generate_market_cap_class, waterfall_value_plot, geometric_return, dendrogram_plot, get_even_clusters
import matplotlib.pyplot as plt
import numpy as np
from Utilities import value_creation_waterfall
import glob
import os
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Utilities import value_creation_waterfall

# Load data
df_full = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\Global_data.csv")

# Filtering parameters
country_list = ["USA", 'AUS', 'INDIA', 'JAPAN', 'EURO', 'UK']
sector = df_full["Sector"].unique()  # Assuming you want to filter by all available sectors

# Filter data by Country & Sector
df_slice = df_full[(df_full['Country'].isin(country_list)) & (df_full["Sector"].isin(sector))]

# Time points
year_init = 2018
year_final = 2023

# Define genome classifications for filtering
genome_classification_init = ["CHALLENGED", "UNTENABLE", "VIRTUOUS", "TRAPPED"]
genome_classification_final = ["FAMOUS", "BRAVE", "FEARLESS", "LEGENDARY"]

# Filter data for initial and final years based on specific criteria
df_init = df_slice.loc[(df_slice["Year"] == year_init) &
                       (df_slice["Genome_classification_bespoke"].isin(genome_classification_init)) &
                       (df_slice["Sector"].isin(sector))]

df_final = df_slice.loc[(df_slice["Year"] == year_final) &
                        (df_slice["Genome_classification_bespoke"].isin(genome_classification_final)) &
                        (df_slice["Sector"].isin(sector))]

# Identify common companies between initial and final dataframes
common_companies = df_init[df_init["Company_name"].isin(df_final["Company_name"])]

# Filter to only include these common companies
df_init_filtered = df_init[df_init["Company_name"].isin(common_companies["Company_name"])]
df_final_filtered = df_final[df_final["Company_name"].isin(common_companies["Company_name"])]

# Calculate value creation metrics using a utility function
value_creation_waterfall(df_init_filtered, df_final_filtered, year_init, year_final)

# Extract relevant metrics for analysis
df_init_filtered["Revenue_init"] = df_init_filtered["Revenue"]
df_final_filtered["Revenue_final"] = df_final_filtered["Revenue"]
df_init_filtered["NPAT_margin_init"] = df_init_filtered["NPAT"] / df_init_filtered["Revenue"]
df_final_filtered["NPAT_margin_final"] = df_final_filtered["NPAT"] / df_final_filtered["Revenue"]
df_init_filtered["PE_init"] = df_init_filtered["Market_Capitalisation"] / df_init_filtered["NPAT"]
df_final_filtered["PE_final"] = df_final_filtered["Market_Capitalisation"] / df_final_filtered["NPAT"]

# Merge the initial and final dataframes on 'Company_name'
df_combined = pd.merge(df_init_filtered[["Company_name", "Revenue_init", "NPAT_margin_init", "PE_init"]],
                       df_final_filtered[["Company_name", "Revenue_final", "NPAT_margin_final", "PE_final"]],
                       on="Company_name")

# Calculate growth, margin differences, and changes in P/E ratio
df_combined["Revenue_growth"] = (df_combined["Revenue_final"] - df_combined["Revenue_init"]) / df_combined["Revenue_init"]
df_combined["Profit_margin_diff"] = df_combined["NPAT_margin_final"] - df_combined["NPAT_margin_init"]
df_combined["PE_diff"] = df_combined["PE_final"] - df_combined["PE_init"]

# Create summary table
df_summary = df_combined[["Company_name", "Revenue_init", "Revenue_final",
                          "Revenue_growth", "NPAT_margin_init", "NPAT_margin_final",
                          "Profit_margin_diff", "PE_init", "PE_final", "PE_diff"]]

# Print or display the summary table
print(df_summary.to_string(index=False))
