import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ------------------ Parameters ------------------ #
start_year = 2014
end_year   = 2024
n_years    = end_year - start_year + 1

selected_countries = [
    "Australia", "Canada", "Hong_Kong", "Singapore",
    "Netherlands", "USA", "United_Kingdom", "Japan"
]

# ------------------ Load and Combine Global + Bespoke ------------------ #
global_path   = r"C:\Users\60848\OneDrive - Bain\Desktop\Genome_code_250605\Genome-pipeline-code\Genome-pipeline-code\global_platform_data\Global_data.csv"
bespoke_path  = r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\bespoke_data.csv"

df_global  = pd.read_csv(global_path)
df_bespoke = pd.read_csv(bespoke_path)
df_all     = pd.concat([df_global, df_bespoke], ignore_index=True)

# Get insurers
insurers = df_global.loc[df_global["Sector"]=="Insurance"][["Company_name", "Country"]]

