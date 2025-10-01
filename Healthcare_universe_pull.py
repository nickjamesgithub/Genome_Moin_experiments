import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import dataframe
df = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Genome_code_250605\Genome-pipeline-code\Genome-pipeline-code\global_platform_data\Global_data.csv")

# Healthcare dataframe
sector_include = ["Healthcare", "Health Care"]
healthcare_df = df.loc[(df["Sector"].isin(sector_include))]

# Select the relevant columns and drop duplicates
export = healthcare_df[["Company_name", "Sector", "Ticker_full"]].drop_duplicates(
    subset=["Company_name", "Ticker_full"]
).reset_index(drop=True)

healthcare_df.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\EBO\Healthcare_market_data\global_healthcare_data.csv")
export.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\EBO\Healthcare_market_data\global_healthcare_tickers.csv")

x=1
y=2