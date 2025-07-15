import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
matplotlib.use('TkAgg')

# Read the data
df = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\Global_data.csv")

# Desired sectors and date range
country_list = ["USA", 'AUS', 'INDIA', 'JAPAN', 'EURO', 'UK'] # "USA", 'AUS', 'INDIA', 'JAPAN', 'EURO', 'UK'
unique_sectors = df["Sector"].unique()
desired_sectors = unique_sectors
start_year = 2019
end_year = 2024

# 1. Filter by Country & Sector
df_filtered = df[(df['Country'].isin(country_list)) & (df["Sector"].isin(desired_sectors))]
unique_companies = df_filtered["Company_name"].unique()

# Create a list to store features
features_list = []

for company in unique_companies:
    print(company)
    try:
        # Filter data for the company in start_year and end_year
        unique_company_init = df_filtered.loc[(df_filtered["Company_name"] == company) & (df_filtered["Year"] == start_year)]
        unique_company_final = df_filtered.loc[(df_filtered["Company_name"] == company) & (df_filtered["Year"] == end_year)]
        country = unique_company_final["Country"].values[0]

        # Genome segment initial & final
        genome_init = unique_company_init["Genome_classification_bespoke"].values[0]
        genome_final = unique_company_final["Genome_classification_bespoke"].values[0]

        # Sector
        sector = unique_company_init["Sector"].values[0]

        # Get the coordinates
        firefly_y_beginning = unique_company_init["EVA_ratio_bespoke"].values[0]
        firefly_y_end = unique_company_final["EVA_ratio_bespoke"].values[0]
        firefly_x_beginning = unique_company_init["Revenue_growth_3_f"].values[0]
        firefly_x_end = unique_company_final["Revenue_growth_3_f"].values[0]

        # Compute the angle in degrees
        radians = math.atan2(firefly_y_end - firefly_y_beginning,
                             firefly_x_end - firefly_x_beginning)
        degree_angle = math.degrees(radians)
        degrees = (degree_angle + 360) % 360  # Normalize angle to 0-360

        # Determine direction of movement
        if 0 <= degrees < 90:
            direction = "NORTH EAST"
        elif 90 <= degrees < 180:
            direction = "NORTH WEST"
        elif 180 <= degrees < 270:
            direction = "SOUTH WEST"
        else:
            direction = "SOUTH EAST"

        # Compute the changes
        change_in_ep_fe = firefly_y_end - firefly_y_beginning
        change_in_revenue_growth = firefly_x_end - firefly_x_beginning

        # Append data to the features list
        features_list.append([company, country, sector, firefly_x_beginning, firefly_x_end, firefly_y_beginning, firefly_y_end, genome_init, genome_final,
                              direction, degrees, change_in_ep_fe, change_in_revenue_growth])
        # Unique company
        print("Iteration ", company)

    except Exception as e:
        print("Issue in result iteration with ", company, ": ", e)

# Define column names using dynamic years
column_names = ["Company_name", "Country", "Sector", f"Revenue_growth_{start_year}", f"Revenue_growth_{end_year}",
                f"EVA_{start_year}", f"EVA_{end_year}", "Genome_initial", "Genome_final",
                "Direction", "Angle_degrees", "Change_in_EP_FE", "Change_in_Revenue_Growth"]

# Create a dataframe to store the results
df_features = pd.DataFrame(features_list, columns=column_names)

x=1
y=2