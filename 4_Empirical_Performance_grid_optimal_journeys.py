import pandas as pd
from Utilities import generate_market_cap_class, waterfall_value_plot, geometric_return
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# Read the data
df_full = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Journeys_summary_Global.csv")

# Desired sectors and date range
country_list = ["USA", 'AUS', 'INDIA', 'JAPAN', 'EURO', 'UK'] # "USA", 'AUS', 'INDIA', 'JAPAN', 'EURO', 'UK'
unique_sectors = df_full["Sector"].unique()
desired_sectors = unique_sectors
start_year = 2014
end_year = 2024

# 1. Filter by Country & Sector
df = df_full[(df_full['Country'].isin(country_list)) & (df_full["Sector"]).isin(unique_sectors)]

# Get all the companies that start at trapped, brave, virtuous, famous

### BELOW THE LINE ###
df_untenable = df.loc[df["Genome_classification_bespoke_beginning"]=="UNTENABLE"]
df_trapped = df.loc[df["Genome_classification_bespoke_beginning"]=="TRAPPED"]
df_brave = df.loc[df["Genome_classification_bespoke_beginning"]=="BRAVE"]
df_fearless = df.loc[df["Genome_classification_bespoke_beginning"]=="FEARLESS"]

### ABOVE THE LINE ###
df_challenged = df.loc[df["Genome_classification_bespoke_beginning"]=="CHALLENGED"]
df_virtuous = df.loc[df["Genome_classification_bespoke_beginning"]=="VIRTUOUS"]
df_famous = df.loc[df["Genome_classification_bespoke_beginning"]=="FAMOUS"]
df_legendary = df.loc[df["Genome_classification_bespoke_beginning"]=="LEGENDARY"]

# Genome class
genome_class = "Trapped" # Untenable, Trapped, Brave, Fearless, Challenged, Virtuous, Famous, Legendary
# DONE ON STABLE: Untenable, fearless, Legendary, Challenged, Trapped, Brave, Virtuous

if genome_class == "Untenable":

    # Store values in Untenable & moves
    untenable_north_east_list = []
    untenable_north_list = []
    untenable_east_list = []
    untenable_stable_list = []

    for i in range(len(df_untenable)):
        # Data & company name
        untenable_i = df_untenable.iloc[i, :]
        company_i = df_untenable["Company_name"].iloc[i]

        # North East
        if (untenable_i["EVA_end"] > 0) and (untenable_i["Revenue_growth_end"] >= 0):
            print("North East")
            untenable_north_east_list.append(untenable_i)
        # North
        if (untenable_i["EVA_end"] > 0) and (untenable_i["Revenue_growth_end"] < 0):
            print("North")
            untenable_north_list.append(untenable_i)
        # EAST
        if (untenable_i["EVA_end"] < 0) and (untenable_i["Revenue_growth_end"] >= 0):
            print("East")
            untenable_east_list.append(untenable_i)
        # STABLE
        if (untenable_i["EVA_end"] < 0) and (untenable_i["Revenue_growth_end"] < 0):
            print("Stable")
            untenable_stable_list.append(untenable_i)

        # Print company and year range
        print(company_i + str(untenable_i["Year_beginning"]) + "-" + str(untenable_i["Year_final"]))

    # Turn these into dataframes
    untenable_north_east_df = pd.DataFrame(untenable_north_east_list)
    untenable_north_df = pd.DataFrame(untenable_north_list)
    untenable_east_df = pd.DataFrame(untenable_east_list)
    untenable_stable_df = pd.DataFrame(untenable_stable_list)

    # Print expected annualized TSR with moves
    try:
        print("Untenable, North East move", untenable_north_east_df["Annualized_TSR_Capiq"].median(), len(untenable_north_east_df))
    except:
        print("Untenable North East empty")
    try:
        print("Untenable, North", untenable_north_df["Annualized_TSR_Capiq"].median(), len(untenable_north_df))
    except:
        print("Untenable North empty")
    try:
        print("Untenable, East", untenable_east_df["Annualized_TSR_Capiq"].median(), len(untenable_east_df))
    except:
        print("Untenable, East empty")
    try:
        print("Untenable, Stable", untenable_stable_df["Annualized_TSR_Capiq"].median(), len(untenable_stable_df))
    except:
        print("Untenable, Stable empty")

if genome_class == "Fearless":

    # Store values in Fearless & moves
    fearless_north_west_list = []
    fearless_north_list = []
    fearless_west_list = []
    fearless_stable_list = []

    for i in range(len(df_fearless)):
        # Data & company name
        fearless_i = df_fearless.iloc[i, :]
        company_i = df_fearless["Company_name"].iloc[i]

        # North West
        if (fearless_i["EVA_end"] > 0) and (fearless_i["Revenue_growth_end"] < 0.2):
            print("North East")
            fearless_north_west_list.append(fearless_i)
        # North
        if (fearless_i["EVA_end"] > 0) and (fearless_i["Revenue_growth_end"] >= 0.2):
            print("North")
            fearless_north_list.append(fearless_i)
        # West
        if (fearless_i["EVA_end"] < 0) and (fearless_i["Revenue_growth_end"] < 0.2):
            print("West")
            fearless_west_list.append(fearless_i)
        # Stable
        if (fearless_i["EVA_end"] < 0) and (fearless_i["Revenue_growth_end"] >= 0.2):
            print("Stable")
            fearless_stable_list.append(fearless_i)

        # Print company and year range
        print(company_i + str(fearless_i["Year_beginning"]) + "-" + str(fearless_i["Year_final"]))

    # Turn these into dataframes
    fearless_north_west_df = pd.DataFrame(fearless_north_west_list)
    fearless_north_df = pd.DataFrame(fearless_north_list)
    fearless_west_df = pd.DataFrame(fearless_west_list)
    fearless_stable_df = pd.DataFrame(fearless_stable_list)

    # Print expected annualized TSR with moves
    try:
        print("Fearless, North West move", fearless_north_west_df["Annualized_TSR_Capiq"].median(), len(fearless_north_west_df))
    except:
        print("Fearless, North West empty")
    try:
        print("Fearless, North", fearless_north_df["Annualized_TSR_Capiq"].median(), len(fearless_north_df))
    except:
        print("Fearless, North empty")
    try:
        print("Fearless, West", fearless_west_df["Annualized_TSR_Capiq"].median(), len(fearless_west_df))
    except:
        print("Fearless, West empty")
    try:
        print("Fearless, Stable", fearless_stable_df["Annualized_TSR_Capiq"].median(), len(fearless_stable_df))
    except:
        print("Fearless, Stable empty")

if genome_class == "Legendary":

    # Store values in Legendary & moves
    legendary_south_west_list = []
    legendary_south_list = []
    legendary_west_list = []
    legendary_stable_list = []

    for i in range(len(df_legendary)):
        # Data & company name
        legendary_i = df_legendary.iloc[i, :]
        company_i = df_legendary["Company_name"].iloc[i]

        # South West
        if (legendary_i["EVA_end"] < 0) and (legendary_i["Revenue_growth_end"] < 0.2):
            print("South west")
            legendary_south_west_list.append(legendary_i)
        # South
        if (legendary_i["EVA_end"] < 0) and (legendary_i["Revenue_growth_end"] >= 0.2):
            print("South")
            legendary_south_list.append(legendary_i)
        # West
        if (legendary_i["EVA_end"] > 0) and (legendary_i["Revenue_growth_end"] < 0.2):
            print("East")
            legendary_west_list.append(legendary_i)
        # Stable
        if (legendary_i["EVA_end"] > 0) and (legendary_i["Revenue_growth_end"] >= 0.2):
            print("East")
            legendary_stable_list.append(legendary_i)

        # Print company and year range
        print(company_i + str(legendary_i["Year_beginning"]) + "-" + str(legendary_i["Year_final"]))

    # Turn these into dataframes
    legendary_south_west_df = pd.DataFrame(legendary_south_west_list)
    legendary_south_df = pd.DataFrame(legendary_south_list)
    legendary_west_df = pd.DataFrame(legendary_west_list)
    legendary_stable_df = pd.DataFrame(legendary_stable_list)

    # Print expected annualized TSR with moves
    try:
        print("Legendary, South West move", legendary_south_west_df["Annualized_TSR_Capiq"].median(), len(legendary_south_west_df))
    except:
        print("Legendary South west empty")
    try:
        print("Legendary, South", legendary_south_df["Annualized_TSR_Capiq"].median(), len(legendary_south_df))
    except:
        print("Legendary South empty")
    try:
        print("Legendary, West", legendary_west_df["Annualized_TSR_Capiq"].median(), len(legendary_west_df))
    except:
        print("Legendary, West empty")
    try:
        print("Legendary, Stable", legendary_stable_df["Annualized_TSR_Capiq"].median(), len(legendary_stable_df))
    except:
        print("Legendary, Stable empty")

if genome_class == "Challenged":

    # Store values in Challenged & moves
    challenged_south_east_list = []
    challenged_south_list = []
    challenged_east_list = []
    challenged_stable_list = []

    for i in range(len(df_challenged)):
        # Data & company name
        challenged_i = df_challenged.iloc[i, :]
        company_i = df_challenged["Company_name"].iloc[i]

        # South East
        if (challenged_i["EVA_end"] < 0) and (challenged_i["Revenue_growth_end"] >= 0):
            print("South East")
            challenged_south_east_list.append(challenged_i)
        # South
        if (challenged_i["EVA_end"] < 0) and (challenged_i["Revenue_growth_end"] < 0):
            print("South")
            challenged_south_list.append(challenged_i)
        # EAST
        if (challenged_i["EVA_end"] > 0) and (challenged_i["Revenue_growth_end"] >= 0):
            print("East")
            challenged_east_list.append(challenged_i)
        # STABLE
        if (challenged_i["EVA_end"] > 0) and (challenged_i["Revenue_growth_end"] < 0):
            print("Stable")
            challenged_stable_list.append(challenged_i)

        # Print company and year range
        print(company_i + str(challenged_i["Year_beginning"]) + "-" + str(challenged_i["Year_final"]))

    # Turn these into dataframes
    challenged_south_east_df = pd.DataFrame(challenged_south_east_list)
    challenged_south_df = pd.DataFrame(challenged_south_list)
    challenged_east_df = pd.DataFrame(challenged_east_list)
    challenged_stable_df = pd.DataFrame(challenged_stable_list)

    # Print expected annualized TSR with moves
    try:
        print("Challenged, South East move", challenged_south_east_df["Annualized_TSR_Capiq"].median(), len(challenged_south_east_df))
    except:
        print("Challenged, South East empty")
    try:
        print("Challenged, South", challenged_south_df["Annualized_TSR_Capiq"].median(), len(challenged_south_df))
    except:
        print("Challenged, South empty")
    try:
        print("Challenged, East", challenged_east_df["Annualized_TSR_Capiq"].median(), len(challenged_east_df))
    except:
        print("Challenged, East empty")
    try:
        print("Challenged, Stable", challenged_stable_df["Annualized_TSR_Capiq"].median(), len(challenged_stable_df))
    except:
        print("Challenged, Stable empty")

if genome_class == "Trapped":

    # Store values in Trapped & moves
    trapped_north_east_list = []
    trapped_north_list = []
    trapped_east_list = []
    trapped_north_west_list = []
    trapped_west_list = []
    trapped_stable_list = []

    for i in range(len(df_trapped)):
        # Data & company name
        trapped_i = df_trapped.iloc[i,:]
        company_i = df_trapped["Company_name"].iloc[i]

        # NORTH EAST
        if (trapped_i["EVA_end"] >= 0) and (trapped_i["Revenue_growth_end"] >= 0.10):
            print("North East")
            trapped_north_east_list.append(trapped_i)
        # NORTH
        if (trapped_i["EVA_end"] >= 0) and (trapped_i["Revenue_growth_end"] < 0.10) and (trapped_i["Revenue_growth_end"] > 0):
            print("North")
            trapped_north_list.append(trapped_i)
        # EAST
        if (trapped_i["EVA_end"] < 0) and (trapped_i["Revenue_growth_end"] >= 0.10):
            print("East")
            trapped_east_list.append(trapped_i)
        # NORTH WEST
        if (trapped_i["EVA_end"] >= 0) and (trapped_i["Revenue_growth_end"] < 0):
            print("North West")
            trapped_north_west_list.append(trapped_i)
        # WEST
        if (trapped_i["EVA_end"] < 0) and (trapped_i["Revenue_growth_end"] < 0):
            print("West")
            trapped_west_list.append(trapped_i)
        # STABLE
        if (trapped_i["EVA_end"] < 0) and (trapped_i["Revenue_growth_end"] > 0) and (trapped_i["Revenue_growth_end"] < 0.1):
            print("Stable")
            trapped_stable_list.append(trapped_i)

        # Print company and year range
        print(company_i + str(trapped_i["Year_beginning"]) + "-" + str(trapped_i["Year_final"]))

    # Turn these into dataframes
    trapped_north_east_df = pd.DataFrame(trapped_north_east_list)
    trapped_north_df = pd.DataFrame(trapped_north_list)
    trapped_east_df = pd.DataFrame(trapped_east_list)
    trapped_north_west_df = pd.DataFrame(trapped_north_west_list)
    trapped_west_df = pd.DataFrame(trapped_west_list)
    trapped_stable_df = pd.DataFrame(trapped_stable_list)

    # Print expected annualized TSR with moves
    try:
        print("Trapped, North East move", trapped_north_east_df["Annualized_TSR_Capiq"].median(), len(trapped_north_east_df))
    except:
        print("Trapped, North East empty")
    try:
        print("Trapped, North move", trapped_north_df["Annualized_TSR_Capiq"].median(), len(trapped_north_df))
    except:
        print("Trapped, North empty")
    try:
        print("Trapped, East move", trapped_east_df["Annualized_TSR_Capiq"].median(), len(trapped_east_df))
    except:
        print("Trapped, East empty")
    try:
        print("Trapped, North West move", trapped_north_west_df["Annualized_TSR_Capiq"].median(), len(trapped_north_west_df))
    except:
        print("Trapped, North West empty")
    try:
        print("Trapped, West move", trapped_west_df["Annualized_TSR_Capiq"].median(), len(trapped_west_df))
    except:
        print("Trapped, West empty")
    try:
        print("Trapped, Stable move", trapped_stable_df["Annualized_TSR_Capiq"].median(), len(trapped_stable_df))
    except:
        print("Trapped, Stable empty")

if genome_class == "Brave":

    # Store values in Brave & moves
    brave_north_east_list = []
    brave_north_list = []
    brave_east_list = []
    brave_north_west_list = []
    brave_west_list = []
    brave_stable_list = []

    for i in range(len(df_brave)):
        # Data & company name
        brave_i = df_brave.iloc[i,:]
        company_i = df_brave["Company_name"].iloc[i]

        # NORTH EAST
        if (brave_i["EVA_end"] >= 0) and (brave_i["Revenue_growth_end"] >= 0.20):
            print("North East")
            brave_north_east_list.append(brave_i)
        # NORTH
        if (brave_i["EVA_end"] >= 0) and (brave_i["Revenue_growth_end"] < 0.20) and (brave_i["Revenue_growth_end"] > 0.10):
            print("North")
            brave_north_list.append(brave_i)
        # EAST
        if (brave_i["EVA_end"] < 0) and (brave_i["Revenue_growth_end"] >= 0.20):
            print("East")
            brave_east_list.append(brave_i)
        # NORTH WEST
        if (brave_i["EVA_end"] >= 0) and (brave_i["Revenue_growth_end"] < 0.1):
            print("North West")
            brave_north_west_list.append(brave_i)
        # WEST
        if (brave_i["EVA_end"] < 0) and (brave_i["Revenue_growth_end"] < 0.1):
            print("West")
            brave_west_list.append(brave_i)
        # STABLE
        if (brave_i["EVA_end"] < 0) and (brave_i["Revenue_growth_end"] > 0.1) and (brave_i["Revenue_growth_end"] < 0.2):
            print("Stable")
            brave_stable_list.append(brave_i)

        # Print company and year range
        print(company_i + str(brave_i["Year_beginning"]) + "-" + str(brave_i["Year_final"]))

    # Turn these into dataframes
    brave_north_east_df = pd.DataFrame(brave_north_east_list)
    brave_north_df = pd.DataFrame(brave_north_list)
    brave_east_df = pd.DataFrame(brave_east_list)
    brave_north_west_df = pd.DataFrame(brave_north_west_list)
    brave_west_df = pd.DataFrame(brave_west_list)
    brave_stable_df = pd.DataFrame(brave_stable_list)

    # Print expected annualized TSR with moves
    try:
        print("Brave, North East move", brave_north_east_df["Annualized_TSR_Capiq"].median(), len(brave_north_east_df))
    except:
        print("Brave, North East empty")
    try:
        print("Brave, North move", brave_north_df["Annualized_TSR_Capiq"].median(), len(brave_north_df))
    except:
        print("Brave, North empty")
    try:
        print("Brave, East move", brave_east_df["Annualized_TSR_Capiq"].median(), len(brave_east_df))
    except:
        print("Brave, East empty")
    try:
        print("Brave, North West move", brave_north_west_df["Annualized_TSR_Capiq"].median(), len(brave_north_west_df))
    except:
        print("Brave, North West empty")
    try:
        print("Brave, West move", brave_west_df["Annualized_TSR_Capiq"].median(), len(brave_west_df))
    except:
        print("Brave, West empty")
    try:
        print("Brave, Stable move", brave_stable_df["Annualized_TSR_Capiq"].median(), len(brave_stable_df))
    except:
        print("Brave, Stable empty")

if genome_class == "Virtuous":

    # Store values in Virtuous & moves
    virtuous_south_east_list = []
    virtuous_south_list = []
    virtuous_south_west_list = []
    virtuous_east_list = []
    virtuous_west_list = []
    virtuous_stable_list = []

    for i in range(len(df_virtuous)):
        # Data & company name
        virtuous_i = df_virtuous.iloc[i, :]
        company_i = df_virtuous["Company_name"].iloc[i]

        # South East
        if (virtuous_i["EVA_end"] < 0) and (virtuous_i["Revenue_growth_end"] >= 0.10):
            print("South East")
            virtuous_south_east_list.append(virtuous_i)
        # South
        if (virtuous_i["EVA_end"] < 0) and (virtuous_i["Revenue_growth_end"] < 0.10) and (virtuous_i["Revenue_growth_end"] > 0):
            print("South")
            virtuous_south_list.append(virtuous_i)
        # South West
        if (virtuous_i["EVA_end"] < 0) and (virtuous_i["Revenue_growth_end"] < 0):
            print("South West")
            virtuous_south_west_list.append(virtuous_i)
        # EAST
        if (virtuous_i["EVA_end"] > 0) and (virtuous_i["Revenue_growth_end"] >= 0.10):
            print("East")
            virtuous_east_list.append(virtuous_i)
        # WEST
        if (virtuous_i["EVA_end"] > 0) and (virtuous_i["Revenue_growth_end"] < 0):
            print("West")
            virtuous_west_list.append(virtuous_i)
        # Stable
        if (virtuous_i["EVA_end"] > 0) and (virtuous_i["Revenue_growth_end"] > 0) and (virtuous_i["Revenue_growth_end"] < 0.1):
            print("Stable")
            virtuous_stable_list.append(virtuous_i)

        # Print company and year range
        print(company_i + str(virtuous_i["Year_beginning"]) + "-" + str(virtuous_i["Year_final"]))

    # Turn these into dataframes
    virtuous_south_east_df = pd.DataFrame(virtuous_south_east_list)
    virtuous_south_df = pd.DataFrame(virtuous_south_list)
    virtuous_south_west_df = pd.DataFrame(virtuous_south_west_list)
    virtuous_east_df = pd.DataFrame(virtuous_east_list)
    virtuous_west_df = pd.DataFrame(virtuous_west_list)
    virtuous_stable_df = pd.DataFrame(virtuous_stable_list)

    # Print expected annualized TSR with moves
    try:
        print("Virtuous, South East move", virtuous_south_east_df["Annualized_TSR_Capiq"].median(), len(virtuous_south_east_df))
    except:
        print("Virtuous, South East empty")
    try:
        print("Virtuous, South", virtuous_south_df["Annualized_TSR_Capiq"].median(), len(virtuous_south_df))
    except:
        print("Virtuous, South empty")
    try:
        print("Virtuous, South West", virtuous_south_west_df["Annualized_TSR_Capiq"].median(), len(virtuous_south_west_df))
    except:
        print("Virtuous, South West empty")
    try:
        print("Virtuous, East", virtuous_east_df["Annualized_TSR_Capiq"].median(), len(virtuous_east_df))
    except:
        print("Virtuous, East empty")
    try:
        print("Virtuous, West", virtuous_west_df["Annualized_TSR_Capiq"].median(), len(virtuous_west_df))
    except:
        print("Virtuous, West empty")
    try:
        print("Virtuous, Stable", virtuous_stable_df["Annualized_TSR_Capiq"].median(), len(virtuous_stable_df))
    except:
        print("Virtuous, Stable empty")

if genome_class == "Famous":

    # Store values in Famous & moves
    famous_south_east_list = []
    famous_south_list = []
    famous_south_west_list = []
    famous_east_list = []
    famous_west_list = []
    famous_stable_list = []

    for i in range(len(df_famous)):
        # Data & company name
        famous_i = df_famous.iloc[i, :]
        company_i = df_famous["Company_name"].iloc[i]

        # South East
        if (famous_i["EVA_end"] < 0) and (famous_i["Revenue_growth_end"] >= 0.20):
            print("South East")
            famous_south_east_list.append(famous_i)
        # South
        if (famous_i["EVA_end"] < 0) and (famous_i["Revenue_growth_end"] < 0.20) and (famous_i["Revenue_growth_end"] > 0.1):
            print("South")
            famous_south_list.append(famous_i)
        # South West
        if (famous_i["EVA_end"] < 0) and (famous_i["Revenue_growth_end"] < 0.1):
            print("South West")
            famous_south_west_list.append(famous_i)
        # EAST
        if (famous_i["EVA_end"] > 0) and (famous_i["Revenue_growth_end"] >= 0.20):
            print("East")
            famous_east_list.append(famous_i)
        # WEST
        if (famous_i["EVA_end"] > 0) and (famous_i["Revenue_growth_end"] < 0.1):
            print("West")
            famous_west_list.append(famous_i)
        # STABLE
        if (famous_i["EVA_end"] > 0) and (famous_i["Revenue_growth_end"] > 0.1) and (famous_i["Revenue_growth_end"] < 0.2):
            print("Stable")
            famous_stable_list.append(famous_i)

        # Print company and year range
        print(company_i + str(famous_i["Year_beginning"]) + "-" + str(famous_i["Year_final"]))

    # Turn these into dataframes
    famous_south_east_df = pd.DataFrame(famous_south_east_list)
    famous_south_df = pd.DataFrame(famous_south_list)
    famous_south_west_df = pd.DataFrame(famous_south_west_list)
    famous_east_df = pd.DataFrame(famous_east_list)
    famous_west_df = pd.DataFrame(famous_west_list)
    famous_stable_df = pd.DataFrame(famous_stable_list)

    # Print expected annualized TSR with moves
    try:
        print("Famous, South East move", famous_south_east_df["Annualized_TSR_Capiq"].median(), len(famous_south_east_df))
    except:
        print("Famous, South East empty")
    try:
        print("Famous, South", famous_south_df["Annualized_TSR_Capiq"].median(), len(famous_south_df))
    except:
        print("Famous, South empty")
    try:
        print("Famous, South West", famous_south_west_df["Annualized_TSR_Capiq"].median(), len(famous_south_west_df))
    except:
        print("Famous, South West empty")
    try:
        print("Famous, East", famous_east_df["Annualized_TSR_Capiq"].median(), len(famous_east_df))
    except:
        print("Famous, East empty")
    try:
        print("Famous, West", famous_west_df["Annualized_TSR_Capiq"].median(), len(famous_west_df))
    except:
        print("Famous, West empty")
    try:
        print("Famous, Stable", famous_stable_df["Annualized_TSR_Capiq"].median(), len(famous_stable_df))
    except:
        print("Famous, Stable empty")




