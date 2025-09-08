import pandas as pd

# Read in the datasets
global_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Genome_code_250605\Genome-pipeline-code\Genome-pipeline-code\global_platform_data\Global_data.csv")
global_insurance_data = global_data.loc[global_data["Sector"] == "Insurance"]
insurance_bespoke = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\insurance_data.csv")
# 1. Find extra columns
extra_cols = set(insurance_bespoke.columns) - set(global_insurance_data.columns)
print("Extra columns:", extra_cols)
# 2. Common columns, preserving the order from global_insurance_data
common_cols = [col for col in global_insurance_data.columns if col in insurance_bespoke.columns]

# 3. Merge (stack) vertically
merged = pd.concat(
    [global_insurance_data[common_cols], insurance_bespoke[common_cols]],
    axis=0
)

print(merged.shape)
print(merged.head())

merged.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\Clean\Clean_insurance_data_genome.csv")

x=1
y=2

# # Directory containing the CSVs
# data_dir = r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\bespoke"
# # List of tickers
# tickers = [
#     "1299","2628","2318","601601","601336","966","PRU","MFC","SLF","G07",
#     "8750","8795","7181","HDFCLIFE","ICICIPRULI","SBILIFE","A032830","A088350","TLI","BLA",
#     "AGS","PHNX","LGEN","PRU","AFL","2328","6060","CS","ZURN","CB","QBE","IAG","SUN",
#     "A005830","A001450","A000060","8725","8725","AIG","TRV","CNA","ADM","AV.","HSX","BEZ",
#     "TLX","HELN","BALN","CS","ALV","G","966","8725","8630","8766","AV.","MET","AGN","MAP",
#     "BVH","BKIH","TIPH","TAKAFUL","A000810","A000370","TUGU","CINF","LNC","PFG","PZU","VIG",
#     "ALL","PGR","UNM","HIG","2328","LICI","600015","2882","2881","MPL","CI","UNH","HUM","CVS"
# ]
# # Collect dataframes
# dfs = []
# for ticker in tickers:
#     file_path = os.path.join(data_dir, f"_{ticker}.csv")
#     if os.path.exists(file_path):
#         df = pd.read_csv(file_path)
#         df["ticker"] = ticker  # add ticker column for traceability
#         dfs.append(df)
#
# # Concatenate into one DataFrame
# combined_df = pd.concat(dfs, ignore_index=True)
# print(combined_df.shape)
