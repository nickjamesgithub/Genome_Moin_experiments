import pandas as pd
import numpy as np
import os
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

# ---------- Load Master Data ---------- #
global_df = pd.read_csv(
    r"C:\Users\60848\OneDrive - Bain\Desktop\Genome_code_250605\Genome-pipeline-code\Genome-pipeline-code\global_platform_data\Global_data.csv"
)

# ---------- Filter for Technology Sector and Year Range ---------- #
tech_df = global_df[
    (global_df['Sector'] == 'Technology') &
    (global_df['Year'].between(2014, 2024))
]

# ---------- Compute Average EVA and Growth Per Ticker ---------- #
meta_df = tech_df.groupby(['Ticker', 'Country'], as_index=False).agg({
    'EVA_ratio_bespoke': 'mean',
    'Revenue_growth_3_f': 'mean'
})

# ---------- Filter Reasonable Values ---------- #
meta_df = meta_df[
    (meta_df['EVA_ratio_bespoke'].between(-1, 1)) &
    (meta_df['Revenue_growth_3_f'].between(-0.8, 3))
].drop_duplicates(subset='Ticker')

# ---------- Load Share Price Data ---------- #
base_dir = r"C:\Users\60848\OneDrive - Bain\Desktop\Genome_code_250605\Genome-pipeline-code\Genome-pipeline-code\global_platform_data\share_price"
price_data = {}

for _, row in meta_df[['Ticker', 'Country']].drop_duplicates().iterrows():
    ticker, country = row['Ticker'], row['Country']
    file_path = os.path.join(base_dir, country, f"_{ticker}_price.csv")
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, parse_dates=['Date'])
            df = df[['Date', 'Price']].rename(columns={'Price': ticker})
            df.set_index('Date', inplace=True)
            price_data[ticker] = df
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    else:
        print(f"Missing file: {file_path}")

if not price_data:
    raise ValueError("No price data files were loaded.")

# ---------- Combine All Price Data ---------- #
combined_df = pd.concat(price_data.values(), axis=1).sort_index()
log_returns = np.log(combined_df / combined_df.shift(1)).dropna()

# ---------- Keep Only Valid Tickers ---------- #
valid_tickers = sorted(set(meta_df['Ticker']) & set(log_returns.columns))
log_returns = log_returns[valid_tickers]
meta_df = meta_df[meta_df['Ticker'].isin(valid_tickers)].copy()

# ---------- Compute Cosine Distance Matrix ---------- #
normalized_returns = normalize(log_returns.T)
distance_matrix = squareform(pdist(normalized_returns, metric='cosine'))
distance_df = pd.DataFrame(distance_matrix, index=valid_tickers, columns=valid_tickers)
distance_vectors = distance_df.values

# ---------- KMeans Clustering ---------- #
kmeans = KMeans(n_clusters=5, random_state=0)
meta_df['Cluster'] = kmeans.fit_predict(distance_vectors)

# ---------- Hierarchical Clustering ---------- #
linkage_matrix = linkage(distance_vectors, method='ward')
meta_df['Dendro_Cluster'] = fcluster(linkage_matrix, t=6, criterion='maxclust')

# ---------- Cluster Summary ---------- #
summary = meta_df.groupby('Cluster').agg(
    EVA_ratio_bespoke=('EVA_ratio_bespoke', 'mean'),
    Revenue_growth_3_f=('Revenue_growth_3_f', 'mean'),
    Country=('Country', lambda x: x.value_counts(normalize=True).to_dict()),
    N=('Ticker', 'count')
).reset_index()

# ---------- Plot Dendrogram WITHOUT Ticker Labels ---------- #
plt.figure(figsize=(12, 7))
dendrogram(linkage_matrix, labels=None, no_labels=True, color_threshold=None)
plt.title('Dendrogram of Technology Company Stock Patterns (Cosine Distance)')
plt.xlabel('Companies')
plt.ylabel('Distance')
plt.tight_layout()
plt.show()

# ---------- Output Summary ---------- #
print("\nCluster Summary:")
print(summary)

x=1
y=2

# # ---------- Optional: Save results ---------- #
# summary.to_csv("cluster_summary.csv", index=False)
# meta_df.to_csv("ticker_cluster_assignments.csv", index=False)
