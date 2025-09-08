# --- Imports ---
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- Paths (edit these if needed) ---
GLOBAL_DATA_PATH     = r"C:\Users\60848\OneDrive - Bain\Desktop\Genome_code_250605\Genome-pipeline-code\Genome-pipeline-code\global_platform_data\Global_data.csv"
ASIAN_INSURANCE_PATH = r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\Insurance_data\merged_insurance_data_with_genome.csv"
OUT_MERGED_PATH      = r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\Insurance_data\merge_asian_global.csv"
OUT_TSR_QSUMMARY     = r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\Insurance_data\TSR_quartile_medians.csv"
OUT_CLUSTER_SUMMARY  = r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\Insurance_data\cluster_summary.csv"

# --- Load ---
global_data     = pd.read_csv(GLOBAL_DATA_PATH)
asian_insurance = pd.read_csv(ASIAN_INSURANCE_PATH)

# --- Keep Insurance rows from global data ---
insurance_global = global_data.loc[global_data["Sector"] == "Insurance"].copy()

# --- Stack on overlapping columns (row-wise union) ---
common_cols = asian_insurance.columns.intersection(insurance_global.columns)
stacked = (
    pd.concat([asian_insurance[common_cols], insurance_global[common_cols]], axis=0, ignore_index=True)
    .drop_duplicates(subset=list(common_cols), keep="first")
)
stacked.to_csv(OUT_MERGED_PATH, index=False)

# --- Prepare time/price fields ---
stacked["Year"] = pd.to_numeric(stacked.get("Year"), errors="coerce")
if "Adjusted_Stock_Price" in stacked:
    stacked["Adjusted_Stock_Price"] = pd.to_numeric(stacked["Adjusted_Stock_Price"], errors="coerce")

# -------------------- Feature construction --------------------
# Filter window
mask_years = (stacked["Year"] <= 2024) & (stacked["Year"] >= 2014)

# Count # years with positive ROE_above_Cost_of_equity
roe_pos_years = (
    stacked.loc[mask_years, ["Company_name", "ROE_above_Cost_of_equity"]]
           .assign(_roe=pd.to_numeric(stacked.loc[mask_years, "ROE_above_Cost_of_equity"], errors="coerce"))
           .assign(_pos=lambda d: (d["_roe"] > 0).astype(int))
           .groupby("Company_name")["_pos"].sum()
           .rename("ROE_positive_years")
)

# Company-level medians for extra metrics to add
# (P:BV, BVE_per_share_1_f, EVA_momentum) over mask_years
for col in ["PBV", "BVE_per_share_1_f", "EVA_momentum",
            "Revenue_growth_3_f", "ROE_above_Cost_of_equity"]:
    if col in stacked.columns:
        stacked[col] = pd.to_numeric(stacked[col], errors="coerce")

extra_medians = (
    stacked.loc[mask_years, ["Company_name", "PBV", "BVE_per_share_1_f", "EVA_momentum"]]
           .groupby("Company_name").median()
           .rename(columns={
               "PBV": "PBV_median",
               "BVE_per_share_1_f": "BVE_per_share_1_f_median",
               "EVA_momentum": "EVA_momentum_median"
           })
)

# Country (categorical): take the most frequent (mode) per company
if "Country" in stacked.columns:
    country_by_co = (
        stacked.loc[mask_years, ["Company_name", "Country"]]
               .groupby("Company_name")["Country"]
               .agg(lambda s: s.mode().iat[0] if not s.mode().empty
                    else (s.dropna().iat[0] if s.notna().any() else np.nan))
               .to_frame("Country")
    )
else:
    country_by_co = pd.DataFrame(columns=["Country"])

# Averages per company (2014–2024 window here)
v = (
    stacked.loc[mask_years, ["Company_name", "Revenue_growth_3_f", "ROE_above_Cost_of_equity"]]
           .groupby("Company_name")[["Revenue_growth_3_f", "ROE_above_Cost_of_equity"]]
           .mean()
           .merge(roe_pos_years, left_index=True, right_index=True, how="left")
           .merge(extra_medians, left_index=True, right_index=True, how="left")
           .merge(country_by_co, left_index=True, right_index=True, how="left")
)

# Annualised 10-year TSR: (Price_2024 / Price_2014)**(1/10) - 1
p = stacked[stacked["Year"].isin([2014, 2024])].pivot_table(
    index="Company_name", columns="Year", values="Adjusted_Stock_Price", aggfunc="last"
)
v["annualised TSR"] = (p.get(2024) / p.get(2014))**(1/10) - 1

v = v.reset_index()

# -------------------- K-Means clustering --------------------
features = ["Revenue_growth_3_f", "ROE_above_Cost_of_equity", "annualised TSR"]  # (keep same features)

# clean + prepare X (handle inf/NaN)
X = v[features].replace([np.inf, -np.inf], np.nan)
X = SimpleImputer(strategy="median").fit_transform(X)
X = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
v["cluster"] = kmeans.fit_predict(X)

# -------------------- Summaries --------------------
# Safety: replace +/-inf with NaN before aggregation
v_clean = v.replace([np.inf, -np.inf], np.nan)

# Cluster summary (medians) — now include P:BV, BVE_per_share_1_f, EVA_momentum
cluster_summary = (
    v_clean.groupby("cluster")
           .agg({
               "Revenue_growth_3_f": "median",
               "ROE_above_Cost_of_equity": "median",
               "annualised TSR": "median",
               "ROE_positive_years": "median",
               "PBV_median": "median",
               "BVE_per_share_1_f_median": "median",
               "EVA_momentum_median": "median",
               "Company_name": "count"
           })
           .rename(columns={"Company_name": "n_companies"})
           .reset_index()
)
cluster_summary.to_csv(OUT_CLUSTER_SUMMARY, index=False)
print("\nCluster summary (medians):")
print(cluster_summary)

# TSR quartiles (uniform spread) — add the same columns and Country to v_q
v_q = v_clean.copy()
v_q["TSR_quartile"] = pd.qcut(v_q["annualised TSR"], q=4, labels=False, duplicates="drop") + 1

quartile_medians = (
    v_q.groupby("TSR_quartile")
       .agg(
            Revenue_median=("Revenue_growth_3_f", "median"),
            ROE_median=("ROE_above_Cost_of_equity", "median"),
            ROE_pos_years_median=("ROE_positive_years", "median"),
            TSR_median=("annualised TSR", "median"),
            P_BV_median=("PBV_median", "median"),
            BVE_per_share_median=("BVE_per_share_1_f_median", "median"),
            EVA_momentum_median=("EVA_momentum_median", "median"),
            n_companies=("Company_name", "count")
       )
       .reset_index()
)
quartile_medians.to_csv(OUT_TSR_QSUMMARY, index=False)
print("\nTSR quartile medians:")
print(quartile_medians)

# v_q now carries Country at the row level already (from v); if you want a per-quartile top country:
# top country per quartile (optional)
# top_country = (v_q.groupby(['TSR_quartile', 'Country']).size()
#                  .groupby(level=0).idxmax().apply(lambda x: x[1]).rename('top_country'))
# print("\nTop country per quartile (optional):\n", top_country.reset_index())

# keep throwaway vars if you need them
x = 1
y = 2
