# --- Imports ---
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ===== Paths for outputs (added; change as needed) =====
OUT_CLUSTER_SUMMARY = "cluster_summary.csv"
OUT_TSR_QSUMMARY = "tsr_quartile_medians.csv"

# ===== Load INSURANCE file =====
df = pd.read_csv(
    r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\Clean\Clean_insurance_data_genome_country.csv"
)

stacked = df.loc[df["Sector"] == "Insurance"].copy()

# --- Prepare time/price fields ---
stacked["Year"] = pd.to_numeric(stacked.get("Year"), errors="coerce")
if "Adjusted_Stock_Price" in stacked.columns:
    stacked["Adjusted_Stock_Price"] = pd.to_numeric(stacked["Adjusted_Stock_Price"], errors="coerce")

# -------------------- Feature construction --------------------
# Filter window
mask_years = (stacked["Year"] <= 2024) & (stacked["Year"] >= 2014)

# Count # years with positive ROE_above_Cost_of_equity
roe_series = pd.to_numeric(stacked.loc[mask_years, "ROE_above_Cost_of_equity"], errors="coerce")
roe_pos_years = (
    stacked.loc[mask_years, ["Company_name"]]
           .assign(_roe=roe_series)
           .assign(_pos=lambda d: (d["_roe"] > 0).astype(int))
           .groupby("Company_name")["_pos"].sum()
           .rename("ROE_positive_years")
)

# Coerce numerics we need for medians/means
for col in ["PBV", "BVE_per_share_1_f", "EVA_momentum",
            "Revenue_growth_3_f", "ROE_above_Cost_of_equity"]:
    if col in stacked.columns:
        stacked[col] = pd.to_numeric(stacked[col], errors="coerce")

# Company-level medians (PBV, BVE_per_share_1_f, EVA_momentum)
extra_medians = (
    stacked.loc[mask_years, ["Company_name", "PBV", "BVE_per_share_1_f", "EVA_momentum"]]
           .groupby("Company_name").median(numeric_only=True)
           .rename(columns={
               "PBV": "PBV_median",
               "BVE_per_share_1_f": "BVE_per_share_1_f_median",
               "EVA_momentum": "EVA_momentum_median"
           })
)

# Country (categorical): most frequent per company
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

# Averages per company (2014–2024 window)
v = (
    stacked.loc[mask_years, ["Company_name", "Revenue_growth_3_f", "ROE_above_Cost_of_equity"]]
           .groupby("Company_name")[["Revenue_growth_3_f", "ROE_above_Cost_of_equity"]]
           .mean(numeric_only=True)
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
features = ["Revenue_growth_3_f", "ROE_above_Cost_of_equity", "annualised TSR"]  # keep same features

# clean + prepare X (handle inf/NaN)
X = v[features].replace([np.inf, -np.inf], np.nan)
X = SimpleImputer(strategy="median").fit_transform(X)
X = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
v["cluster"] = kmeans.fit_predict(X)

# -------------------- Summaries --------------------
# Safety: replace +/-inf with NaN before aggregation
v_clean = v.replace([np.inf, -np.inf], np.nan)

# Cluster summary (medians) — include PBV, BVE_per_share_1_f, EVA_momentum
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

# keep throwaway vars if you need them
x = 1
y = 2
