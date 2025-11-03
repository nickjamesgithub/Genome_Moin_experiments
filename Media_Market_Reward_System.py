# --- IMPORTS ---
import pandas as pd
import matplotlib.pyplot as plt

# --- LOAD DATA ---
file_path = r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\Nine\media_data_global_mapped.csv"
data = pd.read_csv(file_path)

# --- CONFIG ---
TSR_COL = "TSR_CIQ_no_buybacks"
GROWTH_COL = "Revenue_growth_3_f"   # use 3-year growth
EVA_COL = "EVA_ratio_bespoke"
COUNTRY_COL = "Country_label"
SCALE_TSR_BY_100 = True  # Set False if TSR is already in %

DEVELOPED_SET = {
    "Australia","United States","Canada","France","Japan","United Kingdom","Luxembourg",
    "Italy","Switzerland","Germany","Finland","South Korea","Spain","Sweden","Israel",
    "Hong Kong","Singapore"
}
EMERGING_SET = {
    "South Africa","Malaysia","China","Indonesia","Saudi Arabia","India","Mexico",
    "Thailand","Argentina"
}

# --- HELPERS ---
def classify_market(country: str) -> str:
    if country in DEVELOPED_SET:
        return "Developed"
    if country in EMERGING_SET:
        return "Emerging"
    return "Unclassified"

# ====== BIN LOGIC: <0, 0–5, 5–10, 10–15, 15–20, >20 ======
def make_bins(series, is_percent_like=False):
    if is_percent_like:
        edges = [-float("inf"), 0.0, 5.0, 10.0, 15.0, 20.0, float("inf")]
    else:
        edges = [-float("inf"), 0.0, 0.05, 0.10, 0.15, 0.20, float("inf")]
    labels = ["<0%", "0–5%", "5–10%", "10–15%", "15–20%", ">20%"]
    return pd.cut(series, bins=edges, labels=labels, include_lowest=True, right=False)

# --- PREP DATA ---
df = data.copy()
df["Market_Group"] = df[COUNTRY_COL].map(classify_market)
df = df[df["Market_Group"].isin(["Developed","Emerging"])].copy()

# Convert numeric columns
for col in [TSR_COL, GROWTH_COL, EVA_COL]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

def is_percent_like(col):
    s = df[col].dropna()
    if s.empty:
        return False
    return s.abs().median() > 1.5  # heuristic for % vs decimal

growth_is_pct = is_percent_like(GROWTH_COL)
eva_is_pct = is_percent_like(EVA_COL)

df["Growth_Bin"] = make_bins(df[GROWTH_COL], is_percent_like=growth_is_pct)
df["EVA_Bin"] = make_bins(df[EVA_COL], is_percent_like=eva_is_pct)

# Scale TSR if needed
df["_TSR_plot"] = df[TSR_COL] * 100.0 if SCALE_TSR_BY_100 else df[TSR_COL]

# --- AGGREGATE ---
def ensure_all_bins(agg, bin_col):
    bins = ["<0%", "0–5%", "5–10%", "10–15%", "15–20%", ">20%"]
    idx = pd.MultiIndex.from_product([["Developed","Emerging"], bins], names=["Market_Group", bin_col])
    return (agg.set_index(["Market_Group", bin_col])
               .reindex(idx)
               .reset_index())

growth_agg = (
    df.groupby(["Market_Group", "Growth_Bin"], dropna=False)["_TSR_plot"]
      .median()
      .rename("Median_TSR")
      .reset_index()
)
growth_agg = ensure_all_bins(growth_agg, "Growth_Bin")

eva_agg = (
    df.groupby(["Market_Group", "EVA_Bin"], dropna=False)["_TSR_plot"]
      .median()
      .rename("Median_TSR")
      .reset_index()
)
eva_agg = ensure_all_bins(eva_agg, "EVA_Bin")

# --- PLOT ---
fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
fig.suptitle("Median TSR by Market & Bin", fontsize=15)

# Top row: Revenue Growth
for i, market in enumerate(["Developed","Emerging"]):
    ax = axes[0, i]
    sub = growth_agg[growth_agg["Market_Group"] == market]
    ax.bar(sub["Growth_Bin"].astype(str), sub["Median_TSR"], color="#4a90e2")
    ax.set_title(f"{market} — Revenue Growth bins")
    ax.set_xlabel("Revenue Growth bin")
    ax.set_ylabel("Median TSR (%)" if SCALE_TSR_BY_100 else "Median TSR")
    ax.axhline(0, linewidth=1, color="gray")
    for x, y in zip(sub["Growth_Bin"].astype(str), sub["Median_TSR"]):
        if pd.notna(y):
            ax.text(x, y, f"{y:.1f}", ha="center", va="bottom", fontsize=9)

# Bottom row: EVA ratio
for i, market in enumerate(["Developed","Emerging"]):
    ax = axes[1, i]
    sub = eva_agg[eva_agg["Market_Group"] == market]
    ax.bar(sub["EVA_Bin"].astype(str), sub["Median_TSR"], color="#50c878")
    ax.set_title(f"{market} — EVA ratio bins")
    ax.set_xlabel("EVA ratio bin")
    ax.set_ylabel("Median TSR (%)" if SCALE_TSR_BY_100 else "Median TSR")
    ax.axhline(0, linewidth=1, color="gray")
    for x, y in zip(sub["EVA_Bin"].astype(str), sub["Median_TSR"]):
        if pd.notna(y):
            ax.text(x, y, f"{y:.1f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
