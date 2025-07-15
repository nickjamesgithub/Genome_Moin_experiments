import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ------------------ Parameters ------------------ #
start_year = 2015
end_year = 2024
n_years = end_year - start_year + 1

selected_countries = [
    "Australia", "Denmark", "Hong_Kong", "India", "Malaysia", "Netherlands",
    "Singapore", "Sweden", "Switzerland", "Thailand", "USA", "United_Kingdom"
]

# ------------------ Load and Combine Data ------------------ #
global_path = r"C:\Users\60848\OneDrive - Bain\Desktop\Genome_code_250605\Genome-pipeline-code\Genome-pipeline-code\global_platform_data\Global_data.csv"
bespoke_path = r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\bespoke_data.csv"

df_global = pd.read_csv(global_path)
df_bespoke = pd.read_csv(bespoke_path)

df_all = pd.concat([df_global, df_bespoke], ignore_index=True)

# ------------------ Filter Banking Data ------------------ #
df_filtered = df_all[
    (df_all["Sector"] == "Banking") &
    (df_all["Year"].between(start_year, end_year)) &
    (df_all["Country"].isin(selected_countries + ["Bespoke"]))
].copy()

# Convert to numeric
df_filtered["PBV"] = pd.to_numeric(df_filtered["PBV"], errors="coerce")
df_filtered["ROE"] = pd.to_numeric(df_filtered["ROE"], errors="coerce")

# ------------------ Identify Companies with Full Coverage ------------------ #
valid_years = (
    df_filtered.dropna(subset=["PBV", "ROE"])
    .groupby("Company_name")["Year"]
    .nunique()
)

complete_companies = valid_years[valid_years == n_years].index
df_complete = df_filtered[df_filtered["Company_name"].isin(complete_companies)].copy()

# ------------------ Compute Averages and Keep One Ticker ------------------ #
tickers_df = df_complete.groupby("Company_name")["Ticker_full"].first().reset_index()

df_avg = (
    df_complete.groupby(["Company_name", "Country"])
    .agg(Avg_PBV=("PBV", "mean"), Avg_ROE=("ROE", "mean"))
    .reset_index()
)

df_avg = pd.merge(df_avg, tickers_df, on="Company_name", how="left")

# ------------------ Clean and Filter ------------------ #
df_avg = df_avg[
    (df_avg["Avg_PBV"] > 0) &
    (df_avg["Avg_ROE"] >= 0) &
    (df_avg["Avg_ROE"] <= 0.25)
].drop_duplicates(subset=["Company_name"])

# ------------------ Regression ------------------ #
X = df_avg["Avg_ROE"]
y = df_avg["Avg_PBV"]
X_const = sm.add_constant(X)

model = sm.OLS(y, X_const).fit()
df_avg["PBV_Predicted"] = model.predict(X_const)
df_avg["Position_vs_Line"] = np.where(df_avg["Avg_PBV"] > df_avg["PBV_Predicted"], "Above", "Below")

# ------------------ Plot Regression with Ticker Labels ------------------ #
plt.figure(figsize=(8, 6))

for label, color in zip(["Above", "Below"], ["green", "orange"]):
    subset = df_avg[df_avg["Position_vs_Line"] == label]
    plt.scatter(subset["Avg_ROE"], subset["Avg_PBV"], alpha=0.7, label=label, color=color)

    for _, row in subset.iterrows():
        label_text = row["Ticker_full"].split(":")[-1] if pd.notnull(row["Ticker_full"]) else ""
        plt.text(row["Avg_ROE"], row["Avg_PBV"], label_text, fontsize=7, alpha=0.7)

# Plot regression line
x_vals = np.linspace(X.min(), X.max(), 100)
y_vals = model.params[0] + model.params[1] * x_vals
plt.plot(x_vals, y_vals, color="red", label="Regression Line")

plt.xlabel("Avg ROE (2015–2024)")
plt.ylabel("Avg PBV (2015–2024)")
plt.title("Avg ROE vs Avg PBV — Selected Countries + Bespoke (2015–2024)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("WBC_Regression_banking.png")
plt.show()

# ------------------ Regression Summary ------------------ #
print("\n📊 Regression Summary (PBV ~ ROE):")
print(model.summary())
