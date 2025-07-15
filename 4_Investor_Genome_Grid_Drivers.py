import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
from matplotlib.patches import Rectangle
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib
matplotlib.use('TkAgg')

"""
This is a TSR Driver analysis where you can choose the Genome grid and identify the key drivers
"""

# "UNTENABLE", "CHALLENGED", "TRAPPED", "VIRTUOUS", "BRAVE" "FAMOUS", "FEARLESS", "LEGENDARY"
genome_grid_initial = ["UNTENABLE", "TRAPPED", "BRAVE", "FEARLESS"]
genome_grid_final = ["CHALLENGED", "VIRTUOUS", "FAMOUS", "LEGENDARY"]

# Import data
df_full = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Journeys_summary_Global.csv")
df_slice = df_full.loc[(df_full["Genome_classification_bespoke_beginning"].isin(genome_grid_initial)) & (df_full["Genome_classification_bespoke_end"].isin(genome_grid_final))]

# Random Forest Fit
X_drivers = ["Market_Capitalisation_beginning", "Market_Capitalisation_end",
                                "Company_revenue_avg", "Sector_revenue_avg", "Delta_revenue_avg",
                                "Company_revenue_cagr", "Sector_revenue_cagr", "Delta_revenue_cagr",
                                "Company_leverage", "Sector_leverage", "Delta_leverage",
                                "Company_investment", "Sector_investment", "Delta_investment",
                                "Company_eva_avg", "Sector_eva_avg", "delta_eva_avg",
                                "Company_acquisition_propensity", "Sector_acquisition_propensity", "delta_acquisition_propensity",
                                "Company_capex/revenue", "Sector_capex/revenue", "Delta_capex/revenue",
                                "Company_npat_per_employee", "Sector_npat_per_employee", "Delta_npat_per_employee",
                                "Company_gross_margin", "Sector_gross_margin", "Delta_gross_margin", "Sector_TSR"]


# Drop rows with any NaN values in X or y
df_slice_clean = df_slice.dropna(subset=X_drivers + ["Annualized_TSR_Capiq"])
X = df_slice_clean[X_drivers]
y = df_slice_clean["Annualized_TSR_Capiq"]

# Fit random forest regression model
rf = RandomForestRegressor(n_estimators=500)
rf.fit(X, y)

# Run Shapely values
explainer = shap.Explainer(rf, X)
shap_values = explainer(X)
fig = shap.summary_plot(shap_values, X, max_display=10, show=False)
plt.title("Transformation_drivers")
plt.savefig("Transformation_drivers_shap.png")
plt.show()

# Compute average values of features
# Compute the average values for the features in X_drivers
average_feature_values = X.median()
print(average_feature_values)

# Transformations csv spit out
# df_slice_clean.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Growth_Transformations.csv")
