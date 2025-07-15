import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
from matplotlib.patches import Rectangle
from sklearn.ensemble import RandomForestRegressor
import shap
from Utilities import compute_percentiles
import matplotlib
from Utilities import generate_market_cap_class, waterfall_value_plot, geometric_return
matplotlib.use('TkAgg')

"""
This is a script to compute the waterfall for an entire sector, or collection of sectors
"""

# Import data
data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\Global_data.csv")
# Define countries and sectors to include
countries_to_include = ['USA', 'AUS', 'INDIA', 'JAPAN', 'EURO', 'UK'] # 'USA', 'AUS', 'INDIA', 'JAPAN', 'EURO', 'UK'
sectors_to_include = ["Telecommunications"]
plot_label = "Global_Telco"

# Filter data based on countries and sectors
df_merge = data.loc[(data['Country'].isin(countries_to_include)) & (data['Sector'].isin(sectors_to_include))]

# Required tickers
tickers_ = np.unique(df_merge["Ticker"].values)
df_merge_clean = df_merge.dropna()

# Random Forest Fit
X_drivers = ["Revenue_growth_1_f", "NAV_1_f", "Economic_profit_1_f", "EP/FE_1_f", "profit_margin_1_f",
"ROTE", "ROTE_above_Cost_of_equity", "roa_1_f", "Revenue_growth_2_f", "NAV_growth_2_f", "EP_growth_2_f", "EP/FE_growth_2_f",
"profit_margin_growth_2_f", "roa_growth_2_f", "Revenue_growth_3_f", "NAV_growth_3_f",
"EP_growth_3_f", "EP/FE_growth_3_f", "profit_margin_growth_3_f", "roa_growth_3_f", "CROTE_TE", "BVE_per_share_1_f", "BVE_per_share_growth_2_f", "BVE_per_share_growth_3_f", "EVA_momentum",  "EVA_shock",
             "EVA_Profitable_Growth", "EVA_Productivity_Gains", "EVA_ratio_bespoke"]

# Fill na values with 0
X = df_merge_clean[X_drivers]
# Clean y
y = df_merge_clean["TSR"]

# Fit random forest regression model
rf = RandomForestRegressor(n_estimators=500)
rf.fit(X, y)

# Run Shapely values
explainer = shap.Explainer(rf, X)
shap_values = explainer(X)
fig = shap.summary_plot(shap_values, X, max_display=10, show=False)
plt.savefig("_Capiq" + plot_label + "_" + "_Shap.png")
plt.show()
