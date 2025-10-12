# --- Imports ---
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# ====== CONFIG ======
RESPONSE = "TSR_CIQ_no_buybacks"

FEATURE_COLS = [
    "Profit_margin","ROE","ROE_above_Cost_of_equity","ROA","BVE_per_share","CROTE_TE",
    "EVA_Margin","EVA_momentum","EVA_shock","EVA_Profitable_Growth","EVA_Productivity_Gains",
    "Economic_profit_1_f","EP_growth_2_f","EP_growth_3_f","Revenue_growth_1_f",
    "Revenue_growth_2_f","Revenue_growth_3_f","NAV_1_f","NAV_growth_2_f","NAV_growth_3_f",
    "profit_margin_1_f","profit_margin_growth_2_f","profit_margin_growth_3_f",
    "BVE_per_share_1_f" ,"Dividend_Yield","Buyback_Yield"
]
FEATURE_LABELS = {
    "Profit_margin":"Profit margin","ROE":"ROE","ROE_above_Cost_of_equity":"ROE - Cost of equity",
    "ROA":"ROA","BVE_per_share":"BVE per share","CROTE_TE":"CROTE - Cost of equity",
    "EVA_Margin":"EVA Margin","EVA_momentum":"EVA momentum","EVA_shock":"EVA shock",
    "EVA_Profitable_Growth":"EVA Profitable Growth","EVA_Productivity_Gains":"EVA Productivity Gains",
    "Economic_profit_1_f":"EP (1-year)","EP_growth_2_f":"EP growth (2-year)","EP_growth_3_f":"EP growth (3-year)",
    "Revenue_growth_1_f":"Revenue growth (1-year)","Revenue_growth_2_f":"Revenue growth (2-year)",
    "Revenue_growth_3_f":"Revenue growth (3-year)","NAV_1_f":"NAV growth (1-year)",
    "NAV_growth_2_f":"NAV growth (2-year)","NAV_growth_3_f":"NAV growth (3-year)",
    "profit_margin_1_f":"Profit margin (1-year)","profit_margin_growth_2_f":"Profit margin growth (2-year)",
    "profit_margin_growth_3_f":"Profit margin growth (3-year)","BVE_per_share_1_f":"BVE per share growth (1-year)",
    "Dividend_Yield":"Dividend Yield","Buyback_Yield":"Buyback Yield"
}

# --- Load & filter to Healthcare ---
df = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\grocery_data.csv")

# --- Safety: ensure required columns exist ---
need = FEATURE_COLS + [RESPONSE]
missing = [c for c in need if c not in df.columns]
if missing:
    raise KeyError(f"Missing columns in healthcare dataframe: {missing}")

# --- Build X, y with numeric coercion & strict finite filtering ---
y = pd.to_numeric(df[RESPONSE], errors="coerce").replace([np.inf, -np.inf], np.nan)
X = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
X = X.where(X.abs() < 1e30)  # guard against absurd magnitudes

mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
X, y = X.loc[mask].astype(np.float64), y.loc[mask].astype(np.float64)

print(f"Healthcare rows (clean): {len(X):,} | Features: {X.shape[1]}")
print(f"TSR range: [{y.min():.4g}, {y.max():.4g}]")

# --- Fit Random Forest ---
rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
rf.fit(X, y)

# --- Feature importances (pretty labels) ---
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
importances_pretty = importances.rename(index=lambda n: FEATURE_LABELS.get(n, n))
print("\nTop feature importances:")
print(importances_pretty.head(25))

# --- Plot: Top-25 Importances (with labels) ---
top = importances_pretty.head(25)[::-1]  # reverse for bottom-to-top barh
plt.figure(figsize=(10, 6), dpi=150)
ax = top.plot(kind="barh")
for bar, val in zip(ax.patches, top.values):
    ax.text(bar.get_width()*1.01, bar.get_y()+bar.get_height()/2, f"{val:.3f}", va="center")
plt.title("TSR Drivers — Healthcare (Random Forest)")
plt.xlabel("Importance"); plt.ylabel("Feature"); plt.tight_layout()
plt.savefig("Grocery_TSR_Drivers_RF.png", dpi=150);
plt.show()

# ================== SHAP BEESWARM ==================
try:
    import shap
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)
    X_pretty = X.rename(columns=lambda c: FEATURE_LABELS.get(c, c))

    # Beeswarm (top-25)
    plt.figure(figsize=(10, 7), dpi=150)
    shap.summary_plot(shap_values, X_pretty, plot_type="dot", max_display=25, show=False)
    plt.title("TSR Drivers — Grocery (SHAP Beeswarm)")
    plt.tight_layout();
    plt.savefig("Grocery_TSR_SHAP_Beeswarm.png", dpi=150, bbox_inches="tight"); plt.show()
    plt.show()

    # Violin-style
    plt.figure(figsize=(10, 7), dpi=150)
    shap.summary_plot(shap_values, X_pretty, plot_type="violin", max_display=25, show=False)
    plt.title("TSR Drivers — Grocery (SHAP Violin)")
    plt.tight_layout();
    plt.savefig("Grocery_TSR_SHAP_Violin.png", dpi=150, bbox_inches="tight");
    plt.show()
except ImportError:
    raise SystemExit("SHAP is not installed. Please run: pip install shap --upgrade")
