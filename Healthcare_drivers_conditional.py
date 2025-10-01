# --- Imports ---
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# --- Load data ---
df = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Genome_code_250605\Genome-pipeline-code\Genome-pipeline-code\global_platform_data\Global_data.csv")
df_hc = df[df["Sector"].isin(["Healthcare","Health Care"])].copy()

# --- Read clusters + entity match on Company_name (robust normalization) ---
clusters_df = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\EBO\Healthcare_market_data\healthcare_clusters.csv")  # columns: company_name, cluster

# ====== CONFIG ======
RESPONSE = "TSR_CIQ_no_buybacks"

FEATURE_COLS = [
    "Profit_margin","ROE","ROE_above_Cost_of_equity","ROA","BVE_per_share","CROTE_TE",
    "EVA_Margin","EVA_momentum","EVA_shock","EVA_Profitable_Growth","EVA_Productivity_Gains",
    "Economic_profit_1_f","EP_growth_2_f","EP_growth_3_f","Revenue_growth_1_f",
    "Revenue_growth_2_f","Revenue_growth_3_f","NAV_1_f","NAV_growth_2_f","NAV_growth_3_f",
    "profit_margin_1_f","profit_margin_growth_2_f","profit_margin_growth_3_f",
    "BVE_per_share_1_f","Dividend_Yield","Buyback_Yield"
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

# ====== SELECT FILTERS ======
# Leave as None or [] to include ALL
SELECT_COUNTRIES = ['China', 'India','Malaysia','Saudi_Arabia','Thailand','Hong_Kong']

# e.g., ['Australia', 'Belgium', 'China', 'Denmark', 'France', 'Germany',
#        'Hong_Kong', 'India', 'Italy', 'Japan', 'Malaysia', 'Netherlands',
#        'Saudi_Arabia', 'South_Korea', 'Sweden', 'Switzerland', 'Thailand',
#        'United_Kingdom', 'USA']

### DEVELOPED ###
# ['Australia', 'Belgium','Denmark', 'France', 'Germany','Hong_Kong', 'Italy',
# 'Japan','Netherlands','South_Korea', 'Sweden', 'Switzerland','United_Kingdom', 'USA']

### EMERGING ###
# ['China', 'India','Malaysia','Saudi_Arabia','Thailand','Hong_Kong']

SELECT_CLUSTERS  = ['Healthcare Payers & Insurance']
# e.g., ['Pharmaceuticals & Biotechnology', 'Medical Devices & Equipment',
#        'Healthcare Providers & Facilities','Healthcare Payers & Insurance',
#        'Healthcare Support Services & Others']

PER_COMBO = False  # True -> fit per (Country × Cluster); False -> single fit on combined filter
MIN_ROWS = 50      # guardrail for tiny samples

# --- Normalize & merge cluster labels ---
def _norm(s):
    return (s.astype(str).str.normalize("NFKD").str.encode("ascii","ignore").str.decode("ascii")
            .str.lower().str.strip().str.replace(r"[\u200b\u200c\u200d]+","",regex=True)
            .str.replace(r"\s+"," ",regex=True))

df_hc["__key"] = _norm(df_hc["Company_name"])
clusters_df["__key"] = _norm(clusters_df["company_name"])
df_hc = df_hc.merge(clusters_df[["__key","cluster"]], on="__key", how="left").drop(columns="__key")
df_hc["cluster"] = df_hc["cluster"].fillna("Unclassified")

# --- Checks ---
need = FEATURE_COLS + [RESPONSE,"Company_name","Country","cluster"]
miss = [c for c in need if c not in df_hc.columns]
if miss: raise KeyError(f"Missing columns: {miss}")
print(f"Healthcare rows: {len(df_hc):,}; unmatched cluster labels: {(df_hc['cluster']=='Unclassified').sum():,}")

# --- Helper: clean & fit one group, save plots (TOP-10 with labels) ---
def fit_group(g: pd.DataFrame, tag: str, min_rows: int = MIN_ROWS):
    y = pd.to_numeric(g[RESPONSE], errors="coerce").replace([np.inf,-np.inf], np.nan)
    X = g[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf], np.nan)
    X = X.where(X.abs() < 1e30)
    mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X, y = X.loc[mask].astype(np.float64), y.loc[mask].astype(np.float64)
    if len(X) < min_rows:
        print(f"[SKIP] {tag}: rows={len(X)} < {min_rows}")
        return None

    rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1).fit(X, y)
    imps = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    imps_pretty = imps.rename(index=lambda n: FEATURE_LABELS.get(n,n))

    # --- Plot TOP-10 feature importances with value labels ---
    top = imps_pretty.head(min(10, len(imps_pretty)))[::-1]  # reverse for barh bottom-to-top
    plt.figure(figsize=(9,6), dpi=150)
    ax = top.plot(kind="barh")
    for bar, val in zip(ax.patches, top.values):
        ax.text(bar.get_width()*1.01, bar.get_y()+bar.get_height()/2, f"{val:.3f}", va="center")
    plt.title(f"TSR Drivers — {tag}")
    plt.xlabel("Importance"); plt.ylabel("Feature"); plt.tight_layout()
    fn_imp = f"TSR_Drivers_RF_TOP10_{tag}.png".replace(" ", "_")
    plt.savefig(fn_imp, dpi=150);
    plt.show(); print(f"[OK] {tag}: {fn_imp}")

    # --- SHAP (optional) ---
    try:
        import shap
        explainer = shap.TreeExplainer(rf); sv = explainer.shap_values(X)
        Xp = X.rename(columns=lambda c: FEATURE_LABELS.get(c,c))

        plt.figure(figsize=(10,7), dpi=150)
        shap.summary_plot(sv, Xp, plot_type="dot", max_display=10, show=False)  # align top-10
        fn_sw = f"TSR_SHAP_Beeswarm_TOP10_{tag}.png".replace(" ", "_")
        plt.title(f"TSR Drivers — {tag} (SHAP)"); plt.tight_layout()
        plt.savefig(fn_sw, dpi=150, bbox_inches="tight");
        plt.show()

        plt.figure(figsize=(10,7), dpi=150)
        shap.summary_plot(sv, Xp, plot_type="violin", max_display=10, show=False)
        fn_sv = f"TSR_SHAP_Violin_TOP10_{tag}.png".replace(" ", "_")
        plt.title(f"TSR Drivers — {tag} (SHAP)"); plt.tight_layout()
        plt.savefig(fn_sv, dpi=150, bbox_inches="tight");
        plt.show()
        print(f"[OK] {tag}: {fn_sw}, {fn_sv}")
    except Exception as e:
        print(f"[WARN] {tag}: SHAP skipped ({e})")

    return rf, imps

# ====== FILTER & RUN ======
def _pick(all_vals, picks):
    if picks is None or len(picks)==0: return list(pd.Series(all_vals).dropna().unique())
    return picks

countries_sel = _pick(df_hc["Country"], SELECT_COUNTRIES)
clusters_sel  = _pick(df_hc["cluster"],  SELECT_CLUSTERS)

if not PER_COMBO:
    sub = df_hc[df_hc["Country"].isin(countries_sel) & df_hc["cluster"].isin(clusters_sel)].copy()
    tag = f"{'__'.join(sorted(set(countries_sel))[:3])}_AND_{'__'.join(sorted(set(clusters_sel))[:3])}"
    tag = tag.replace("/","-")[:120]
    fit_group(sub, tag)
else:
    for (cty, clus), g in (df_hc[df_hc["Country"].isin(countries_sel) & df_hc["cluster"].isin(clusters_sel)]
                           .groupby(["Country","cluster"], dropna=False)):
        fit_group(g, f"{cty}__{clus}".replace("/","-"))

print("Done.")
