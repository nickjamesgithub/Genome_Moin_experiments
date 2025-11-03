# --- Imports ---
import re
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from itertools import product

# ====== CONFIG ======
RESPONSE = "TSR_CIQ_no_buybacks"

# -------- Slice controls --------
# RUN_MODE options:
#   "all"        -> run all 8 combos (4 media types × {Developed, Emerging})
#   "single"     -> run a single combo specified by MARKET_GROUP and MEDIA_TYPE
#   "everything" -> run on full dataset (original behavior)
RUN_MODE = "all"   # "all" | "single" | "everything"

# For RUN_MODE="single" (case-insensitive):
MARKET_GROUP = "Developed"     # "Developed" | "Emerging"
MEDIA_TYPE   = "Broadcasting"  # "Broadcasting" | "Publishing" | "Cable and Satellite" | "Advertising"

# Country & media labels (column names provided by you)
COUNTRY_COL = "Country_label"
MEDIA_COL   = "Media_type_label"

# -------- Market classification sets (MSCI/IMF style) --------
DEVELOPED_SET = {
    "Australia","United States","Canada","France","Japan","United Kingdom","Luxembourg",
    "Italy","Switzerland","Germany","Finland","South Korea","Spain","Sweden","Israel",
    "Hong Kong","Singapore"
}
EMERGING_SET = {
    "South Africa","Malaysia","China","Indonesia","Saudi Arabia","India","Mexico",
    "Thailand","Argentina"
}

# (Optional) Normalize a few common variants to reduce unmapped countries
NORMALIZE_COUNTRY = {
    "United States of America": "United States",
    "USA": "United States",
    "U.S.": "United States",
    "Republic of Korea": "South Korea",
    "Korea, Republic of": "South Korea",
    "UK": "United Kingdom",
}

# Thresholds
MIN_ROWS = 10        # guardrail: minimum rows required for fitting a slice
MIN_FEATURE_NONNA = 0.05  # keep features with ≥5% non-missing in the slice

# Base list (we'll prune BVE/NAV and then add engineered features)
FEATURE_COLS = [
    "Profit_margin","ROE","ROA","EVA_Margin","EVA_momentum","EVA_shock","EVA_Profitable_Growth","EVA_Productivity_Gains","Gross_margin",
    "Economic_profit_1_f","EP_growth_2_f","EP_growth_3_f","Revenue_growth_1_f",
    "Revenue_growth_2_f","Revenue_growth_3_f",
    "profit_margin_1_f","profit_margin_growth_2_f","profit_margin_growth_3_f",
    "BVE_per_share_1_f","Dividend_Yield","Buyback_Yield","EBIT_margin"  # ,"Capex_to_Revenue"
]

FEATURE_LABELS = {
    "Profit_margin":"Profit margin","ROE":"ROE",
    "ROA":"ROA","EVA_Margin":"EVA Margin","EVA_momentum":"EVA momentum","EVA_shock":"EVA shock",
    "EVA_Profitable_Growth":"EVA Profitable Growth","EVA_Productivity_Gains":"EVA Productivity Gains",
    "Gross_margin":"Gross margin",
    "Economic_profit_1_f":"EP (1-year)","EP_growth_2_f":"EP growth (2-year)","EP_growth_3_f":"EP growth (3-year)",
    "Revenue_growth_1_f":"Revenue growth (1-year)","Revenue_growth_2_f":"Revenue growth (2-year)","Revenue_growth_3_f":"Revenue growth (3-year)",
    "profit_margin_1_f":"Profit margin (1-year)","profit_margin_growth_2_f":"Profit margin growth (2-year)","profit_margin_growth_3_f":"Profit margin growth (3-year)",
    "Dividend_Yield":"Dividend Yield","Buyback_Yield":"Buyback Yield",
    # Engineered labels:
    "ROFE":"ROFE",
    "ROFE - WACC":"ROFE - WACC",
    "ROFE_1_f":"ROFE (1-year)",
    "ROFE_growth_2_f":"ROFE (2-year)",
    "ROFE_growth_3_f":"ROFE (3-year)",
    "EBIT_margin":"EBIT margin",
    "EBITDA_margin": "EBITDA margin",
    "Capex_to_Revenue":"Capex / Revenue",
    "ROIC":"ROIC",
    "Payout_ratio":"Payout ratio",
    "Asset_turnover":"Asset turnover"
}

# --- Load data ---
df = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\Nine\media_data_global_mapped.csv")

# Normalize country labels (optional)
df[COUNTRY_COL] = df[COUNTRY_COL].replace(NORMALIZE_COUNTRY)

# -------- Add Market_Group column from Country_label --------
def _market_group(country: str) -> str:
    if pd.isna(country):
        return np.nan
    c = str(country)
    if c in DEVELOPED_SET:
        return "Developed"
    if c in EMERGING_SET:
        return "Emerging"
    return np.nan  # exclude unknowns from sliced runs

df["Market_Group"] = df[COUNTRY_COL].map(_market_group)

# --- Helper to coerce numeric & safe-divide ---
def _num(s):
    return pd.to_numeric(s, errors="coerce")

def _safe_div(numer, denom):
    d = _num(denom).replace(0, np.nan)
    return _num(numer) / d

# --- Engineer ROFE + ROFE - WACC ---
for col in ["EBIT","Funds_employed","WACC_Damodaran"]:
    if col not in df.columns:
        raise KeyError(f"Missing required column for engineering '{col}'")
ebit = _num(df["EBIT"])
fe   = _num(df["Funds_employed"]).replace(0, np.nan)
wacc = _num(df["WACC_Damodaran"])  # assumes same unit as ROFE
df["ROFE"] = ebit / fe
df["ROFE - WACC"] = df["ROFE"] - wacc

# --- Engineer EBIT_margin, EBITDA_margin, Capex_to_Revenue ---
for col in ["Revenue","CAPEX"]:
    if col not in df.columns:
        raise KeyError(f"Missing required column for engineering '{col}'")
rev   = _num(df["Revenue"]).replace(0, np.nan)
capex = _num(df["CAPEX"])
df["EBIT_margin"]      = ebit / rev
if "EBITDA" in df.columns:
    df["EBITDA_margin"] = _num(df["EBITDA"]) / rev
else:
    df["EBITDA_margin"] = np.nan
df["Capex_to_Revenue"] = capex / rev

# --- Engineer Gross_margin = Gross_profit / Revenue ---
if "Gross_profit" in df.columns:
    df["Gross_margin"] = _num(df["Gross_profit"]) / rev
elif "Gross_margin" in df.columns:
    df["Gross_margin"] = _num(df["Gross_margin"])
else:
    raise KeyError("Missing 'Gross_profit' (or precomputed 'Gross_margin') needed to engineer Gross_margin")

# --- Engineer ROIC = NOPAT / Funds_employed ---
if "NOPAT" in df.columns:
    df["ROIC"] = _safe_div(df["NOPAT"], df["Funds_employed"])
else:
    raise KeyError("Missing 'NOPAT' needed to engineer ROIC")

# --- Engineer Payout_ratio = DPS / Diluted_EPS ---
if ("DPS" in df.columns) and ("Diluted_EPS" in df.columns):
    df["Payout_ratio"] = _safe_div(df["DPS"], df["Diluted_EPS"])
else:
    missing = [c for c in ["DPS","Diluted_EPS"] if c not in df.columns]
    raise KeyError(f"Missing {missing} needed to engineer Payout_ratio")

# --- Engineer Asset_turnover = Revenue / Total_assets ---
if "Total_assets" in df.columns:
    df["Asset_turnover"] = _safe_div(df["Revenue"], df["Total_assets"])
else:
    raise KeyError("Missing 'Total_assets' needed to engineer Asset_turnover")

# --- Detect entity/time for per-entity rolling (fallback to whole-frame order) ---
ENTITY_CANDIDATES = ["Ticker","Company","ISIN","GVKEY","permno","PERMNO","CUSIP","Entity","Firm","Sid","SID"]
TIME_CANDIDATES   = ["Year","year","fyear","FY","Period","period","Date","date"]

entity_col = next((c for c in ENTITY_CANDIDATES if c in df.columns), None)
time_col   = next((c for c in TIME_CANDIDATES if c in df.columns), None)

if time_col:
    df = df.sort_values([entity_col, time_col] if entity_col else [time_col])

# --- ROFE rolling means: 1y / 2y / 3y (per entity when available) ---
def _rolling_mean(s: pd.Series, window: int) -> pd.Series:
    if entity_col:
        return s.groupby(df[entity_col], dropna=False).transform(lambda x: x.rolling(window=window, min_periods=1).mean())
    else:
        return s.rolling(window=window, min_periods=1).mean()

df["ROFE_1_f"]        = _rolling_mean(df["ROFE"], 1)
df["ROFE_growth_2_f"] = _rolling_mean(df["ROFE"], 2)
df["ROFE_growth_3_f"] = _rolling_mean(df["ROFE"], 3)

# --- Remove ALL BVE* and NAV* features (case-insensitive) ---
drop_pattern = re.compile(r"(BVE|NAV)", flags=re.IGNORECASE)
FEATURE_COLS = [c for c in FEATURE_COLS if not drop_pattern.search(c)]
to_drop_cols = [c for c in df.columns if drop_pattern.search(c)]
df = df.drop(columns=to_drop_cols, errors="ignore")

# Add engineered features to model inputs (ensure uniqueness)
FEATURE_COLS += [
    "ROFE","ROFE - WACC","ROFE_1_f","ROFE_growth_2_f","ROFE_growth_3_f",
    "EBIT_margin","EBITDA_margin","Capex_to_Revenue",
    "ROIC","Payout_ratio","Asset_turnover","Gross_margin"
]
FEATURE_COLS = list(dict.fromkeys(FEATURE_COLS))  # de-dupe while preserving order

# --- Safety: ensure required columns exist ---
need = FEATURE_COLS + [RESPONSE]
missing = [c for c in need if c not in df.columns]
if missing:
    raise KeyError(f"Missing columns in dataframe: {missing}")

# --- Surface unmapped countries once (for awareness) ---
unmapped = df.loc[df["Market_Group"].isna(), COUNTRY_COL].value_counts()
if len(unmapped):
    print("Unmapped countries (won't appear in 'all'/'single' slice runs):")
    print(unmapped.head(20))

# -------- Helper to fit & report for any slice (with imputation) --------
def _fit_and_report(df_slice: pd.DataFrame, tag: str):
    # Diagnostics (before filtering)
    n0 = len(df_slice)
    y_all = pd.to_numeric(df_slice[RESPONSE], errors="coerce").replace([np.inf, -np.inf], np.nan)
    X_all = df_slice[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    print(f"[{tag}] Rows (raw slice): {n0:,}")
    print(f"[{tag}] Rows with valid RESPONSE: {y_all.notna().sum():,}")

    na_rate = X_all.isna().mean().sort_values(ascending=False)
    print(f"[{tag}] Feature NA rates (top 10):")
    print((na_rate.head(10)*100).round(1).astype(str) + "%")
    if "EBITDA_margin" in X_all.columns:
        print(f"[{tag}] EBITDA_margin all-NaN? {X_all['EBITDA_margin'].isna().all()}")

    # 1) Filter only on y being finite
    mask_y = np.isfinite(y_all)
    df_slice = df_slice.loc[mask_y]
    y = y_all.loc[mask_y].astype(np.float64)

    # 2) Per-slice feature sparsity filter (keep cols with >= MIN_FEATURE_NONNA data)
    non_empty = []
    for c in FEATURE_COLS:
        if c in df_slice.columns:
            frac_non_nan = pd.to_numeric(df_slice[c], errors="coerce").replace([np.inf, -np.inf], np.nan).notna().mean()
            if frac_non_nan >= MIN_FEATURE_NONNA:
                non_empty.append(c)
    if not non_empty:
        print(f"[{tag}] No usable features after sparsity filter. Skipping.")
        return

    # 3) Build X (coerce numerics, cap absurd magnitudes, leave NaNs for imputer)
    X = df_slice[non_empty].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = X.where(X.abs() < 1e30)
    X = X.astype(np.float64)

    # Guardrail: require some rows
    if len(X) < MIN_ROWS:
        print(f"[{tag}] Insufficient rows after cleaning: {len(X)}. Skipping.")
        return

    print(f"[{tag}] Rows (clean after y filter): {len(X):,} | Features kept: {len(non_empty)}")

    # 4) Pipeline: impute -> RF
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)),
    ])
    pipe.fit(X, y)

    rf = pipe.named_steps["rf"]

    # 5) Feature importances (pretty labels)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    importances_pretty = importances.rename(index=lambda n: FEATURE_LABELS.get(n, n))

    print(f"\n[{tag}] Top feature importances:")
    print(importances_pretty.head(25))

    # 6) Plot: Top-25 Importances (with labels)
    top = importances_pretty.head(25)[::-1]
    plt.figure(figsize=(10, 6), dpi=150)
    ax = top.plot(kind="barh")
    for bar, val in zip(ax.patches, top.values):
        ax.text(bar.get_width()*1.01, bar.get_y()+bar.get_height()/2, f"{val:.3f}", va="center")
    plt.title(f"TSR Drivers — Media (ML model) [{tag}]")
    plt.xlabel("Importance"); plt.ylabel("Feature"); plt.tight_layout()
    fname_imp = f"Media_TSR_Drivers_RF_{tag.replace(' ','_')}.png"
    plt.savefig(fname_imp, dpi=150)
    plt.show()

    # 7) SHAP (on imputed X)
    try:
        import shap
        # Recreate the imputed X to pass to SHAP (same imputer as in pipeline)
        X_imputed = pipe.named_steps["imputer"].fit_transform(X)
        X_imputed = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)

        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_imputed)
        X_pretty = X_imputed.rename(columns=lambda c: FEATURE_LABELS.get(c, c))

        plt.figure(figsize=(10, 7), dpi=150)
        shap.summary_plot(shap_values, X_pretty, plot_type="dot", max_display=25, show=False)
        plt.title(f"TSR Drivers — Media (SHAP Beeswarm) [{tag}]")
        plt.tight_layout()
        fname_bee = f"Media_TSR_SHAP_Beeswarm_{tag.replace(' ','_')}.png"
        plt.savefig(fname_bee, dpi=150, bbox_inches="tight")
        plt.show()

        plt.figure(figsize=(10, 7), dpi=150)
        shap.summary_plot(shap_values, X_pretty, plot_type="violin", max_display=25, show=False)
        plt.title(f"TSR Drivers — Media (SHAP Violin) [{tag}]")
        plt.tight_layout()
        fname_vio = f"Media_TSR_SHAP_Violin_{tag.replace(' ','_')}.png"
        plt.savefig(fname_vio, dpi=150, bbox_inches="tight")
        plt.show()
    except ImportError:
        raise SystemExit("SHAP is not installed. Please run: pip install shap --upgrade")

# -------- Driver for the 3 run modes --------
media_options = ["Broadcasting", "Publishing", "Cable and Satellite", "Advertising"]

if RUN_MODE.lower() == "everything":
    _fit_and_report(df, tag="ALL")

elif RUN_MODE.lower() == "single":
    mg = str(MARKET_GROUP).strip().title()
    mt = str(MEDIA_TYPE).strip()
    if mg not in {"Developed","Emerging"}:
        raise ValueError("MARKET_GROUP must be 'Developed' or 'Emerging'")
    if mt not in media_options:
        raise ValueError(f"MEDIA_TYPE must be one of {media_options}")

    dfx = df[(df["Market_Group"] == mg) & (df[MEDIA_COL] == mt)]
    tag = f"{mg} × {mt}"
    _fit_and_report(dfx, tag=tag)

elif RUN_MODE.lower() == "all":
    # Cartesian product of 2 market groups × 4 media types
    for mg, mt in product(["Developed","Emerging"], media_options):
        dfx = df[(df["Market_Group"] == mg) & (df[MEDIA_COL] == mt)]
        tag = f"{mg} × {mt}"
        _fit_and_report(dfx, tag=tag)
else:
    raise ValueError("RUN_MODE must be 'all', 'single', or 'everything'")
