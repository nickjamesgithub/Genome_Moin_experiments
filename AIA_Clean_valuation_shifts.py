# --- Imports ---
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # set backend BEFORE importing pyplot
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# --- Load data ---
CSV_PATH = r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\insurance_data_clean_250925.csv"
df = pd.read_csv(CSV_PATH)

# --------------------------
# 1) Cleaning helpers
# --------------------------
def clean_col(df, col):
    """Make a column numeric, drop infs, drop rows where it's NA."""
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    return df.dropna(subset=[col])

def ensure_sector(df, sector_col="Sector", default_value=None):
    df = df.copy()
    if sector_col not in df.columns and default_value is not None:
        df[sector_col] = default_value
    return df

# Clean PE and ensure Sector exists
df = clean_col(df, "PE")
df = ensure_sector(df, "Sector", default_value="Insurance")

# --------------------------
# 2) Label Growth/Value/Middle (by PBV, per Year x Sector)
# --------------------------
def label_growth_value(
    df, value_col='PBV', year_col='Year', sector_col='Sector', out_col='factor',
    low_q=0.3, high_q=0.7, start_year=2011, end_year=2022
):
    df = df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    df[year_col]  = pd.to_numeric(df[year_col], errors='coerce').astype('Int64')
    df = df[(df[year_col] >= start_year) & (df[year_col] <= end_year)]

    # Quantiles computed within (Year, Sector)
    by_group = df.groupby([year_col, sector_col])[value_col]
    q_low  = by_group.transform(lambda s: s.quantile(low_q))
    q_high = by_group.transform(lambda s: s.quantile(high_q))

    df[out_col] = np.select(
        [df[value_col] < q_low, df[value_col] > q_high],
        ['VALUE', 'GROWTH'],
        default='MIDDLE'
    )
    df[out_col] = pd.Categorical(df[out_col], categories=['VALUE','MIDDLE','GROWTH'], ordered=True)
    return df

df_labeled = label_growth_value(df, value_col='PBV', year_col='Year', sector_col='Sector')

# --------------------------
# 3) RF setup (PE as response)
# --------------------------
RESPONSE = "PBV"
BASE_FEATURE_COLS = [
    "Profit_margin","ROE","ROE_above_Cost_of_equity","CROTE_TE",
    "EVA_Margin","EVA_momentum","EVA_shock","Economic_profit_1_f",
    "EP_growth_2_f","EP_growth_3_f",
    "Revenue_growth_1_f","Revenue_growth_2_f","Revenue_growth_3_f",
    "NAV_1_f","NAV_growth_2_f","NAV_growth_3_f",
    "profit_margin_1_f","profit_margin_growth_2_f","profit_margin_growth_3_f",
    "Dividend_Yield","EVA_Profitable_Growth","EVA_Productivity_Gains","Buyback_Yield",
    # "BVE_per_share", "BVE_per_share_1_f","BVE_per_share_growth_2_f","BVE_per_share_growth_3_f",
]

def fit_rf_and_plot_importance(
    df, title, filename,
    response_col=RESPONSE, base_features=BASE_FEATURE_COLS,
    cat_cols=("Sector",), year_col="Year",
    year_min=2011, year_max=2025,
    n_estimators=100, random_state=42, n_jobs=-1, top_n=10, subset_factor=None,
    clip_ceiling=1e30  # guardrail for absurd magnitudes (float32 max ~3.4e38)
):
    """Fit RF on response_col for VALUE/GROWTH subset and plot feature importances."""
    work = df.copy()
    work[year_col] = pd.to_numeric(work[year_col], errors="coerce").astype("Int64")
    work = work[(work[year_col] >= year_min) & (work[year_col] <= year_max)]
    if subset_factor:
        work = work[work["factor"] == subset_factor]

    # Ensure response present and numeric
    if response_col not in work.columns:
        raise KeyError(f"Response column '{response_col}' not found.")
    y = pd.to_numeric(work[response_col], errors="coerce")
    y = y.replace([np.inf, -np.inf], np.nan)
    y = y.where(y.abs() < clip_ceiling)

    # Numeric features available in data
    available_num = [c for c in base_features if c in work.columns]
    if not available_num:
        raise ValueError("None of the requested numeric features are present in the data.")

    X_num = work[available_num].apply(pd.to_numeric, errors="coerce")
    X_num = X_num.replace([np.inf, -np.inf], np.nan)
    # Guard against overflow when scikit-learn casts to float32
    X_num = X_num.where(X_num.abs() < clip_ceiling)

    # Categorical dummies (optional)
    cat_cols = [c for c in cat_cols if c in work.columns]
    X_cat = pd.get_dummies(work[cat_cols], drop_first=False) if cat_cols else pd.DataFrame(index=work.index)

    # Build X and drop rows with any NA
    X = pd.concat([X_num, X_cat], axis=1)
    mask = y.notna() & X.notna().all(axis=1)
    X = X.loc[mask]
    y = y.loc[mask]

    if X.shape[0] < 10:
        raise ValueError(f"Too few rows after cleaning: n={X.shape[0]}.")

    # Drop constant columns
    keep_cols = X.columns[X.nunique(dropna=False) > 1].tolist()
    X = X[keep_cols]
    if X.empty:
        raise ValueError("All candidate features were constant; no usable features left after cleaning.")

    # Final sanity: ensure finite after clipping, then cast to float32
    X = X.replace([np.inf, -np.inf], np.nan)
    finite_mask = np.isfinite(X.to_numpy(dtype=np.float64)).all(axis=1)
    X = X.loc[finite_mask]
    y = y.loc[finite_mask]
    if X.empty:
        raise ValueError("After removing non-finite values, no data remains for modeling.")

    X = X.astype(np.float32)
    y = y.astype(np.float32)

    # Fit RF
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs)
    rf.fit(X, y)

    # Importances
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    # Plot
    top_features = importances.head(top_n)
    plt.figure(figsize=(10, 6), dpi=150)
    top_features.plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(f"AIA_{filename}", dpi=150)  # PNG filenames prefixed with AIA_
    plt.show()

    return importances

# --------------------------
# 4) Run two insurer models with custom titles & AIA filenames
# --------------------------
imp_ins_growth = fit_rf_and_plot_importance(
    df_labeled,
    "Valuation drivers: Global growth insurers",
    "Global_growth_insurers_PE.png",
    subset_factor="GROWTH"
)

imp_ins_value = fit_rf_and_plot_importance(
    df_labeled,
    "Valuation drivers: Global value insurers",
    "Global_value_insurers_PE.png",
    subset_factor="VALUE"
)

# --------------------------
# (Optional) quick probe for outliers by column (uncomment if needed)
# --------------------------
# probe = df_labeled[BASE_FEATURE_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
# print(probe.abs().max().sort_values(ascending=False).head(15))
