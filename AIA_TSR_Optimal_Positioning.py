# -*- coding: utf-8 -*-
"""
Flexible TSR window + MEDIAN top-quartile coordinates with TSR appended.
- File: merged_insurance_data_with_genome.csv
- TSR window: [START_YEAR, END_YEAR] => annualized
- Optional pre-filters:
    * HOME_COUNTRY_FILTER: keep only these Home_Country values (case-insensitive)
    * TYPE_FILTER:         keep only these Type values (case-insensitive)
    * ONLY_INSURANCE:      if True, left-merge Global_data.csv and keep Sector == "Insurance"
- Outputs (saved to /reports next to merged CSV):
    1) insurer_{span}y_tsr_ranked_{tag}.csv
    2) median_coords_top_quartile_overall_{tag}.csv  (adds Median_TSR_{span}y)
    3) median_coords_top_quartile_by_type_{tag}.csv  (adds Median_TSR_{span}y)
    4) median_coords_top_quartile_by_home_country_{tag}.csv  (adds Median_TSR_{span}y)
    5) median_tsr_by_home_country_all_{tag}.csv      (all insurers, not just top quartile)
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ========= Paths =========
merged_path = Path(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\Insurance_data\merged_insurance_data_with_genome.csv")
out_dir     = merged_path.parent / "reports"
out_dir.mkdir(parents=True, exist_ok=True)

# Optional: merge in Global_data to restrict to Sector == "Insurance"
ONLY_INSURANCE   = False  # set True to enable the merge + filter
GLOBAL_DATA_PATH = r"C:\Users\60848\OneDrive - Bain\Desktop\Genome_code_250605\Genome-pipeline-code\Genome-pipeline-code\global_platform_data\Global_data.csv"

# ========= Filters (EDIT THESE) =========
# Examples:
# HOME_COUNTRY_FILTER = ["United States", "Japan", "Australia"]
TYPE_FILTER         = ["Life", "Multiline", "P&C"]
HOME_COUNTRY_FILTER: list[str] = ['Hong Kong', 'China', 'Hong Kong/China',
       'Singapore', 'Japan', 'India', 'South Korea', 'Thailand',
       'Australia', 'Vietnam', 'Malaysia', 'Indonesia']   # leave empty => keep all
TYPE_FILTER: list[str] = []           # leave empty => keep all
# =======================================

# ========= Window (EDIT THESE TWO ONLY) =========
START_YEAR = 2014
END_YEAR   = 2024
# ================================================
PERIOD_YEARS = END_YEAR - START_YEAR
if PERIOD_YEARS <= 0:
    raise ValueError("END_YEAR must be > START_YEAR.")
SPAN_LABEL  = f"{PERIOD_YEARS}y"
TAG         = f"{START_YEAR}_{END_YEAR}"

# ========= Outputs =========
rank_csv                 = out_dir / f"insurer_{SPAN_LABEL}_tsr_ranked_{TAG}.csv"
median_overall_csv       = out_dir / f"median_coords_top_quartile_overall_{TAG}.csv"
median_by_type_csv       = out_dir / f"median_coords_top_quartile_by_type_{TAG}.csv"
median_by_home_csv       = out_dir / f"median_coords_top_quartile_by_home_country_{TAG}.csv"
median_tsr_home_all_csv  = out_dir / f"median_tsr_by_home_country_all_{TAG}.csv"

# ========= Helpers =========
def pick(df, *cands):
    for c in cands:
        if c in df.columns:
            return c
    return None

def to_decimal(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    med = x.dropna().median()
    return x / 100.0 if pd.notna(med) and med > 1.5 else x

def ann_return(start_px: float, end_px: float, years: int) -> float:
    if pd.isna(start_px) or pd.isna(end_px) or start_px == 0:
        return np.nan
    return (end_px / start_px) ** (1.0 / years) - 1.0

def first_scalar(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return s.iloc[0] if not s.empty else np.nan

def normalize_str(s: pd.Series) -> pd.Series:
    """Lower + strip for case-insensitive matching, preserves NA."""
    return s.astype("string").str.strip().str.lower()

def apply_list_filter(df: pd.DataFrame, col: str, keep_values: list[str]) -> pd.DataFrame:
    """Case-insensitive filter. Empty list => no filtering."""
    if not keep_values or col not in df.columns:
        return df
    # normalize
    df_norm = df.copy()
    df_norm[col] = normalize_str(df_norm[col])
    want = pd.Series(keep_values, dtype="string").str.strip().str.lower()
    return df.loc[df_norm[col].isin(set(want))].copy()

# ========= Load =========
df = pd.read_csv(merged_path).replace([np.inf, -np.inf], np.nan)

# ========= Column picks =========
year_col   = "Year"
id_col     = pick(df, "Ticker_full", "Ticker", "Company_name")
name_col   = pick(df, "Company_name")
type_col   = pick(df, "Type")
home_col   = pick(df, "Home_Country", "Home Country", "Country_of_Domicile", "HQ_Country", "Country")
adj_px_col = pick(df, "Adjusted_Stock_Price")

roe_col    = pick(df, "ROE_BCN", "ROE_BCN_x", "ROE_BCN_y")
hurdle_col = pick(df, "ROE_above_hurdle")
gwp3y_col  = pick(df, "GWP_growth_3y", "GWP_growth_3y_x", "GWP_growth_3y_y")

pbv_col = pick(df, "Price_to_book", "PBV")
if pbv_col is None:
    mktcap = pick(df, "Market_Capitalisation", "Market Cap", "MarketCap")
    bve    = pick(df, "Book_Value_Equity", "Total_equity", "Total Equity")
    if mktcap and bve:
        df["Price_to_book"] = pd.to_numeric(df[mktcap], errors="coerce") / pd.to_numeric(df[bve], errors="coerce")
        pbv_col = "Price_to_book"

# ========= OPTIONAL: merge Global_data to restrict to Sector == "Insurance" =========
if ONLY_INSURANCE:
    try:
        global_data = pd.read_csv(GLOBAL_DATA_PATH)
        id_col_global = pick(global_data, "Ticker_full", "Ticker", "Company_name")
        if id_col is None or id_col_global is None:
            raise KeyError("Could not find a common ID column to merge on.")
        cols_to_pull = [id_col_global, "Sector"] if "Sector" in global_data.columns else [id_col_global]
        df = df.merge(global_data[cols_to_pull], left_on=id_col, right_on=id_col_global, how="left")
        if "Sector" in df.columns:
            df = df[df["Sector"].astype("string").str.strip().str.lower() == "insurance"].copy()
        # Clean up merge key if created
        if id_col_global in df.columns and id_col_global != id_col:
            df.drop(columns=[id_col_global], inplace=True, errors="ignore")
    except FileNotFoundError:
        print(f"[WARN] Global_data file not found at: {GLOBAL_DATA_PATH}. Continuing without Sector filter.")
    except Exception as e:
        print(f"[WARN] Skipped Sector filter due to: {e}")

# ========= Coercions =========
df[year_col] = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")
for c in [id_col, name_col, type_col, home_col]:
    if c and c in df.columns:
        df[c] = df[c].astype("string")

for c in [roe_col, hurdle_col, gwp3y_col, pbv_col, adj_px_col]:
    if c and c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

if roe_col:
    df[roe_col] = to_decimal(df[roe_col])

# ========= APPLY LIST FILTERS (before windowing) =========
if HOME_COUNTRY_FILTER and home_col:
    df = apply_list_filter(df, home_col, HOME_COUNTRY_FILTER)

if TYPE_FILTER and type_col:
    df = apply_list_filter(df, type_col, TYPE_FILTER)

# ========= Restrict to the window =========
w = df[df[year_col].between(START_YEAR, END_YEAR)].copy()

# ========= Per-insurer metrics & TSR =========
if id_col is None or adj_px_col is None:
    raise KeyError("Need an ID column and Adjusted_Stock_Price to compute TSR.")

def summarize(ins: pd.DataFrame) -> pd.Series:
    # Scalar prices for window ends
    p0 = first_scalar(ins.loc[ins[year_col].eq(START_YEAR), adj_px_col])
    p1 = first_scalar(ins.loc[ins[year_col].eq(END_YEAR),   adj_px_col])
    tsr = ann_return(p0, p1, PERIOD_YEARS)

    n_hurdle_pos = (pd.to_numeric(ins.get(hurdle_col, np.nan), errors="coerce") > 0).sum() if hurdle_col else np.nan
    avg_roe      = pd.to_numeric(ins.get(roe_col, np.nan), errors="coerce").mean(skipna=True) if roe_col else np.nan

    s_gwp        = pd.to_numeric(ins.get(gwp3y_col, np.nan), errors="coerce") if gwp3y_col else pd.Series(dtype=float)
    avg_gwp3y    = s_gwp.mean(skipna=True) if not s_gwp.empty else np.nan
    n_gwp_pos    = (s_gwp > 0).sum() if not s_gwp.empty else np.nan

    s_pbv        = pd.to_numeric(ins.get(pbv_col, np.nan), errors="coerce") if pbv_col else pd.Series(dtype=float)
    avg_pbv      = s_pbv[s_pbv > 0].mean(skipna=True) if not s_pbv.empty else np.nan

    name_val = ins[name_col].dropna().mode().iloc[0] if name_col in ins and not ins[name_col].dropna().empty else np.nan
    type_val = ins[type_col].dropna().mode().iloc[0] if type_col in ins and not ins[type_col].dropna().empty else np.nan
    home_val = ins[home_col].dropna().mode().iloc[0] if home_col in ins and not ins[home_col].dropna().empty else np.nan

    return pd.Series({
        "Company_name": name_val,
        "Type": type_val,
        "Home_Country": home_val,
        f"TSR_{SPAN_LABEL}_annualized": tsr,
        "Years_ROE_above_hurdle_pos": n_hurdle_pos,
        "Avg_ROE_BCN": avg_roe,
        "Avg_GWP_growth_3y": avg_gwp3y,
        "Years_GWP3y_pos": n_gwp_pos,
        "Avg_PBV": avg_pbv,
        "Years_observed": ins[year_col].nunique(),
        "Years_span": PERIOD_YEARS
    })

rank = (
    w.groupby(id_col, dropna=False, sort=False)
     .apply(summarize)
     .reset_index()
     .rename(columns={id_col: "Ticker"})
     .sort_values(f"TSR_{SPAN_LABEL}_annualized", ascending=False)
     .reset_index(drop=True)
)
rank.to_csv(rank_csv, index=False)

# ========= Top-quartile cohort & MEDIAN coordinates (append median TSR) =========
tsr_col = f"TSR_{SPAN_LABEL}_annualized"
tsr_nonan = rank[tsr_col].dropna()

if not tsr_nonan.empty:
    q75 = tsr_nonan.quantile(0.75)
    top_tickers = set(rank.loc[rank[tsr_col] >= q75, "Ticker"].astype("string"))

    # Bring in ALL rows from the window for those tickers
    top_rows = w[w[id_col].astype("string").isin(top_tickers)].copy()

    # Overall medians
    med_x_all   = pd.to_numeric(top_rows[gwp3y_col], errors="coerce").median(skipna=True) if gwp3y_col else np.nan
    med_y_all   = pd.to_numeric(top_rows[roe_col],    errors="coerce").median(skipna=True) if roe_col    else np.nan
    med_tsr_all = rank.loc[rank["Ticker"].astype("string").isin(top_tickers), tsr_col].median(skipna=True)

    pd.DataFrame([{
        "Median_GWP_growth_3y": med_x_all,
        "Median_ROE_BCN": med_y_all,
        f"Median_TSR_{SPAN_LABEL}": med_tsr_all,
        "N_tickers_top_quartile": len(top_tickers),
        "N_rows_used": int(top_rows[[gwp3y_col, roe_col]].apply(pd.to_numeric, errors="coerce").dropna(how="any").shape[0]) if gwp3y_col and roe_col else 0
    }]).to_csv(median_overall_csv, index=False)

    # By Type
    if type_col in top_rows.columns:
        med_by_type = (
            top_rows.groupby(type_col, dropna=False)
                    .apply(lambda g: pd.Series({
                        "Median_GWP_growth_3y": pd.to_numeric(g[gwp3y_col], errors="coerce").median(skipna=True) if gwp3y_col else np.nan,
                        "Median_ROE_BCN":       pd.to_numeric(g[roe_col],    errors="coerce").median(skipna=True) if roe_col    else np.nan,
                        f"Median_TSR_{SPAN_LABEL}": rank.loc[
                            rank["Ticker"].astype("string").isin(set(g[id_col].astype("string"))), tsr_col
                        ].median(skipna=True),
                        "N_rows_used": int(
                            g[[gwp3y_col, roe_col]].apply(pd.to_numeric, errors="coerce").dropna(how="any").shape[0]
                        ) if gwp3y_col and roe_col else 0
                    }))
                    .reset_index()
                    .rename(columns={type_col: "Type"})
                    .sort_values("Type")
        )
        med_by_type.to_csv(median_by_type_csv, index=False)
    else:
        pd.DataFrame(columns=["Type","Median_GWP_growth_3y","Median_ROE_BCN",f"Median_TSR_{SPAN_LABEL}","N_rows_used"]).to_csv(median_by_type_csv, index=False)

    # By Home_Country
    if home_col in top_rows.columns:
        med_by_home = (
            top_rows.groupby(home_col, dropna=False)
                    .apply(lambda g: pd.Series({
                        "Median_GWP_growth_3y": pd.to_numeric(g[gwp3y_col], errors="coerce").median(skipna=True) if gwp3y_col else np.nan,
                        "Median_ROE_BCN":       pd.to_numeric(g[roe_col],    errors="coerce").median(skipna=True) if roe_col    else np.nan,
                        f"Median_TSR_{SPAN_LABEL}": rank.loc[
                            rank["Ticker"].astype("string").isin(set(g[id_col].astype("string"))), tsr_col
                        ].median(skipna=True),
                        "N_rows_used": int(
                            g[[gwp3y_col, roe_col]].apply(pd.to_numeric, errors="coerce").dropna(how="any").shape[0]
                        ) if gwp3y_col and roe_col else 0
                    }))
                    .reset_index()
                    .rename(columns={home_col: "Home_Country"})
                    .sort_values("Home_Country")
        )
        med_by_home.to_csv(median_by_home_csv, index=False)
    else:
        pd.DataFrame(columns=["Home_Country","Median_GWP_growth_3y","Median_ROE_BCN",f"Median_TSR_{SPAN_LABEL}","N_rows_used"]).to_csv(median_by_home_csv, index=False)
else:
    # Write empty median files if no valid TSRs
    pd.DataFrame(columns=["Median_GWP_growth_3y","Median_ROE_BCN",f"Median_TSR_{SPAN_LABEL}","N_tickers_top_quartile","N_rows_used"]).to_csv(median_overall_csv, index=False)
    pd.DataFrame(columns=["Type","Median_GWP_growth_3y","Median_ROE_BCN",f"Median_TSR_{SPAN_LABEL}","N_rows_used"]).to_csv(median_by_type_csv, index=False)
    pd.DataFrame(columns=["Home_Country","Median_GWP_growth_3y","Median_ROE_BCN",f"Median_TSR_{SPAN_LABEL}","N_rows_used"]).to_csv(median_by_home_csv, index=False)

# ========= Median TSR by Home_Country (all insurers, not just top quartile) =========
if "Home_Country" in rank.columns:
    med_tsr_home_all = (
        rank.dropna(subset=[tsr_col])
            .groupby("Home_Country")[tsr_col]
            .median()
            .reset_index()
            .rename(columns={tsr_col: f"Median_TSR_{SPAN_LABEL}"})
            .sort_values("Home_Country")
    )
    med_tsr_home_all.to_csv(median_tsr_home_all_csv, index=False)
else:
    pd.DataFrame(columns=["Home_Country", f"Median_TSR_{SPAN_LABEL}"]).to_csv(median_tsr_home_all_csv, index=False)

# ========= Console peek =========
with pd.option_context("display.max_columns", 80, "display.width", 180):
    print(f"\nTop 15 insurers by {SPAN_LABEL} TSR (annualized) [{START_YEAR}→{END_YEAR}]:")
    print(rank.head(15))
    print("\nOverall MEDIAN coords (top-quartile cohort):")
    print(pd.read_csv(median_overall_csv))
    print("\nMEDIAN coords by Type (top-quartile cohort):")
    print(pd.read_csv(median_by_type_csv))
    print("\nMEDIAN coords by Home_Country (top-quartile cohort):")
    print(pd.read_csv(median_by_home_csv))
    print("\nMedian TSR by Home_Country (ALL insurers):")
    print(pd.read_csv(median_tsr_home_all_csv))

print("\nSaved:")
print(" ", rank_csv)
print(" ", median_overall_csv)
print(" ", median_by_type_csv)
print(" ", median_by_home_csv)
print(" ", median_tsr_home_all_csv)
