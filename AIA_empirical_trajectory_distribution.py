import numpy as np
import pandas as pd
from pathlib import Path
import math

# ========= Config =========
csv_path = Path(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\Insurance_data\merged_insurance_data_with_genome.csv")
out_dir  = Path(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\Insurance_data\journeys_asian_insurance")
out_dir.mkdir(parents=True, exist_ok=True)

VERBOSE = True
iter_no = 0

begin_year, end_year, W = 2011, 2024, 3  # 3y window

# ========= Helpers =========
def pick(df, *cands):
    for c in cands:
        if c in df.columns:
            return c
    return None

def ann_return(start, end, add=0.0, years=1):
    if pd.isna(start) or start == 0 or pd.isna(end):
        return np.nan
    cum = (end - start + (add or 0.0)) / start
    return (1.0 + cum) ** (1.0 / years) - 1.0

def pct_change_safe(end, begin):
    end = pd.to_numeric(end, errors="coerce")
    begin = pd.to_numeric(begin, errors="coerce")
    out = np.full(len(end), np.nan)
    mask = (begin != 0) & (~begin.isna()) & (~end.isna())
    out[mask] = (end[mask] / begin[mask]) - 1.0
    return out

# ========= Load & prepare =========
df = pd.read_csv(csv_path)

# Required columns (present in your file)
id_col     = pick(df, "Ticker_full", "Ticker", "Company_name")
name_col   = "Company_name"
year_col   = "Year"
home_col   = "Home_Country"
type_col   = "Type"
x_col      = pick(df, "GWP_growth_3y_x", "GWP_growth_3y", "GWP_growth_3y_y")  # growth axis
y_col      = "ROE_above_hurdle"                                              # profitability axis
px_col     = pick(df, "Stock_Price", "Adjusted_Stock_Price")
adj_px_col = pick(df, "Adjusted_Stock_Price", "Stock_Price")
dbbps_col  = "DBBPS"
pe_col     = "PE_Implied"
mktcap_col = "Market_Capitalisation"

# Compute Price-to-book if missing
if "Price_to_book" not in df.columns:
    bv = pick(df, "Book_Value_Equity", "Total_equity")
    df["Price_to_book"] = (df[mktcap_col] / df[bv]) if bv else np.nan

# Dtypes & basic cleanup
df[year_col] = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")
for c in [id_col, name_col, home_col, type_col]:
    if c in df.columns:
        df[c] = df[c].astype("string")
if dbbps_col in df.columns:
    df[dbbps_col] = pd.to_numeric(df[dbbps_col], errors="coerce").fillna(0.0)

# Limit year range
df = df[df[year_col].between(begin_year, end_year)].copy()

# ========= Build journeys =========
years = np.arange(begin_year, end_year + 1)
rows = []

for y0 in years[:-W]:
    y1 = y0 + W

    left = df[df[year_col].eq(y0)][[
        id_col, name_col, home_col, type_col,
        x_col, y_col, px_col, adj_px_col,
        "Price_to_book", pe_col, mktcap_col
    ]]
    right = df[df[year_col].eq(y1)][[
        id_col,
        x_col, y_col, px_col, adj_px_col,
        "Price_to_book", pe_col, mktcap_col   # ← include PBV on the right too
    ]]

    L, R = left.add_suffix("_0"), right.add_suffix("_1")
    merged = L.merge(R, left_on=f"{id_col}_0", right_on=f"{id_col}_1", how="inner")
    if merged.empty:
        continue

    # Sum DBBPS across window (inclusive)
    flows = (
        df[df[year_col].between(y0, y1)]
        .groupby(id_col, as_index=False)[dbbps_col]
        .sum()
        .rename(columns={dbbps_col: "DBBPS_total"})
    )
    merged = merged.merge(flows, left_on=f"{id_col}_0", right_on=id_col, how="left").drop(columns=[id_col])
    merged["DBBPS_total"] = merged["DBBPS_total"].fillna(0.0)

    # Angle between (X0,Y0) -> (X1,Y1)
    dx = merged[f"{x_col}_1"] - merged[f"{x_col}_0"]
    dy = merged[f"{y_col}_1"] - merged[f"{y_col}_0"]
    radians = np.arctan2(dy, dx)
    degrees = (np.degrees(radians) + 360) % 360

    # TSR (cash-inclusive) and CapIQ-style (adjusted price only)
    ann_tsr = merged.apply(lambda r: ann_return(r[f"{px_col}_0"], r[f"{px_col}_1"], r["DBBPS_total"], W), axis=1)
    ann_tsr_capiq = merged.apply(lambda r: ann_return(r[f"{adj_px_col}_0"], r[f"{adj_px_col}_1"], 0.0, W), axis=1)

    # Valuation: PBV & PE begin/end + changes
    pb0 = merged["Price_to_book_0"]
    pb1 = merged["Price_to_book_1"]
    pe0 = merged[f"{pe_col}_0"]
    pe1 = merged[f"{pe_col}_1"]

    pb_delta = (pb1 - pb0)
    pb_uplift_pct = pct_change_safe(pb1, pb0)
    pe_delta = (pe1 - pe0)
    pe_uplift_pct = pct_change_safe(pe1, pe0)

    # Journey classification by sign of ROE_above_hurdle
    y0v, y1v = merged[f"{y_col}_0"], merged[f"{y_col}_1"]
    journey = np.select(
        [
            (y0v < 0) & (y1v < 0),
            (y0v < 0) & (y1v >= 0),
            (y0v >= 0) & (y1v >= 0),
            (y0v > 0) & (y1v < 0),
        ],
        ["Remain_negative", "Move_up", "Remain_positive", "Move_down"],
        default="Unknown",
    )

    rows.append(pd.DataFrame({
        "Journey": journey,
        "Company_name": merged[f"{name_col}_0"],
        "Ticker": merged[f"{id_col}_0"],
        "Home_Country": merged[f"{home_col}_0"],
        "Type": merged[f"{type_col}_0"],

        "Year_beginning": y0,
        "Year_final": y1,

        # Axes
        "X_beginning": merged[f"{x_col}_0"],
        "X_end": merged[f"{x_col}_1"],
        "Y_beginning": merged[f"{y_col}_0"],
        "Y_end": merged[f"{y_col}_1"],

        # Prices (begin/end)
        "Stock_price_beginning": merged[f"{px_col}_0"],
        "Stock_price_end": merged[f"{px_col}_1"],
        "Adj_stock_price_beginning": merged[f"{adj_px_col}_0"],
        "Adj_stock_price_end": merged[f"{adj_px_col}_1"],

        # Valuation (PBV & PE begin/end + uplift)
        "Price_to_book_beginning": pb0,
        "Price_to_book_end": pb1,
        "PBV_change_abs": pb_delta,
        "PBV_change_pct": pb_uplift_pct,

        "PE_beginning": pe0,
        "PE_end": pe1,
        "PE_change_abs": pe_delta,
        "PE_change_pct": pe_uplift_pct,

        # Cash & TSRs
        "DBBPS_total": merged["DBBPS_total"],
        "Angle": degrees,
        "Radians": radians,
        "Annualized_TSR": ann_tsr,
        "Annualized_TSR_Capiq": ann_tsr_capiq,

        # Size (begin/end)
        "Market_Capitalisation_beginning": merged[f"{mktcap_col}_0"],
        "Market_Capitalisation_end": merged[f"{mktcap_col}_1"],
    }))

    # --- Debug prints ---------------------------------------------------------
    if VERBOSE:
        n_in_window = len(merged)
        names = merged[f"{name_col}_0"].astype("string")
        types = merged.get(f"{type_col}_0", pd.Series(pd.NA, index=merged.index, dtype="string")).astype("string")
        homes = merged.get(f"{home_col}_0", pd.Series(pd.NA, index=merged.index, dtype="string")).astype("string")
        for k, (nm, tp, hc) in enumerate(zip(names, types, homes), start=1):
            iter_no += 1
            print(f"[{iter_no:06d}] {nm} | Years {y0}-{y1} | Type={tp} | Home_Country={hc} | ({k}/{n_in_window} in {y0}->{y1})")

# Concatenate & clean
journeys = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
journeys.replace([np.inf, -np.inf], np.nan, inplace=True)

# Optional: light sanity filter
filtered = journeys.loc[
    journeys["Y_end"].between(-0.3, 0.5, inclusive="both") &
    journeys["X_end"].between(-0.3, 1.5, inclusive="both") &
    journeys["Annualized_TSR_Capiq"].between(-0.4, 1.0, inclusive="both") &
    (journeys["Price_to_book_beginning"] > -200)
].copy()

# Deltas
for df_out in (journeys, filtered):
    if not df_out.empty:
        df_out["X_change"] = df_out["X_end"] - df_out["X_beginning"]
        df_out["Y_change"] = df_out["Y_end"] - df_out["Y_beginning"]

# Write
journeys.to_csv(out_dir / "Journeys_summary_insurance.csv", index=False)
filtered.to_csv(out_dir / "Journeys_summary_insurance_filtered.csv", index=False)

print("Done:",
      f"\n  All journeys  -> {out_dir / 'Journeys_summary_insurance.csv'}",
      f"\n  Filtered      -> {out_dir / 'Journeys_summary_insurance_filtered.csv'}")
