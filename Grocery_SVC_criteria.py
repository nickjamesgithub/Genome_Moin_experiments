import pandas as pd
import numpy as np
from pathlib import Path

# ============= PATHS (EDIT IF NEEDED) =============
BASE   = Path(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\share_price\Grocery")
GLOBAL = Path(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\grocery_data.csv")  # optional but recommended
OUTDIR = Path(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\Woolworths\SVC_data")

# ============= SIMPLE HELPERS =============
def to_datetime_any(s: pd.Series) -> pd.Series:
    """Parse dates from strings or Excel serials. Return datetime64[ns]."""
    dt = pd.to_datetime(s, errors="coerce")
    na = dt.isna()
    if na.any():
        nums = pd.to_numeric(s, errors="coerce")
        m = na & nums.notna()
        if m.any():
            dt.loc[m] = pd.to_datetime(nums[m], unit="D", origin="1899-12-30", errors="coerce")
    return dt

def norm_sector(s: pd.Series) -> pd.Series:
    """'Gro cery' -> 'grocery'."""
    return s.astype(str).str.replace(r"\s+", "", regex=True).str.lower()

def base_symbol(t: str) -> str:
    """Take the part before '.', '-', '/', or space. Uppercase + strip."""
    t = str(t).upper().strip()
    for sep in [".", "-", "/", " "]:
        if sep in t:
            t = t.split(sep, 1)[0]
    return t

def extract_file_ticker(p: Path) -> str:
    """'_WOW_price.csv' -> 'WOW' (raw); then we normalize with base_symbol()."""
    name = p.name
    if name.startswith("_") and name.lower().endswith("_price.csv"):
        raw = name[1:-10].strip()
        return raw
    return ""

def tsr_for(master_sorted: pd.DataFrame, latest: pd.DataFrame, years: int) -> pd.Series:
    """Compute TSR over `years` using nearest earlier trading date per Listing."""
    targets = latest[["latest_date"]].copy()
    targets["target"] = targets["latest_date"] - pd.DateOffset(years=years)

    left = targets.reset_index()
    left["target"] = pd.to_datetime(left["target"], errors="coerce")
    left = left.dropna(subset=["target"]).sort_values(["target", "Listing"])

    right = master_sorted[["Listing","Date","Price"]].copy()
    right["Date"] = pd.to_datetime(right["Date"], errors="coerce")
    right = right.dropna(subset=["Date"]).sort_values(["Date","Listing"])

    asof = pd.merge_asof(
        left, right,
        left_on="target", right_on="Date",
        by="Listing",
        direction="backward",
        allow_exact_matches=True
    ).set_index("Listing")

    denom = asof["Price"].replace({0: pd.NA})
    tsr = ((latest["latest_price"] / denom) ** (1/years) - 1)
    return tsr.rename(f"TSR_{years}Y")

# ============= CHECK PATHS =============
if not BASE.is_dir():
    raise FileNotFoundError(f"BASE folder not found: {BASE}")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ============= READ ALL PRICE FILES (NO SCOPE NEEDED) =============
price_files = list(BASE.glob("_*_price.csv"))
if not price_files:
    raise ValueError(f"No price files found in BASE: {BASE}")

frames = []
for p in price_files:
    try:
        df = pd.read_csv(p)
        if "Date" not in df.columns or "Price" not in df.columns:
            # Skip bad file shapes silently to keep things simple
            continue
        df["Date"]  = to_datetime_any(df["Date"])
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
        df = df.dropna(subset=["Date","Price"]).copy()

        file_ticker_raw = extract_file_ticker(p)  # e.g., WOW, WES.AX, etc.
        file_ticker_norm = base_symbol(file_ticker_raw)  # normalize to base (WOW)
        df["Ticker_raw"]  = file_ticker_raw
        df["Ticker_base"] = file_ticker_norm

        frames.append(df[["Date","Price","Ticker_raw","Ticker_base"]])
    except Exception:
        # Keep it simple: skip problematic files
        continue

if not frames:
    raise ValueError("No valid price data rows parsed from files.")

prices = pd.concat(frames, ignore_index=True)
# At this point we have price history for EVERY file. Now we *optionally* enrich from GLOBAL.

# ============= (OPTIONAL) LOAD GLOBAL METADATA FOR GROCERY =============
meta = None
if GLOBAL.is_file():
    try:
        g = pd.read_csv(GLOBAL)
        # Try to standardize minimal needed columns
        col_company = "Company_name" if "Company_name" in g.columns else ("Company name" if "Company name" in g.columns else None)
        col_ticker  = "Ticker" if "Ticker" in g.columns else ("Ticker_full" if "Ticker_full" in g.columns else None)
        col_country = "Country" if "Country" in g.columns else None
        col_sector  = "Sector" if "Sector" in g.columns else ("Sector_new" if "Sector_new" in g.columns else None)

        if col_ticker:
            g["Ticker_base"] = g[col_ticker].astype(str).map(base_symbol)
        if col_sector:
            g["Sector_norm"] = norm_sector(g[col_sector])
        else:
            g["Sector_norm"] = ""

        # Keep grocery only if we can detect sector; else just keep all rows
        if "grocery" in set(g["Sector_norm"]):
            g2 = g[g["Sector_norm"].eq("grocery")].copy()
        else:
            g2 = g.copy()

        # Build a slim metadata table keyed by Ticker_base
        keep_cols = ["Ticker_base"]
        if col_company: keep_cols.append(col_company)
        if col_country: keep_cols.append(col_country)
        meta = g2[keep_cols].dropna(subset=["Ticker_base"]).drop_duplicates("Ticker_base").copy()
        meta = meta.rename(columns={
            (col_company or "Company_name"): "Company_name",
            (col_country or "Country"): "Country"
        })
    except Exception:
        meta = None

# ============= BUILD MASTER WITH LISTING KEY =============
if meta is not None and not meta.empty:
    prices = prices.merge(meta, on="Ticker_base", how="left")
    prices["Company_name"] = prices["Company_name"].fillna(prices["Ticker_raw"])
    prices["Country"] = prices["Country"].fillna("UNK")
else:
    prices["Company_name"] = prices["Ticker_raw"]
    prices["Country"] = "UNK"

prices["Listing"] = prices["Country"].astype(str) + "::" + prices["Ticker_base"]

master = (prices[["Ticker_base","Company_name","Country","Listing","Date","Price"]]
          .sort_values(["Listing","Date"])
          .reset_index(drop=True))

# Safety: if still empty, bail early
if master.empty:
    raise ValueError("No price data after normalization. Check files contain valid Date/Price columns.")

# ============= LATEST PRICE + TSRs =============
latest = (master.groupby("Listing").tail(1)
          .rename(columns={"Date":"latest_date","Price":"latest_price"})
          .set_index("Listing")[["Company_name","Country","Ticker_base","latest_date","latest_price"]])

out = latest[["Company_name","Country","Ticker_base"]].copy()
for y in (1, 3, 5, 10):
    out = out.join(tsr_for(master, latest, y))

tsr_table = (out.reset_index()
             .rename(columns={"Ticker_base":"Ticker"})
             [["Company_name","Country","Ticker","Listing","TSR_1Y","TSR_3Y","TSR_5Y","TSR_10Y"]])

# ============= (OPTIONAL) 2015–2024 COUNT METRICS =============
# Only if GLOBAL has the needed columns; otherwise skip quietly.
add_counts = False
if GLOBAL.is_file():
    try:
        g = pd.read_csv(GLOBAL)
        needed = ["Ticker","Year","Revenue_growth_1_f","EVA_ratio_bespoke"]
        if all(col in g.columns for col in needed):
            g["Ticker_base"] = g["Ticker"].astype(str).map(base_symbol)
            g["Year"] = pd.to_numeric(g["Year"], errors="coerce").astype("Int64")
            g["Revenue_growth_1_f"] = pd.to_numeric(g["Revenue_growth_1_f"], errors="coerce")
            g["EVA_ratio_bespoke"]  = pd.to_numeric(g["EVA_ratio_bespoke"],  errors="coerce")

            panel = g[g["Year"].between(2015, 2024, inclusive="both")].copy()
            panel["rg_ok"]   = (panel["Revenue_growth_1_f"] > 0.03).fillna(False)
            panel["eva_ok"]  = (panel["EVA_ratio_bespoke"]  >= 0).fillna(False)
            panel["both_ok"] = panel["rg_ok"] & panel["eva_ok"]

            by_year = (panel.groupby(["Ticker_base","Year"], as_index=False)[["rg_ok","eva_ok","both_ok"]]
                             .max())

            counts = (by_year.groupby("Ticker_base")[["rg_ok","eva_ok","both_ok"]]
                             .sum()
                             .clip(upper=10)
                             .rename(columns={
                                 "rg_ok":   "Years_RG_gt_3pct_2015_2024",
                                 "eva_ok":  "Years_EVA_ge_0_2015_2024",
                                 "both_ok": "Years_Both_2015_2024"
                             })
                             .reset_index())

            tsr_table["Ticker_base"] = tsr_table["Ticker"].map(base_symbol)
            tsr_table = (tsr_table
                         .merge(counts, on="Ticker_base", how="left")
                         .drop(columns=["Ticker_base"])
                         .fillna({
                             "Years_RG_gt_3pct_2015_2024": 0,
                             "Years_EVA_ge_0_2015_2024":   0,
                             "Years_Both_2015_2024":       0
                         })
                         .astype({
                             "Years_RG_gt_3pct_2015_2024": "int64",
                             "Years_EVA_ge_0_2015_2024":   "int64",
                             "Years_Both_2015_2024":       "int64"
                         }))
            add_counts = True
    except Exception:
        add_counts = False

# ============= OUTPUT =============
OUTDIR.mkdir(parents=True, exist_ok=True)
out_path = OUTDIR / "grocery_company_tsr_1_3_5_10_years_with_counts.csv"
tsr_table.to_csv(out_path, index=False)

# ============= PRINTS =============
print(f"Price files found: {len(price_files)}")
print(f"Master price rows: {len(master):,}")
print(f"Listings with latest prices: {len(latest):,}")
print("Counts added:" , "Yes" if add_counts else "No (GLOBAL columns missing or not usable)")
print("\nHead of output:")
print(tsr_table.head(10).to_string(index=False))
print(f"\nSaved to: {out_path}")

x=1
y=2
