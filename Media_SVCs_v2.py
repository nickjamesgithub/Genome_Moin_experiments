import os
import glob
import math
from datetime import datetime
from dateutil.relativedelta import relativedelta

import pandas as pd

# -------------------------
# Paths (edit if needed)
# -------------------------
BASE_DIR = r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data"
PRICE_DIR = os.path.join(BASE_DIR, r"share_price\media")
# Use the explicit mapped file path you provided
file_path = r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\Nine\media_data_global_mapped.csv"
MEDIA_DATA_PATH = file_path  # <- used below when reading the media data
# Output files (now written to casework\Nine folder)
OUTPUT_DIR = r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\Nine"
TSR_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "tsr_summary_media.csv")
CRITERIA_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "svc_criteria_2015_2024.csv")
MERGED_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "media_tsr_and_svc_summary.csv")


# -------------------------
# Market scope filter
# Choose: "developed", "emerging", or "both"
# -------------------------
MARKET_SCOPE = "developed"

DEVELOPED_SET = {
    "Australia","United States","Canada","France","Japan","United Kingdom","Luxembourg",
    "Italy","Switzerland","Germany","Finland","South Korea","Spain","Sweden","Israel",
    "Hong Kong","Singapore"
}
EMERGING_SET = {
    "South Africa","Malaysia","China","Indonesia","Saudi Arabia","India","Mexico",
    "Thailand","Argentina"
}

def classify_market(country: str) -> str:
    if pd.isna(country):
        return "Unknown"
    name = str(country).strip()
    if not name or name.lower() == "nan":
        return "Unknown"
    if name in DEVELOPED_SET:
        return "Developed"
    if name in EMERGING_SET:
        return "Emerging"
    return "Unknown"

# -------------------------
# Helpers
# -------------------------
def read_prices_for_ticker(ticker: str) -> pd.DataFrame | None:
    """
    Reads _{TICKER}_price.csv from PRICE_DIR and returns a DataFrame with Date (datetime) and Price (float).
    Returns None if not found or unreadable.
    """
    pattern = os.path.join(PRICE_DIR, f"_{ticker}_price.csv")
    matches = glob.glob(pattern)
    if not matches:
        return None
    path = matches[0]
    try:
        df = pd.read_csv(path)
        # Normalize column names just in case
        df.columns = [c.strip() for c in df.columns]
        # Parse dates (month-first typical)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=False)
        df = df.dropna(subset=["Date"]).sort_values("Date")
        # Coerce price to numeric
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
        df = df.dropna(subset=["Price"])
        df = df.reset_index(drop=True)
        return df
    except Exception as e:
        print(f"Failed reading prices for {ticker}: {e}")
        return None


def closest_date(series: pd.Series, target_ts: pd.Timestamp) -> pd.Timestamp | None:
    """
    Returns the date in 'series' that is closest to 'target_ts' (absolute difference).
    If series is empty, returns None.
    """
    if series.empty:
        return None
    diffs = (series - target_ts).abs()
    return series.iloc[diffs.idxmin()]


def compute_tsr_cagr(price_df: pd.DataFrame, horizons=(1, 3, 5, 10)) -> dict:
    """
    Given a price DataFrame with Date & Price, compute TSR and CAGR for each horizon in years.
    Uses last available date as 'end', and the closest available date to N years before as 'start'.
    """
    out = {}
    if price_df.empty:
        return out

    last_row = price_df.iloc[-1]
    last_date = pd.Timestamp(last_row["Date"])
    last_price = float(last_row["Price"])
    out["LastDate"] = last_date.date().isoformat()
    out["LastPrice"] = last_price

    for n in horizons:
        target = last_date - relativedelta(years=n)
        start_date = closest_date(price_df["Date"], target)
        if start_date is None:
            out[f"StartDate_{n}Y"] = None
            out[f"StartPrice_{n}Y"] = None
            out[f"TSR_{n}Y"] = None
            out[f"CAGR_{n}Y"] = None
            continue

        start_price = float(price_df.loc[price_df["Date"] == start_date, "Price"].iloc[0])
        out[f"StartDate_{n}Y"] = start_date.date().isoformat()
        out[f"StartPrice_{n}Y"] = start_price

        if start_price <= 0 or last_price <= 0:
            tsr = None
            cagr = None
        else:
            tsr = (last_price / start_price) - 1.0
            try:
                cagr = (last_price / start_price) ** (1.0 / n) - 1.0
            except Exception:
                cagr = None

        out[f"TSR_{n}Y"] = tsr
        out[f"CAGR_{n}Y"] = cagr

    return out


def safe_pct(v):
    return None if (v is None or (isinstance(v, float) and math.isnan(v))) else v


# -------------------------
# Load media_data + apply market filter
# -------------------------
media = pd.read_csv(MEDIA_DATA_PATH)
media.columns = [c.strip() for c in media.columns]

# Expecting at minimum: ['Ticker', 'Year', 'EVA_ratio_bespoke', 'Revenue_growth_3_f']
if "Ticker" not in media.columns:
    raise ValueError("Column 'Ticker' not found in media_data_global_mapped.csv")
if "Country_label" not in media.columns:
    raise ValueError("Column 'country_label' not found in media_data_global_mapped.csv")

media["Country_label_clean"] = media["Country_label"].astype(str).str.strip()
media["Market_Bucket"] = media["Country_label_clean"].apply(classify_market)

_scope = MARKET_SCOPE.lower().strip()
if _scope not in {"developed", "emerging", "both"}:
    raise ValueError("MARKET_SCOPE must be one of: 'developed', 'emerging', 'both'")

if _scope == "developed":
    media = media[media["Market_Bucket"] == "Developed"].copy()
elif _scope == "emerging":
    media = media[media["Market_Bucket"] == "Emerging"].copy()
# else both -> no filter

if media.empty:
    raise ValueError(f"No rows match MARKET_SCOPE='{MARKET_SCOPE}'. Check country labels or set membership.")

print(f"Rows after market filter ({MARKET_SCOPE}): {len(media)}")
print("Market mix:\n", media["Market_Bucket"].value_counts(dropna=False))

# -------------------------
# Compute TSR/CAGR for each ticker
# -------------------------
tickers = sorted(pd.unique(media["Ticker"].dropna().astype(str)))

tsr_rows = []
for t in tickers:
    price_df = read_prices_for_ticker(t)
    if price_df is None or price_df.empty:
        tsr_rows.append({
            "Ticker": t, "LastDate": None, "LastPrice": None,
            "StartDate_1Y": None, "StartPrice_1Y": None, "TSR_1Y": None, "CAGR_1Y": None,
            "StartDate_3Y": None, "StartPrice_3Y": None, "TSR_3Y": None, "CAGR_3Y": None,
            "StartDate_5Y": None, "StartPrice_5Y": None, "TSR_5Y": None, "CAGR_5Y": None,
            "StartDate_10Y": None, "StartPrice_10Y": None, "TSR_10Y": None, "CAGR_10Y": None
        })
        continue

    metrics = compute_tsr_cagr(price_df, horizons=(1, 3, 5, 10))
    row = {"Ticker": t}
    row.update({k: safe_pct(v) for k, v in metrics.items()})
    tsr_rows.append(row)

tsr_df = pd.DataFrame(tsr_rows)

# Order columns
cols = ["Ticker", "LastDate", "LastPrice"]
for n in (1, 3, 5, 10):
    cols += [f"StartDate_{n}Y", f"StartPrice_{n}Y", f"TSR_{n}Y", f"CAGR_{n}Y"]
tsr_df = tsr_df.reindex(columns=cols)

# Save TSR summary
os.makedirs(OUTPUT_DIR, exist_ok=True)
tsr_df.to_csv(TSR_OUTPUT_PATH, index=False)
print(f"TSR summary written to: {TSR_OUTPUT_PATH}")

# -------------------------
# Criteria counts (2015–2024 inclusive)
# -------------------------
required_cols = {"Ticker", "Year", "EVA_ratio_bespoke", "Revenue_growth_3_f"}
missing = required_cols - set(media.columns)
if missing:
    raise ValueError(f"media_data_global_mapped.csv is missing columns: {missing}")

criteria = media.copy()
criteria["Year"] = pd.to_numeric(criteria["Year"], errors="coerce").astype("Int64")
criteria = criteria[(criteria["Year"] >= 2015) & (criteria["Year"] <= 2024)]

criteria["EVA_ratio_bespoke"] = pd.to_numeric(criteria["EVA_ratio_bespoke"], errors="coerce")
criteria["Revenue_growth_3_f"] = pd.to_numeric(criteria["Revenue_growth_3_f"], errors="coerce")

criteria["EVA_nonneg"] = criteria["EVA_ratio_bespoke"] >= 0
criteria["REV3_nonneg"] = criteria["Revenue_growth_3_f"] >= 0
criteria["SVC_Criteria"] = criteria["EVA_nonneg"] & criteria["REV3_nonneg"]

agg = criteria.groupby("Ticker").agg(
    Years_in_range=("Year", lambda s: s.nunique()),
    EVA_nonneg_years=("EVA_nonneg", lambda s: int(s.sum(skipna=True))),
    REV3_nonneg_years=("REV3_nonneg", lambda s: int(s.sum(skipna=True))),
    SVC_Criteria_years=("SVC_Criteria", lambda s: int(s.sum(skipna=True))),
).reset_index()

agg["Years_out_of_10_EVA_nonneg"] = agg["EVA_nonneg_years"]
agg["Years_out_of_10_REV3_nonneg"] = agg["REV3_nonneg_years"]
agg["Years_out_of_10_SVC_Criteria"] = agg["SVC_Criteria_years"]

# Save criteria summary
agg.to_csv(CRITERIA_OUTPUT_PATH, index=False)
print(f"SVC criteria summary written to: {CRITERIA_OUTPUT_PATH}")

# -------------------------
# Merge TSR + criteria + Company_name (+ Market info)
# -------------------------
cols_for_map = ["Ticker"]
if "Company_name" in media.columns:
    cols_for_map.append("Company_name")
cols_for_map += ["Country_label_clean", "Market_Bucket"]
ticker_map = media[cols_for_map].drop_duplicates()

merged = agg.merge(tsr_df, on="Ticker", how="left")
merged = merged.merge(ticker_map, on="Ticker", how="left")

# Nice column order
first_cols = []
if "Company_name" in merged.columns:
    first_cols.append("Company_name")
first_cols += ["Ticker", "Country_label_clean", "Market_Bucket"]
remaining = [c for c in merged.columns if c not in first_cols]
merged = merged[first_cols + remaining]

os.makedirs(OUTPUT_DIR, exist_ok=True)
merged.to_csv(MERGED_OUTPUT_PATH, index=False)
print(f"Merged summary (with Company_name & Market_Bucket) written to: {MERGED_OUTPUT_PATH}")

x=1
y=2
