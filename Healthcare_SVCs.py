import pandas as pd
from pathlib import Path

# =========================
# Paths
# =========================
BASE   = Path(r"C:\Users\60848\OneDrive - Bain\Desktop\Genome_code_250605\Genome-pipeline-code\Genome-pipeline-code\global_platform_data\share_price")
GLOBAL = Path(r"C:\Users\60848\OneDrive - Bain\Desktop\Genome_code_250605\Genome-pipeline-code\Genome-pipeline-code\global_platform_data\Global_data.csv")

# =========================
# Helpers
# =========================
def normalize_sector(s: pd.Series) -> pd.Series:
    """Turn 'Healthcare' and 'Health Care' (incl. weird spacing) into 'healthcare'."""
    return (s.astype(str).str.replace(r"\s+", "", regex=True).str.lower())

def tsr_for(master_sorted: pd.DataFrame, latest: pd.DataFrame, years: int) -> pd.Series:
    """
    Compute TSR over `years` using nearest earlier trading date per Ticker.
    master_sorted must have Ticker, Date, Price (sorted by Date).
    latest must be indexed by Ticker with columns latest_date, latest_price.
    """
    targets = latest[["latest_date"]].copy()
    targets["target"] = targets["latest_date"] - pd.DateOffset(years=years)
    asof = pd.merge_asof(
        targets.sort_values("target").reset_index(),                        # left: Ticker, target
        master_sorted[["Ticker","Date","Price"]].sort_values("Date"),       # right: price history
        left_on="target", right_on="Date", by="Ticker", direction="backward"
    ).set_index("Ticker")
    return ((latest["latest_price"] / asof["Price"]) ** (1/years) - 1).rename(f"TSR_{years}Y")

# =========================
# Load global file and define Healthcare scope
# =========================
g = pd.read_csv(GLOBAL)
g["Sector"] = normalize_sector(g["Sector"])
scope = (g[g["Sector"].eq("healthcare")]
         [["Company_name","Country","Ticker"]]
         .dropna().drop_duplicates())

# =========================
# Build master price table from country folders
# =========================
frames, missing = [], []
for _, r in scope.iterrows():
    p = BASE / str(r["Country"]) / f"_{r['Ticker']}_price.csv"
    try:
        sp = pd.read_csv(p, parse_dates=["Date"])
        sp = sp.assign(
            Ticker=r["Ticker"],
            Company_name=r["Company_name"],
            Country=r["Country"]
        )
        # Enforce dtypes for safety
        sp["Date"]  = pd.to_datetime(sp["Date"], errors="coerce")
        sp["Price"] = pd.to_numeric(sp["Price"], errors="coerce")
        frames.append(sp)
    except Exception:
        missing.append((r["Company_name"], r["Country"], r["Ticker"], str(p)))

master = (pd.concat(frames, ignore_index=True)
          if frames else pd.DataFrame(columns=["Ticker","Company_name","Country","Date","Price"]))

master = (master[["Ticker","Company_name","Country","Date","Price"]]
          .dropna(subset=["Ticker","Date","Price"])
          .sort_values(["Ticker","Date"])
          .reset_index(drop=True))

# Optional: save the master
# master.to_csv(BASE / "master_share_prices.csv", index=False)

# =========================
# Compute 1/3/5/10-year TSRs
# =========================
latest = (master.groupby("Ticker").tail(1)
          .rename(columns={"Date":"latest_date","Price":"latest_price"})
          .set_index("Ticker")[["Company_name","Country","latest_date","latest_price"]])

out = latest[["Company_name","Country"]].copy()
for y in (1, 3, 5, 10):
    out = out.join(tsr_for(master, latest, y))

tsr_table = (out
             .reset_index()
             [["Company_name","Country","Ticker","TSR_1Y","TSR_3Y","TSR_5Y","TSR_10Y"]])

# =========================
# 2015–2024 count metrics (Revenue growth > 3%, EVA ≥ 0, and BOTH)
# =========================
cols_needed = ["Company_name","Country","Ticker","Sector","Year","Revenue_growth_1_f","EVA_ratio_bespoke"]
raw = g[[c for c in cols_needed if c in g.columns]].copy()

raw["Sector"] = normalize_sector(raw["Sector"])
raw = raw[raw["Sector"].eq("healthcare")]

raw["Year"] = pd.to_numeric(raw["Year"], errors="coerce").astype("Int64")
raw["Revenue_growth_1_f"] = pd.to_numeric(raw["Revenue_growth_1_f"], errors="coerce")
raw["EVA_ratio_bespoke"]  = pd.to_numeric(raw["EVA_ratio_bespoke"], errors="coerce")

panel = raw[raw["Year"].between(2015, 2024, inclusive="both")].copy()

panel["rg_ok"]   = (panel["Revenue_growth_1_f"] > 0.03).fillna(False)
panel["eva_ok"]  = (panel["EVA_ratio_bespoke"] >= 0).fillna(False)
panel["both_ok"] = panel["rg_ok"] & panel["eva_ok"]

counts = (panel.groupby("Ticker")[["rg_ok","eva_ok","both_ok"]]
               .sum()
               .rename(columns={
                   "rg_ok":   "Years_RG_gt_3pct_2015_2024",
                   "eva_ok":  "Years_EVA_ge_0_2015_2024",
                   "both_ok": "Years_Both_2015_2024"
               }))

# Merge counts into TSR table
tsr_table = tsr_table.merge(counts, on="Ticker", how="left").fillna({
    "Years_RG_gt_3pct_2015_2024": 0,
    "Years_EVA_ge_0_2015_2024":   0,
    "Years_Both_2015_2024":       0
}).astype({
    "Years_RG_gt_3pct_2015_2024": "int64",
    "Years_EVA_ge_0_2015_2024":   "int64",
    "Years_Both_2015_2024":       "int64"
})

# =========================
# Output / summary
# =========================
# Optional save of the enriched table
tsr_table.to_csv(BASE / "company_tsr_1_3_5_10_years_with_counts.csv", index=False)

print(f"Master rows: {len(master):,}; Files loaded: {len(frames)}; Missing: {len(missing)}")
print(tsr_table.head(10))
