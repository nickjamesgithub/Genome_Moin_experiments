import re
import pandas as pd
from pathlib import Path

# === Paths ===
BASE   = Path(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\share_price\Grocery")
GLOBAL = Path(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\grocery_data.csv")  # optional

# === Helpers ===
def to_dt(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    m = dt.isna()
    if m.any():
        nums = pd.to_numeric(s, errors="coerce")
        dt.loc[m] = pd.to_datetime(nums[m], unit="D", origin="1899-12-30", errors="coerce")
    return dt

def base_symbol(t: str) -> str:
    return re.split(r"[.\-\/ ]", str(t).upper().strip())[0]

def file_ticker(p: Path) -> str | None:
    m = re.match(r"^_(.+?)_price\.csv$", p.name, flags=re.I)
    return base_symbol(m.group(1)) if m else None

# === Read all price files ===
frames = []
for p in sorted(BASE.glob("_*_price.csv")):
    t = file_ticker(p)
    if not t:
        continue
    d = pd.read_csv(p, usecols=["Date", "Price"])
    d["Date"] = to_dt(d["Date"])
    d["Price"] = pd.to_numeric(d["Price"], errors="coerce")
    frames.append(d.dropna().assign(Ticker=t))

prices = pd.concat(frames, ignore_index=True)
prices_wide = (prices
               .pivot_table(index="Date", columns="Ticker", values="Price", aggfunc="last")
               .sort_index())

# === Optional company name lookup ===
ticker_to_company = {}
if GLOBAL.is_file():
    g = pd.read_csv(GLOBAL)
    col_ticker = next((c for c in ["Ticker", "Ticker_full"] if c in g.columns), None)
    col_name   = next((c for c in ["Company_name", "Company name"] if c in g.columns), None)
    if col_ticker and col_name:
        g["Ticker_base"] = g[col_ticker].astype(str).map(base_symbol)
        ticker_to_company = dict(zip(g["Ticker_base"], g[col_name]))

# fallback names
ticker_to_company = {t: ticker_to_company.get(t, t) for t in prices_wide.columns}

# === Peek ===
print("Tickers → Companies:")
print(pd.Series(ticker_to_company))
print("\nPrice matrix head:")
print(prices_wide.head())

x=1
y=2