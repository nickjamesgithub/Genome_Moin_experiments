from pathlib import Path
import pandas as pd
from pandas.tseries.offsets import DateOffset
import unicodedata
import re

# ==================== CONFIG ====================
PANEL_CSV  = r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\insurance_data_clean_250925.csv"
PRICE_DIR  = r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\share_price\insurance_clean"
OUTPUT_CSV = r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\SVC_analysis\TSR_SVC_Clean.csv"
LOOKBACKS  = [1, 3, 5, 10]  # years
# =================================================

# ---------- normalizer (for mapping dictionaries) ----------
def normalize(s: str) -> str:
    if pd.isna(s): return ""
    s0 = unicodedata.normalize("NFKD", str(s))
    s1 = "".join(ch for ch in s0 if not unicodedata.combining(ch))
    return " ".join(s1.split()).strip().lower()

# ---------- insurer type dictionary (YOUR LIST) ----------
# Labels: "Life & Health", "P&C", "Reinsurance", "Multiline", "OTHER"
company_type_raw = {
    # Australia
    "AUB Group Limited": "OTHER",
    "Challenger Limited": "Life & Health",
    "Helia Group Limited": "P&C",
    "Insurance Australia Group Limited": "P&C",
    "Medibank Private Limited": "Life & Health",
    "nib holdings limited": "Life & Health",
    "QBE Insurance Group Limited": "P&C",
    "Steadfast Group Limited": "OTHER",
    "Suncorp Group Limited": "P&C",

    # Italy / Spain / Nordics / Netherlands / Belgium / France / Vietnam
    "Assicurazioni Generali S.p.A.": "Multiline",
    "Poste Italiane S.p.A.": "Multiline",
    "Unipol Assicurazioni S.p.A.": "Multiline",
    "MAPFRE S.A.": "Multiline",
    "Tryg A/S": "P&C",
    "Aegon Ltd.": "Life & Health",
    "ASR Nederland N.V.": "Multiline",
    "NN Group N.V.": "Multiline",
    "ageas SA/NV": "Multiline",
    "AXA SA": "Multiline",
    "Bao Viet Holdings": "Multiline",

    # Indonesia / Malaysia / Korea
    "PT Asuransi Tugu Pratama Indonesia": "P&C",
    "Syarikat Takaful Malaysia": "Life & Health",
    "Hanwha General Insurance": "P&C",
    "Samsung Fire & Marine Insurance Co., Ltd.": "P&C",
    "Hyundai Marine & Fire Insurance Co., Ltd.": "P&C",
    "DB Insurance Co., Ltd.": "P&C",
    "Samsung Life Insurance Co., Ltd.": "Life & Health",
    "Hanwha Life Insurance Co., Ltd.": "Life & Health",

    # UK
    "Admiral Group plc": "P&C",
    "Aviva plc": "Multiline",
    "Beazley plc": "P&C",
    "Hiscox Ltd": "P&C",
    "Legal & General Group Plc": "Life & Health",
    "M&G plc": "Life & Health",
    "Phoenix Group Holdings plc": "Life & Health",

    # US / Bermuda / India
    "Prudential Financial, Inc.": "Life & Health",
    "Arch Capital Group Ltd.": "P&C",
    "Cincinnati Financial Corporation": "P&C",
    "Erie Indemnity Company": "OTHER",
    "Principal Financial Group, Inc.": "Life & Health",
    "Willis Towers Watson Public Limited Company": "OTHER",
    "HDFC Life Insurance Company Limited": "Life & Health",
    "ICICI Prudential Life": "Life & Health",
    "LIC": "Life & Health",
    "SBI Life Insurance Company Limited": "Life & Health",
    "Aflac Incorporated": "Life & Health",
    "American International Group, Inc.": "P&C",
    "Assurant, Inc.": "P&C",
    "Arthur J. Gallagher & Co.": "OTHER",
    "The Allstate Corporation": "P&C",
    "Aon plc": "OTHER",
    "Brown & Brown, Inc.": "OTHER",
    "Chubb Limited": "P&C",
    "Cigna": "Life & Health",
    "CNA Financial": "P&C",
    "CVS Health": "Life & Health",
    "Everest Group, Ltd.": "Reinsurance",
    "Globe Life Inc.": "Life & Health",
    "The Hartford Insurance Group, Inc.": "Multiline",
    "Humana": "Life & Health",
    "Loews Corporation": "OTHER",
    "Lincoln National Corporation": "Life & Health",
    "MetLife, Inc.": "Life & Health",
    "Marsh & McLennan Companies, Inc.": "OTHER",
    "The Progressive Corporation": "P&C",
    "The Travelers Companies, Inc.": "P&C",
    "United Health": "Life & Health",
    "Unum Group": "Life & Health",
    "W. R. Berkley Corporation": "P&C",

    # Saudi Arabia (KSA)
    "The Company for Cooperative Insurance": "Multiline",
    "Aljazira Takaful Taawuni Company": "Life & Health",
    "Malath Cooperative Insurance Company": "P&C",
    "The Mediterranean and Gulf Cooperative Insurance and Reinsurance Company": "Multiline",
    "Mutakamela Insurance Company": "P&C",
    "Salama Cooperative Insurance Company": "P&C",
    "Walaa Cooperative Insurance Company": "P&C",
    "Arabian Shield Cooperative Insurance Company": "P&C",
    "Saudi Arabian Cooperative Insurance Company": "P&C",
    "Gulf Union Alahlia Cooperative Insurance Company": "P&C",
    "Allied Cooperative Insurance Group": "P&C",
    "Arabia Insurance Cooperative Company": "P&C",
    "Al-Etihad Cooperative Insurance Company": "P&C",
    "Al Sagr Cooperative Insurance Company": "P&C",
    "United Cooperative Assurance Company": "P&C",
    "Saudi Reinsurance Company": "Reinsurance",
    "Bupa Arabia for Cooperative Insurance Company": "Life & Health",
    "Al Rajhi Company for Cooperative Insurance": "Multiline",
    "Chubb Arabia Cooperative Insurance Company": "P&C",
    "Gulf Insurance Group": "Multiline",
    "Gulf General Cooperative Insurance Company": "P&C",
    "Buruj Cooperative Insurance Company": "P&C",
    "Liva Insurance Company": "P&C",
    "Wataniya Insurance Company": "Multiline",
    "Amana Cooperative Insurance Company": "P&C",
    "Saudi Enaya Cooperative Insurance Company": "P&C",

    # Greater China / SE Asia
    "AIA Group Limited": "Life & Health",
    "Ping An Insurance (Group) Company of China, Ltd.": "Multiline",
    "PICC Property & Casualty": "P&C",
    "China Life Insurance Company Limited": "Life & Health",
    "ZhongAn Online P&C Insurance": "P&C",
    "China Taiping Insurance (Life)": "Life & Health",
    "Bangkok Insurance PCL (composite)": "P&C",
    "Bangkok Life Assurance": "Life & Health",
    "Dhipaya Group Holdings": "Multiline",
    "Thai Life Insurance Public Company Limited": "Life & Health",
    "Great Eastern Holdings": "Life & Health",
    "Huaxia": "Life & Health",
    "New China Life Insurance": "Life & Health",
    "China Pacific Insurance (Group)": "Multiline",

    # Switzerland
    "Baloise Holding": "Multiline",
    "Helvetia Holding": "Multiline",
    "Swiss Life Holding AG": "Life & Health",
    "Swiss Re AG": "Reinsurance",
    "Zurich Insurance Group AG": "Multiline",

    # Japan
    "Japan Post Holdings Co., Ltd.": "OTHER",
    "Japan Post Insurance": "Life & Health",
    "Sompo Holdings, Inc.": "Multiline",
    "MS&AD Insurance Group Holdings, Inc.": "Multiline",
    "Dai-ichi Life Holdings, Inc.": "Life & Health",
    "Tokio Marine Holdings, Inc.": "Multiline",
    "T&D Holdings, Inc.": "Life & Health",

    # Canada / Taiwan / CEE / Germany
    "Manulife Financial": "Life & Health",
    "Sun Life Financial": "Life & Health",
    "Fubon Financial": "Multiline",
    "Cathay Life": "Life & Health",
    "Vienna Insurance Group": "Multiline",
    "PZU SA": "Multiline",
    "Allianz SE": "Multiline",
    "Hannover RÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¼ck SE": "Reinsurance",
    "MÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¼nchener RÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¼ckversicherungs-Gesellschaft Aktiengesellschaft in MÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¼nchen": "Reinsurance",
    "Talanx AG (HDI)": "Multiline",
}
company_type_norm = {normalize(k): v for k, v in company_type_raw.items()}

# ---------- country dictionary (YOUR LIST) ----------
_allowed_countries = {
    "Australia","Belgium","Canada","Chile","China","Denmark","France","Germany","Hong_Kong",
    "India","Italy","Japan","Luxembourg","Malaysia","Netherlands","Philippines","Saudi_Arabia",
    "Singapore","South_Korea","Switzerland","Sweden","Thailand","UAE","USA","United_Kingdom","Vietnam","Kuwait","Taiwan"
}
country_raw = {
    # Australia
    "AUB Group Limited": "Australia",
    "Challenger Limited": "Australia",
    "Helia Group Limited": "Australia",
    "Insurance Australia Group Limited": "Australia",
    "Medibank Private Limited": "Australia",
    "nib holdings limited": "Australia",
    "QBE Insurance Group Limited": "Australia",
    "Steadfast Group Limited": "Australia",
    "Suncorp Group Limited": "Australia",

    # Italy / Spain / Nordics / NL / BE / FR / Vietnam
    "Assicurazioni Generali S.p.A.": "Italy",
    "Poste Italiane S.p.A.": "Italy",
    "Unipol Assicurazioni S.p.A.": "Italy",
    "MAPFRE S.A.": "Spain",
    "Tryg A/S": "Denmark",
    "Aegon Ltd.": "Netherlands",
    "ASR Nederland N.V.": "Netherlands",
    "NN Group N.V.": "Netherlands",
    "ageas SA/NV": "Belgium",
    "AXA SA": "France",
    "Bao Viet Holdings": "Vietnam",

    # Indonesia / Malaysia / Korea
    "PT Asuransi Tugu Pratama Indonesia": "Indonesia",
    "Syarikat Takaful Malaysia": "Malaysia",
    "Hanwha General Insurance": "South_Korea",
    "Samsung Fire & Marine Insurance Co., Ltd.": "South_Korea",
    "Hyundai Marine & Fire Insurance Co., Ltd.": "South_Korea",
    "DB Insurance Co., Ltd.": "South_Korea",
    "Samsung Life Insurance Co., Ltd.": "South_Korea",
    "Hanwha Life Insurance Co., Ltd.": "South_Korea",

    # UK
    "Admiral Group plc": "United_Kingdom",
    "Aviva plc": "United_Kingdom",
    "Beazley plc": "United_Kingdom",
    "Hiscox Ltd": "United_Kingdom",
    "Legal & General Group Plc": "United_Kingdom",
    "M&G plc": "United_Kingdom",
    "Phoenix Group Holdings plc": "United_Kingdom",

    # US / Bermuda / India
    "Prudential Financial, Inc.": "USA",
    "Arch Capital Group Ltd.": "USA",
    "Cincinnati Financial Corporation": "USA",
    "Erie Indemnity Company": "USA",
    "Principal Financial Group, Inc.": "USA",
    "Willis Towers Watson Public Limited Company": "United_Kingdom",
    "HDFC Life Insurance Company Limited": "India",
    "ICICI Prudential Life": "India",
    "LIC": "India",
    "SBI Life Insurance Company Limited": "India",
    "Aflac Incorporated": "USA",
    "American International Group, Inc.": "USA",
    "Assurant, Inc.": "USA",
    "Arthur J. Gallagher & Co.": "USA",
    "The Allstate Corporation": "USA",
    "Aon plc": "United_Kingdom",
    "Brown & Brown, Inc.": "USA",
    "Chubb Limited": "Switzerland",
    "Cigna": "USA",
    "CNA Financial": "USA",
    "CVS Health": "USA",
    "Everest Group, Ltd.": "USA",
    "Globe Life Inc.": "USA",
    "The Hartford Insurance Group, Inc.": "USA",
    "Humana": "USA",
    "Loews Corporation": "USA",
    "Lincoln National Corporation": "USA",
    "MetLife, Inc.": "USA",
    "Marsh & McLennan Companies, Inc.": "USA",
    "The Progressive Corporation": "USA",
    "The Travelers Companies, Inc.": "USA",
    "United Health": "USA",
    "Unum Group": "USA",
    "W. R. Berkley Corporation": "USA",

    # Saudi Arabia
    "The Company for Cooperative Insurance": "Saudi_Arabia",
    "Aljazira Takaful Taawuni Company": "Saudi_Arabia",
    "Malath Cooperative Insurance Company": "Saudi_Arabia",
    "The Mediterranean and Gulf Cooperative Insurance and Reinsurance Company": "Saudi_Arabia",
    "Mutakamela Insurance Company": "Saudi_Arabia",
    "Salama Cooperative Insurance Company": "Saudi_Arabia",
    "Walaa Cooperative Insurance Company": "Saudi_Arabia",
    "Arabian Shield Cooperative Insurance Company": "Saudi_Arabia",
    "Saudi Arabian Cooperative Insurance Company": "Saudi_Arabia",
    "Gulf Union Alahlia Cooperative Insurance Company": "Saudi_Arabia",
    "Allied Cooperative Insurance Group": "Saudi_Arabia",
    "Arabia Insurance Cooperative Company": "Saudi_Arabia",
    "Al-Etihad Cooperative Insurance Company": "Saudi_Arabia",
    "Al Sagr Cooperative Insurance Company": "Saudi_Arabia",
    "United Cooperative Assurance Company": "Saudi_Arabia",
    "Saudi Reinsurance Company": "Saudi_Arabia",
    "Bupa Arabia for Cooperative Insurance Company": "Saudi_Arabia",
    "Al Rajhi Company for Cooperative Insurance": "Saudi_Arabia",
    "Chubb Arabia Cooperative Insurance Company": "Saudi_Arabia",
    "Gulf Insurance Group": "Kuwait",
    "Gulf General Cooperative Insurance Company": "Saudi_Arabia",
    "Buruj Cooperative Insurance Company": "Saudi_Arabia",
    "Liva Insurance Company": "Saudi_Arabia",
    "Wataniya Insurance Company": "Saudi_Arabia",
    "Amana Cooperative Insurance Company": "Saudi_Arabia",
    "Saudi Enaya Cooperative Insurance Company": "Saudi_Arabia",

    # Greater China / SE Asia
    "AIA Group Limited": "Hong_Kong",
    "Ping An Insurance (Group) Company of China, Ltd.": "China",
    "PICC Property & Casualty": "China",
    "China Life Insurance Company Limited": "China",
    "ZhongAn Online P&C Insurance": "China",
    "China Taiping Insurance (Life)": "China",
    "Bangkok Insurance PCL (composite)": "Thailand",
    "Bangkok Life Assurance": "Thailand",
    "Dhipaya Group Holdings": "Thailand",
    "Thai Life Insurance Public Company Limited": "Thailand",
    "Great Eastern Holdings": "Singapore",
    "Huaxia": "China",
    "New China Life Insurance": "China",
    "China Pacific Insurance (Group)": "China",

    # Switzerland
    "Baloise Holding": "Switzerland",
    "Helvetia Holding": "Switzerland",
    "Swiss Life Holding AG": "Switzerland",
    "Swiss Re AG": "Switzerland",
    "Zurich Insurance Group AG": "Switzerland",

    # Japan
    "Japan Post Holdings Co., Ltd.": "Japan",
    "Japan Post Insurance": "Japan",
    "Sompo Holdings, Inc.": "Japan",
    "MS&AD Insurance Group Holdings, Inc.": "Japan",
    "Dai-ichi Life Holdings, Inc.": "Japan",
    "Tokio Marine Holdings, Inc.": "Japan",
    "T&D Holdings, Inc.": "Japan",

    # Canada / Taiwan / CEE / Germany
    "Manulife Financial": "Canada",
    "Sun Life Financial": "Canada",
    "Fubon Financial": "Taiwan",
    "Cathay Life": "Taiwan",
    "Vienna Insurance Group": "Germany",
    "PZU SA": "Germany",
    "Allianz SE": "Germany",
    "Hannover RÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¼ck SE": "Germany",
    "MÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¼nchener RÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¼ckversicherungs-Gesellschaft Aktiengesellschaft in MÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¼nchen": "Germany",
    "Talanx AG (HDI)": "Germany",
}
country_norm = {normalize(k): v for k, v in country_raw.items()}

# ================== 1) PANEL (one row per Ticker) ==================
panel = pd.read_csv(PANEL_CSV)
if not {"Ticker", "Company_name"}.issubset(panel.columns):
    raise KeyError("Panel must have columns: 'Ticker', 'Company_name'.")

panel["_name_norm"]      = panel["Company_name"].apply(normalize)
panel["insurance_type"]  = panel["_name_norm"].map(company_type_norm).fillna("OTHER")
panel["Country_label"]   = panel["_name_norm"].map(country_norm).where(
    panel["_name_norm"].map(country_norm).isin(_allowed_countries), "OTHER"
)

# Keep FIRST occurrence of each Ticker (prevents panel-driven duplication),
# but this does NOT limit how many tickers we summarize (that comes from price files).
panel_one = panel[["Ticker", "Company_name", "insurance_type", "Country_label"]].drop_duplicates("Ticker", keep="first")

# ================== 2) LOAD PRICES (all files) ==================
files = sorted(Path(PRICE_DIR).glob("*_price.csv"))
if not files:
    raise RuntimeError(f"No '*_price.csv' files found in {PRICE_DIR}")

all_prices = []
for f in files:
    df = pd.read_csv(f)
    if not {"Date", "Price"}.issubset(df.columns):
        continue
    df = df[["Date", "Price"]].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    if df.empty:
        continue
    # ticker = chunk before '_price' (after last underscore)
    ticker = f.stem.replace("_price", "")
    ticker = ticker.split("_")[-1]
    df["Ticker"] = ticker
    all_prices.append(df)

if not all_prices:
    raise RuntimeError("No usable price files after parsing Date/Price.")

prices = pd.concat(all_prices, ignore_index=True)

# ================== 3) TSR + DATES/PRICES (one row per Ticker) ==================
rows = []
for t, g in prices.groupby("Ticker", sort=False):
    g = g.sort_values("Date").reset_index(drop=True)
    recent_date  = g["Date"].iloc[-1]
    recent_price = float(g["Price"].iloc[-1])

    row = {"Ticker": t, "Recent_Date": recent_date.date(), "Recent_Price": recent_price}

    for y in LOOKBACKS:
        target = recent_date - DateOffset(years=y)
        past   = g[g["Date"] <= target]
        if past.empty or float(past["Price"].iloc[-1]) == 0:
            row[f"Price_{y}Y_Date"] = None
            row[f"Price_{y}Y"]      = None
            row[f"TSR_{y}Y"]        = None
        else:
            past_date  = past["Date"].iloc[-1]
            past_price = float(past["Price"].iloc[-1])
            row[f"Price_{y}Y_Date"] = past_date.date()
            row[f"Price_{y}Y"]      = past_price
            row[f"TSR_{y}Y"]        = (recent_price / past_price) ** (1 / y) - 1.0

    rows.append(row)

tsr = pd.DataFrame(rows)

# ================== 4) MERGE LABELS (DOES NOT REDUCE ROWS) ==================
# LEFT-merge keeps ALL tickers found in price files; panel rows only add metadata.
tsr = tsr.merge(panel_one, on="Ticker", how="left")

# ================== 5) ORDER + SAVE ==================
cols = ["Company_name", "Ticker", "insurance_type", "Country_label",
        "Recent_Date", "Recent_Price"]
for y in LOOKBACKS:
    cols += [f"Price_{y}Y_Date", f"Price_{y}Y", f"TSR_{y}Y"]

# Keep any missing label columns if merge didn't find a panel row
for c in cols:
    if c not in tsr.columns:
        tsr[c] = None
tsr = tsr[cols]

Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
tsr.to_csv(OUTPUT_CSV, index=False)

# ================== 6) QUICK DIAGNOSTICS ==================
uniq_files   = len(files)
uniq_tickers = tsr["Ticker"].nunique()
print(f"Files scanned: {uniq_files}")
print(f"Unique tickers summarized: {uniq_tickers}")
print(f"Saved -> {OUTPUT_CSV}")
print(tsr.head(12))
