import pandas as pd
import unicodedata
from pathlib import Path

# ---------- 1) File paths ----------
in_path  = Path(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\insurance_data_clean_250925.csv")
out_path = in_path.with_name(in_path.stem + "_with_type.csv")

# ---------- 2) Helper: normalize strings for robust matching ----------
def normalize(s: str) -> str:
    if pd.isna(s):
        return ""
    # Lowercase, strip spaces, remove accents, collapse inner spaces
    s0 = unicodedata.normalize("NFKD", str(s))
    s1 = "".join(ch for ch in s0 if not unicodedata.combining(ch))
    s2 = " ".join(s1.split()).strip().lower()
    return s2

# ---------- 3) Full mapping for every company you listed ----------
# Labels: "Life & Health", "P&C", "Reinsurance", "Multiline", "OTHER"
company_type_raw = {
    # Australia
    "AUB Group Limited": "OTHER",                    # broker/agency network
    "Challenger Limited": "Life & Health",           # annuities / life
    "Helia Group Limited": "P&C",                    # mortgage insurance
    "Insurance Australia Group Limited": "P&C",
    "Medibank Private Limited": "Life & Health",
    "nib holdings limited": "Life & Health",
    "QBE Insurance Group Limited": "P&C",
    "Steadfast Group Limited": "OTHER",              # broker/agency network
    "Suncorp Group Limited": "P&C",                  # life divested; general insurance core

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
    "Syarikat Takaful Malaysia": "Life & Health",    # family takaful focus at group ticker
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

    # US / Bermuda
    "Prudential Financial, Inc.": "Life & Health",
    "Arch Capital Group Ltd.": "P&C",                # insurance + reinsurance + mortgage; no life
    "Cincinnati Financial Corporation": "P&C",
    "Erie Indemnity Company": "OTHER",               # mgmt services/agency for reciprocal
    "Principal Financial Group, Inc.": "Life & Health",
    "Willis Towers Watson Public Limited Company": "OTHER",  # broker/advisory
    "HDFC Life Insurance Company Limited": "Life & Health",
    "ICICI Prudential Life": "Life & Health",
    "LIC": "Life & Health",
    "SBI Life Insurance Company Limited": "Life & Health",
    "Aflac Incorporated": "Life & Health",
    "American International Group, Inc.": "P&C",     # group now P&C-focused (life via Corebridge spin)
    "Assurant, Inc.": "P&C",
    "Arthur J. Gallagher & Co.": "OTHER",            # broker
    "The Allstate Corporation": "P&C",
    "Aon plc": "OTHER",                              # broker
    "Brown & Brown, Inc.": "OTHER",                  # broker
    "Chubb Limited": "P&C",
    "Cigna": "Life & Health",
    "CNA Financial": "P&C",
    "CVS Health": "Life & Health",                   # Aetna health benefits
    "Everest Group, Ltd.": "Reinsurance",
    "Globe Life Inc.": "Life & Health",
    "The Hartford Insurance Group, Inc.": "Multiline", # P&C + group benefits
    "Humana": "Life & Health",
    "Loews Corporation": "OTHER",                    # conglomerate; owns CNA
    "Lincoln National Corporation": "Life & Health",
    "MetLife, Inc.": "Life & Health",
    "Marsh & McLennan Companies, Inc.": "OTHER",     # broker/professional services
    "The Progressive Corporation": "P&C",
    "The Travelers Companies, Inc.": "P&C",
    "United Health": "Life & Health",
    "Unum Group": "Life & Health",
    "W. R. Berkley Corporation": "P&C",

    # Saudi Arabia (KSA) — cooperative insurers
    "The Company for Cooperative Insurance": "Multiline",  # Tawuniya: general, health, protection & savings
    "Aljazira Takaful Taawuni Company": "Life & Health",   # family takaful focus
    "Malath Cooperative Insurance Company": "P&C",
    "The Mediterranean and Gulf Cooperative Insurance and Reinsurance Company": "Multiline",  # MEDGULF: general/health/P&S
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
    "Bupa Arabia for Cooperative Insurance Company": "Life & Health",  # health insurer
    "Al Rajhi Company for Cooperative Insurance": "Multiline",         # general, health, P&S
    "Chubb Arabia Cooperative Insurance Company": "P&C",
    "Gulf Insurance Group": "Multiline",            # regional group life + non-life
    "Gulf General Cooperative Insurance Company": "P&C",
    "Buruj Cooperative Insurance Company": "P&C",
    "Liva Insurance Company": "P&C",                # formerly Gulf Union AlAhlia; general/medical lines
    "Wataniya Insurance Company": "Multiline",      # general, medical, protection & savings
    "Amana Cooperative Insurance Company": "P&C",
    "Saudi Enaya Cooperative Insurance Company": "P&C",

    # Greater China / SE Asia
    "AIA Group Limited": "Life & Health",
    "Ping An Insurance (Group) Company of China, Ltd.": "Multiline",
    "PICC Property & Casualty": "P&C",
    "China Life Insurance Company Limited": "Life & Health",
    "ZhongAn Online P&C Insurance": "P&C",
    "China Taiping Insurance (Life)": "Life & Health",
    "Bangkok Insurance PCL (composite)": "P&C",     # BKI is non-life
    "Bangkok Life Assurance": "Life & Health",
    "Dhipaya Group Holdings": "Multiline",          # group across non-life + life
    "Thai Life Insurance Public Company Limited": "Life & Health",
    "Great Eastern Holdings": "Life & Health",
    "Huaxia": "Life & Health",                      # Huaxia Life
    "New China Life Insurance": "Life & Health",
    "China Pacific Insurance (Group)": "Multiline",

    # Switzerland
    "Baloise Holding": "Multiline",
    "Helvetia Holding": "Multiline",
    "Swiss Life Holding AG": "Life & Health",
    "Swiss Re AG": "Reinsurance",
    "Zurich Insurance Group AG": "Multiline",

    # Japan
    "Japan Post Holdings Co., Ltd.": "OTHER",       # postal/logistics/financial holding
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
    "Hannover RÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¼ck SE": "Reinsurance",  # encoding variant
    "MÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¼nchener RÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¼ckversicherungs-Gesellschaft Aktiengesellschaft in MÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¼nchen": "Reinsurance",
    "Talanx AG (HDI)": "Multiline",
}

# ---------- 4) Build normalized lookup ----------
company_type_norm = {normalize(k): v for k, v in company_type_raw.items()}

# ---------- 5) Load data ----------
df = pd.read_csv(in_path)

# Ensure we use the exact column name you mentioned
col = "Company_name"
if col not in df.columns:
    raise KeyError(f"Expected column '{col}' not found. Available columns: {list(df.columns)}")

# ---------- 6) Map to insurance_type with safe fallback ----------
df["_key_norm"] = df[col].apply(normalize)
df["insurance_type"] = df["_key_norm"].map(company_type_norm).fillna("OTHER")

# ---------- 7) (Optional) surface any items that fell to OTHER so you can spot-check ----------
unmapped = sorted(df.loc[df["insurance_type"] == "OTHER", col].unique())
if unmapped:
    print("Heads up — these names mapped to OTHER (check spelling or add to mapping):")
    for name in unmapped:
        print("  -", name)

# ---------- 8) Add Country_label (MINIMAL ADDITION) ----------
# Allowed countries (as provided)
_allowed_countries = {
    "Australia","Belgium","Canada","Chile","China","Denmark","France","Germany","Hong_Kong",
    "India","Italy","Japan","Luxembourg","Malaysia","Netherlands","Philippines","Saudi_Arabia",
    "Singapore","South_Korea","Switzerland","Sweden","Thailand","UAE","USA","United_Kingdom"
}

# Company -> listing/home market mapping (may include values outside the allowed set; those will be coerced to "OTHER")
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

    # US / Bermuda
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

# Map to raw country, then coerce to allowed list
df["Country_label"] = df["_key_norm"].map(country_norm)
df["Country_label"] = df["Country_label"].where(df["Country_label"].isin(_allowed_countries), "OTHER")

# ---------- 9) Save ----------
df.drop(columns=["_key_norm"], inplace=True)
df.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\Clean\Clean_insurance_data_250925.csv", index=False)
print(f"Saved with insurance_type → {out_path}")

x=1
y=2
