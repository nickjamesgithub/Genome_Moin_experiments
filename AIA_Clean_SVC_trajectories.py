import pandas as pd
from pathlib import Path

# ------- PATHS -------
in_path  = r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\SVC_analysis\Insurance_SVC_data_.csv"
out_path = str(Path(in_path).with_name(Path(in_path).stem + "_CLEAN.csv"))

# ------- READ -------
# cp1252 handles Windows-1252 characters (where the x96/en-dash issue comes from)
df = pd.read_csv(in_path, encoding="cp1252")

# ------- NORMALIZE HEADERS & DROP UNNAMED -------
df.columns = df.columns.str.replace(r"\s+", " ", regex=True).str.strip()
df = df.loc[:, ~df.columns.str.match(r"Unnamed", na=False)]

# Column names we expect after normalization
COL_COMPANY   = "Company" if "Company" in df.columns else "Company                           "
COL_YEAR      = "Year"
COL_ROE_TXT   = "Operating ROE"
COL_ROE_CLEAN = "Operating ROE Clean"
COL_GWP_TXT   = "GWP Growth (YoY)"
COL_GWP_CLEAN = "GWP Growth Clean"

# ------- CLEANING HELPERS -------
def clean_percent(series: pd.Series) -> pd.Series:
    """
    Convert messy percent strings to numeric FRACTIONS.
    Examples:
      '–4%' -> -0.04
      '-0.3%' -> -0.003
      '11%' -> 0.11
      '0.059' -> 0.059 (already fraction)
    """
    raw = series.astype(str)

    # Detect rows that explicitly contain a % BEFORE we strip it,
    # so we know we must divide those by 100 (handles sub-1% like -0.3% correctly).
    has_pct = raw.str.contains("%", regex=False)

    s = (
        raw
        .str.replace("x96", "-", regex=False)  # cp1252 artifact -> minus
        .str.replace(r"[\u2012\u2013\u2014\u2015\u2212]", "-", regex=True)  # normalize unicode dashes/minus to '-'
        .str.replace("~", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("\u00A0", " ", regex=False)  # NBSP
        .str.strip()
        .str.replace(r"[^0-9.\-]+", "", regex=True)  # keep only digits, dot, minus
    )

    nums = pd.to_numeric(s, errors="coerce")

    # If the original cell had a % sign, ALWAYS divide by 100 (handles 0.3% -> 0.003 correctly)
    nums = nums.where(~has_pct, nums / 100.0)

    # Also catch obvious percent values that lacked a % sign (e.g., '11' or '-4')
    nums = nums.where(nums.abs() <= 1.5, nums / 100.0)

    return nums

# ------- FORCE CLEAN COLUMNS TO NUMERIC FRACTIONS -------
# Operating ROE Clean
if COL_ROE_CLEAN in df.columns:
    roe_clean = clean_percent(df[COL_ROE_CLEAN])
    # Fallback from the display column where needed
    if COL_ROE_TXT in df.columns:
        roe_clean = roe_clean.fillna(clean_percent(df[COL_ROE_TXT]))
    df[COL_ROE_CLEAN] = roe_clean.astype("float64")

# GWP Growth Clean
if COL_GWP_CLEAN in df.columns:
    gwp_clean = clean_percent(df[COL_GWP_CLEAN])
    if COL_GWP_TXT in df.columns:
        gwp_clean = gwp_clean.fillna(clean_percent(df[COL_GWP_TXT]))
    df[COL_GWP_CLEAN] = gwp_clean.astype("float64")

# Keep your original text columns unchanged so you can eyeball them in Excel
# (we are not rewriting COL_ROE_TXT / COL_GWP_TXT here)

# ------- QUICK VALIDATION -------
print("\n[DTYPES]")
for c in [COL_ROE_CLEAN, COL_GWP_CLEAN]:
    if c in df.columns:
        print(c, "->", df[c].dtype)

# Sanity check the tricky rows (sub-1% and negatives)
check = df[
    ((df[COL_GWP_CLEAN].abs() > 1.5) | (df[COL_ROE_CLEAN].abs() > 1.5))  # likely 100x issues
    | df[COL_GWP_CLEAN].isna()
    | df[COL_ROE_CLEAN].isna()
]
if not check.empty:
    print("\n[CHECK THESE ROWS] Possible scaling/parse issues:")
    cols_to_show = [COL_COMPANY, COL_YEAR, COL_GWP_TXT, COL_GWP_CLEAN, COL_ROE_TXT, COL_ROE_CLEAN]
    print(check[cols_to_show].head(20).to_string(index=False))

# Example: confirm Unipol 2015 is -0.04
if COL_COMPANY in df.columns and COL_YEAR in df.columns:
    m = df[COL_COMPANY].str.contains("Unipol", case=False, na=False) & (df[COL_YEAR] == 2015)
    if m.any():
        print("\n[UNIPOL 2015] GWP Growth Clean =", df.loc[m, COL_GWP_CLEAN].tolist())

# ------- SAVE -------
df.to_csv(out_path, index=False, encoding="utf-8")
print(f"\n[SAVED] {out_path}")
