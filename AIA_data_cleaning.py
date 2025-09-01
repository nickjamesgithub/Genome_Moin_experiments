import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import bootstrap

# ---------- Paths ----------
panel_path = r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\Insurance_data\gwp_roe_panel_with_ifrs_UPDATED.csv"
bespoke_dir = Path(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\bespoke")

out_merged        = Path(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\Insurance_data\merged_insurance_data.csv")
out_growth_1y     = Path(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\Insurance_data\median_gwp_growth_1y.csv")
out_growth_3y     = Path(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\Insurance_data\median_gwp_growth_3y.csv")
out_roe_summary   = Path(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\Insurance_data\median_roe.csv")

# ---------- Tickers ----------
tickers_raw = """
1299
2628
2318
601601
601336
966
PRU
MFC
SLF
G07
8750
8795
7181
HDFCLIFE
ICICIPRULI
SBILIFE
A032830
A088350
TLI
BLA
AGS
PHNX
LGEN
PRU
AFL
2328
6060
CS
ZURN
CB
QBE
IAG
SUN
A005830
A001450
A000060
8725
8725
AIG
TRV
CNA
ADM
AV.
HSX
BEZ
TLX
HELN
BALN
CS
ALV
G
966
8725
8630
8766
AV.
MET
AGN
MAP
BVH
BKIH
TIPH
TAKAFUL
A000810
A000370
TUGU
CINF
LNC
PFG
PZU
VIG
ALL
PGR
UNM
HIG
2328
LICI
600015
2882
2881
MPL
CI
UNH
HUM
CVS
"""

# ---------- Helpers ----------
def to_number(x):
    """Convert strings with commas/currency and (negatives) to float."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s in {"", "-", "NA", "N/A"}:
        return np.nan
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()").replace(",", "")
    for ch in ["$", "€", "£", "¥", "₩", "₫", "₦", "₹", "₨"]:
        s = s.replace(ch, "")
    try:
        val = float(s)
    except Exception:
        val = pd.to_numeric(s, errors="coerce")
    if neg and pd.notna(val):
        val = -val
    return val

def bootstrap_ci(data, statfunc=np.median, n_resamples=2000, conf=0.95):
    """Return (statistic, CI_low, CI_high) via bootstrap on a 1-D array-like."""
    data = pd.Series(data).dropna().values
    if len(data) < 2:
        return np.nan, np.nan, np.nan
    res = bootstrap((data,), statfunc, n_resamples=n_resamples,
                    confidence_level=conf, method="basic", random_state=0)
    return statfunc(data), res.confidence_interval.low, res.confidence_interval.high

# ---------- 1) Read & clean PANEL ----------
df = pd.read_csv(panel_path)

df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
df["GWP_BCN"] = df["GWP"].apply(to_number).astype(float)
df["ROE_BCN"] = df["ROE"].apply(to_number).astype(float)
df["IFRS_change_BCN"] = pd.to_numeric(df["IFRS_change"], errors="coerce").fillna(0).astype(int)

# Sort for time-series ops
df = df.sort_by_values = df.sort_values(["Insurer", "Year"]).reset_index(drop=True)

# 1y GWP growth (on cleaned numeric GWP)
df["GWP_growth_BCN"] = df.groupby("Insurer")["GWP_BCN"].pct_change(fill_method=None)
df["GWP_growth_BCN"].replace([np.inf, -np.inf], np.nan, inplace=True)

# 3y rolling arithmetic mean of growth (per Insurer, need 3 periods)
df["GWP_growth_3y"] = (
    df.groupby("Insurer")["GWP_growth_BCN"]
      .transform(lambda s: s.rolling(window=3, min_periods=3).mean())
)

# Keep the panel columns for merging
df_panel = df[[
    "Insurer", "Type", "Home_Country", "Year",
    "GWP_BCN", "ROE_BCN", "IFRS_change_BCN",
    "GWP_growth_BCN", "GWP_growth_3y"
]].copy()

# ---------- 2) Load all BESPOKE files ----------
seen, tickers = set(), []
for raw in [ln.strip() for ln in tickers_raw.splitlines() if ln.strip()]:
    t = raw.rstrip(".")
    if t not in seen:
        tickers.append(t); seen.add(t)

frames, missing = [], []
for tkr in tickers:
    fpath1 = bespoke_dir / f"_{tkr}.csv"
    fpath2 = bespoke_dir / f"{tkr}.csv"
    if fpath1.exists():
        fpath = fpath1
    elif fpath2.exists():
        fpath = fpath2
    else:
        missing.append(tkr)
        continue
    try:
        tmp = pd.read_csv(fpath)
        tmp["Ticker"] = tkr
        frames.append(tmp)
    except Exception as e:
        missing.append(f"{tkr} (error reading {fpath.name}: {e})")

if not frames:
    raise FileNotFoundError("No bespoke CSVs were matched. Check names like _TICKER.csv / TICKER.csv.")

bespoke_all = pd.concat(frames, ignore_index=True)

# ---------- 3) Clean BESPOKE ----------
if "Year" in bespoke_all.columns:
    bespoke_all["Year"] = pd.to_numeric(bespoke_all["Year"], errors="coerce").astype("Int64")

company_col = "Company_name"  # confirmed

# ---------- 4) Merge PANEL onto BESPOKE ----------
left_keys = [company_col]
right_keys = ["Insurer"]
if "Year" in bespoke_all.columns:
    left_keys.append("Year")
    right_keys.append("Year")

merged = bespoke_all.merge(
    df_panel,
    how="left",
    left_on=left_keys,
    right_on=right_keys,
    suffixes=("", "_panel")
)

# ---------- 5) Save merged ----------
merged.to_csv(out_merged, index=False)
print(f"Saved merged dataset -> {out_merged}")

# ---------- 6) Trim extremes (1–99%) on PANEL for robust summaries ----------
# (Using panel df avoids double-counting companies due to bespoke stacking)
# GWP 1y growth
g1_lo, g1_hi = df["GWP_growth_BCN"].quantile([0.01, 0.99])
df_g1_trim = df[df["GWP_growth_BCN"].between(g1_lo, g1_hi)]

# GWP 3y growth
g3_lo, g3_hi = df["GWP_growth_3y"].quantile([0.01, 0.99])
df_g3_trim = df[df["GWP_growth_3y"].between(g3_lo, g3_hi)]

# ROE
roe_nonan = df["ROE_BCN"].dropna()
if len(roe_nonan) > 0:
    r_lo, r_hi = roe_nonan.quantile([0.01, 0.99])
else:
    r_lo, r_hi = -np.inf, np.inf
df_roe_trim = df[df["ROE_BCN"].between(r_lo, r_hi)]

# ---------- 7) Yearly summaries (Median + 95% bootstrap CI) ----------
# 1y GWP growth
gwp_1y_summary = []
for yr, grp in df_g1_trim.groupby("Year"):
    med, low, high = bootstrap_ci(grp["GWP_growth_BCN"], statfunc=np.median)
    gwp_1y_summary.append({"Year": yr, "Median": med, "CI_low": low, "CI_high": high})
gwp_1y_summary = pd.DataFrame(gwp_1y_summary).sort_values("Year")
gwp_1y_summary.to_csv(out_growth_1y, index=False)
print(f"Saved 1y GWP growth summary -> {out_growth_1y}")

# 3y GWP growth
gwp_3y_summary = []
for yr, grp in df_g3_trim.groupby("Year"):
    med, low, high = bootstrap_ci(grp["GWP_growth_3y"], statfunc=np.median)
    gwp_3y_summary.append({"Year": yr, "Median": med, "CI_low": low, "CI_high": high})
gwp_3y_summary = pd.DataFrame(gwp_3y_summary).sort_values("Year")
gwp_3y_summary.to_csv(out_growth_3y, index=False)
print(f"Saved 3y GWP growth summary -> {out_growth_3y}")

# ROE
roe_summary = []
for yr, grp in df_roe_trim.groupby("Year"):
    med, low, high = bootstrap_ci(grp["ROE_BCN"], statfunc=np.median)
    roe_summary.append({"Year": yr, "Median": med, "CI_low": low, "CI_high": high})
roe_summary = pd.DataFrame(roe_summary).sort_values("Year")
roe_summary.to_csv(out_roe_summary, index=False)
print(f"Saved ROE summary -> {out_roe_summary}")

# ---------- 8) Plots ----------
# A) 1y vs 3y GWP growth (side-by-side)
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# 1y
axes[0].plot(gwp_1y_summary["Year"], gwp_1y_summary["Median"], marker="o", color="tab:blue", label="Median (1y)")
axes[0].fill_between(gwp_1y_summary["Year"], gwp_1y_summary["CI_low"], gwp_1y_summary["CI_high"],
                     alpha=0.2, color="tab:blue", label="95% CI")
axes[0].set_title("1-Year GWP Growth (Median + 95% CI, Trimmed 1–99%)")
axes[0].set_xlabel("Year"); axes[0].set_ylabel("GWP Growth")
axes[0].grid(True); axes[0].legend()

# 3y
axes[1].plot(gwp_3y_summary["Year"], gwp_3y_summary["Median"], marker="o", color="tab:orange", label="Median (3y)")
axes[1].fill_between(gwp_3y_summary["Year"], gwp_3y_summary["CI_low"], gwp_3y_summary["CI_high"],
                     alpha=0.2, color="tab:orange", label="95% CI")
axes[1].set_title("3-Year Rolling GWP Growth (Median + 95% CI, Trimmed 1–99%)")
axes[1].set_xlabel("Year")
axes[1].grid(True); axes[1].legend()

plt.tight_layout()
plt.show()

# B) ROE (single figure)
plt.figure(figsize=(10, 6))
plt.plot(roe_summary["Year"], roe_summary["Median"], marker="o", color="tab:red", label="Median ROE")
plt.fill_between(roe_summary["Year"], roe_summary["CI_low"], roe_summary["CI_high"],
                 alpha=0.2, color="tab:red", label="95% CI")
plt.title("ROE Over Time (Median + 95% CI, Trimmed 1–99%)")
plt.xlabel("Year"); plt.ylabel("ROE")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()
