# =========================
# Affinity + Dendrogram on:
#   - Operating ROE Clean
#   - GWP Growth Clean (3Y rolling; valid from 2017 only)
#   - Remove IAG, QBE, and Suncorp
# =========================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import pylab
import sys

# --- Load data ---
df = pd.read_csv(
    r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\SVC_analysis\Insurance_SVC_data__CLEAN.csv"
)

# --- Numeric parser (handles %, dashes, tildes, spaces) ---
def _to_float(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    s = (s.replace("−", "-").replace("–", "-").replace("—", "-")
           .replace("~", "").replace(" ", ""))
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except Exception:
            return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

# --- Working copy + ensure key columns exist BEFORE anything else ---
df_hc = df.copy()
if "Ticker" not in df_hc.columns:
    df_hc["Ticker"] = df_hc.get("Company", df_hc.index.astype(str))
if "Company_name" not in df_hc.columns:
    df_hc["Company_name"] = df_hc.get("Company", df_hc.get("Ticker", "Unknown"))

# --- Remove IAG, QBE, Suncorp ---
drop_list = ["Insurance Australia Group (IAG)", "QBE Insurance Group", "Suncorp Group Ltd."]
df_hc = df_hc[~df_hc["Company"].str.strip().isin(drop_list)].copy()

# --- Basic cleaning ---
df_hc["Year"] = pd.to_numeric(df_hc["Year"], errors="coerce").astype("Int64")
df_hc["Operating ROE Clean"] = df_hc["Operating ROE Clean"].map(_to_float)
df_hc["GWP Growth Clean"]    = df_hc["GWP Growth Clean"].map(_to_float)

# --- Compute 3Y rolling mean for GWP Growth per company; start at 2017 only ---
df_hc = df_hc.sort_values(["Ticker", "Year"])
df_hc["GWP Growth Clean 3Y"] = (
    df_hc.groupby("Ticker")["GWP Growth Clean"]
         .transform(lambda x: x.rolling(window=3, min_periods=3).mean())
)

# --- Restrict analysis window ---
df_hc = df_hc[df_hc["Year"].between(2015, 2024)].copy()

# --- Universe of tickers ---
tickers = sorted(df_hc["Ticker"].dropna().unique().tolist())
if len(tickers) < 2:
    raise SystemExit("Need at least 2 tickers in the dataset/years 2015–2024 to compute a dendrogram.")

# --- Pivot helper ---
def piv(frame: pd.DataFrame, metric: str, tickers_universe) -> pd.DataFrame:
    present = [t for t in tickers_universe if t in frame["Ticker"].unique()]
    mat = (
        frame.pivot_table(index="Year", columns="Ticker", values=metric, aggfunc="mean")
             .reindex(range(2015, 2025))  # 2015..2024 inclusive
    )
    return mat[present]

# --- L1 distance over overlapping years (ignores NaNs) ---
def l1_ignore_nan(mat: pd.DataFrame) -> pd.DataFrame:
    X = mat.to_numpy(dtype=float).T
    if X.size == 0:
        return pd.DataFrame(index=mat.columns, columns=mat.columns, dtype=float)
    valid = (~np.isnan(X))[:, None, :] & (~np.isnan(X))[None, :, :]
    overlap = valid.sum(axis=2)
    diff = np.abs(np.nan_to_num(X)[:, None, :] - np.nan_to_num(X)[None, :, :])
    D = (diff * valid).sum(axis=2)
    D = np.where(overlap > 0, D, np.nan)
    return pd.DataFrame(D, index=mat.columns, columns=mat.columns)

# --- Distance -> Affinity ---
def to_affinity(D: pd.DataFrame) -> pd.DataFrame:
    if D.empty:
        return D.copy()
    max_d = np.nanmax(D.to_numpy())
    if not np.isfinite(max_d) or max_d == 0:
        max_d = 1.0
    D_filled = D.fillna(max_d)
    A = 1.0 - (D_filled / max_d)
    np.fill_diagonal(A.values, 1.0)
    return A

# --- Build metric matrices ---
roe  = piv(df_hc, "Operating ROE Clean", tickers)
gwp3 = piv(df_hc, "GWP Growth Clean 3Y", tickers)

# Drop firms with no data in each metric
roe  = roe.dropna(axis=1, how="all")
gwp3 = gwp3.dropna(axis=1, how="all")

# Keep only firms present in BOTH metrics
common_cols = sorted(set(roe.columns).intersection(set(gwp3.columns)))
roe  = roe[common_cols]
gwp3 = gwp3[common_cols]

if roe.shape[1] < 2 or gwp3.shape[1] < 2:
    raise SystemExit("Not enough overlapping data across tickers for both metrics (need >= 2 firms with valid ROE and 3Y GWP).")

# --- Distances ---
roe_dist  = l1_ignore_nan(roe)
gwp3_dist = l1_ignore_nan(gwp3)

if roe_dist.empty or gwp3_dist.empty:
    raise SystemExit("Empty distance matrix. Check data availability in the selected years.")

# --- Affinities + stack ---
roe_affinity  = to_affinity(roe_dist)
gwp3_affinity = to_affinity(gwp3_dist)

common_final = sorted(set(roe_affinity.columns).intersection(set(gwp3_affinity.columns)))
roe_affinity  = roe_affinity.loc[common_final, common_final]
gwp3_affinity = gwp3_affinity.loc[common_final, common_final]

stacked_affinity = roe_affinity + gwp3_affinity

if stacked_affinity.shape[0] < 2:
    raise SystemExit("Not enough companies to plot after affinity stacking (need >= 2).")

# --- Map labels ---
name_map = (
    df_hc[df_hc["Ticker"].isin(stacked_affinity.index)]
      .sort_values("Year")
      .groupby("Ticker")["Company_name"]
      .last()
)
labels_series = name_map.reindex(stacked_affinity.index)
labels_series = labels_series.where(labels_series.notna(), stacked_affinity.index)
labels_list = labels_series.values.tolist()
stacked_affinity.index = stacked_affinity.columns = labels_list

# --- Dendrogram plot ---
def dendrogram_plot(matrix, distance_measure, data_generation, labels):
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError("Affinity matrix must be square and at least 2x2.")
    plt.rcParams.update({"font.size": 20})
    fig = pylab.figure(figsize=(15, 10))

    # Dendrogram
    axdendro = fig.add_axes([0.09, 0.1, 0.2, 0.8])
    Y = sch.linkage(matrix, method="centroid")
    Z = sch.dendrogram(Y, orientation="right", labels=labels, leaf_rotation=360, leaf_font_size=9)
    axdendro.set_xticks([])

    # Heatmap
    axmatrix = fig.add_axes([0.3, 0.1, 0.6, 0.8])
    idx = Z["leaves"]
    D = matrix[idx, :][:, idx]
    im = axmatrix.matshow(D, aspect="auto", origin="lower")
    axmatrix.set_xticks([]); axmatrix.set_yticks([])

    # Colorbar
    axcolor = fig.add_axes([0.91, 0.1, 0.02, 0.8])
    pylab.colorbar(im, cax=axcolor)

    out_path = f"{data_generation}{distance_measure}Dendrogram.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return out_path

# --- Run ---
outfile = dendrogram_plot(
    stacked_affinity.to_numpy(),
    distance_measure="StackedAffinity_ROEplusGWP3Y_",
    data_generation="",
    labels=labels_list
)
print(f"Dendrogram saved to: {outfile}")

# anchors
x = 1
y = 2
