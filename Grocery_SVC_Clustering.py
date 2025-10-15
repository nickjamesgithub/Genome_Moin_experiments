# --- Imports ---
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
import scipy.cluster.hierarchy as sch

# --- Load & filter to Healthcare ---
df = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\grocery_data.csv")

# --- Tickers (updated list) ---
tickers = ["WM","AMZN","COST","TGT","KR","ACI","BJ","SFM","GO","WMK","IMKT.A","CASY","VLGE.A",
           "NGVC","DG","L","MRU","EMP.A","ATD","NWC","TSCO","SBRY","MKS","OCDO","BME","AD",
           "CA","COLR","JMT","SLIGR","AXFO","KESKOB","DNP","EUR","B4B","WOW","COL","MTS",
           "WHS","3382","AEON","3038","9948","8198","8194","A139480","A282330","A007070",
           "D01","OV8"]

# --- Subset (2015–2024 & tickers) ---
df_sel = df[df["Year"].between(2019, 2024) & df["Ticker"].isin(tickers)]

# --- Pivot helper (handles duplicates via mean) ---
def piv(metric):
    cols = [t for t in tickers if t in df_sel["Ticker"].unique()]
    return (df_sel.pivot_table(index="Year", columns="Ticker", values=metric, aggfunc="mean")
                 .reindex(range(2019, 2025))
                 [cols])

# --- L1 distance over overlapping years (ignores NaNs) ---
def l1_ignore_nan(mat: pd.DataFrame) -> pd.DataFrame:
    X = mat.to_numpy(float).T                      # n_tickers x n_years
    valid = (~np.isnan(X))[:, None, :] & (~np.isnan(X))[None, :, :]
    diff = np.abs(np.nan_to_num(X)[:, None, :] - np.nan_to_num(X)[None, :, :])
    D = (diff * valid).sum(2)
    return pd.DataFrame(D, index=mat.columns, columns=mat.columns)

# --- Affinity from distance (scaled to [0,1], diag=1) ---
def to_affinity(D: pd.DataFrame) -> pd.DataFrame:
    m = D.to_numpy().max() or 1.0
    A = 1 - (D / m)
    np.fill_diagonal(A.values, 1.0)
    return A

# --- Build metric matrices ---
eva = piv("EVA_ratio_bespoke")
rev = piv("Revenue_growth_3_f")

# --- Distances ---
eva_dist = l1_ignore_nan(eva)
rev_dist = l1_ignore_nan(rev)

# --- Affinities + stack ---
eva_affinity = to_affinity(eva_dist)
rev_affinity = to_affinity(rev_dist)
stacked_affinity = eva_affinity + rev_affinity

# --- Map labels to Company_name (fallback to ticker) ---
name_map = (df[df["Ticker"].isin(stacked_affinity.index)]
            .sort_values("Year")
            .groupby("Ticker")["Company_name"].last())
labels = name_map.reindex(stacked_affinity.index)
labels = labels.where(labels.notna(), stacked_affinity.index)
stacked_affinity.index = stacked_affinity.columns = labels.values

# --- Your dendrogram function (unchanged) ---
def dendrogram_plot(matrix, distance_measure, data_generation, labels):
    plt.rcParams.update({'font.size': 20})
    fig = pylab.figure(figsize=(15,10))
    axdendro = fig.add_axes([0.09,0.1,0.2,0.8])
    Y = sch.linkage(matrix, method='centroid')
    Z = sch.dendrogram(Y, orientation='right', labels=labels,
                       leaf_rotation=360, leaf_font_size=9)
    axdendro.set_xticks([])

    axmatrix = fig.add_axes([0.3,0.1,0.6,0.8])
    index = Z['leaves']
    D = matrix[index,:]
    D = D[:,index]
    im = axmatrix.matshow(D, aspect='auto', origin='lower')
    axmatrix.set_xticks([]); axmatrix.set_yticks([])

    axcolor = fig.add_axes([0.91,0.1,0.02,0.8])
    pylab.colorbar(im, cax=axcolor)

    plt.savefig(data_generation+distance_measure+"Dendrogram", bbox_inches="tight")
    plt.close(fig)

# --- Run plot on stacked_affinity ---
dendrogram_plot(stacked_affinity.to_numpy(), "StackedAffinity_", "Grocery_", stacked_affinity.index.tolist())

x=1
y=2