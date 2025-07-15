import numpy as np
import pandas as pd
import statsmodels.api as sm
import pylab
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def learn_regression_coefficient(y):

    # Fit regression model: y ~ beta_0 + time

    # Generate design matrix
    time_grid = np.linspace(1, len(y), len(y))
    design_matrix = sm.tools.add_constant(time_grid)

    # Fit Model
    ols_model = sm.OLS(y, design_matrix)
    model_results = ols_model.fit()

    # Get regression coefficients
    params = model_results.params[1]

    return params

def compute_stock_metrics(df):
    adjusted_close = df[["Date", "Adj Close"]]
    log_returns = np.log(df["Adj Close"]) - np.log(df["Adj Close"].shift(1))

    # Compute average returns, volatility, sharpe ratio
    average_daily_return = np.mean(log_returns)
    volatility = np.std(log_returns)
    sharpe_ratio = average_daily_return/volatility * np.sqrt(252)

    # Total shareholder return
    tsr = (df["Adj Close"].iloc[-1]/df["Adj Close"].iloc[0]) - 1

    return tsr, sharpe_ratio

def dendrogram_plot(matrix, distance_measure, data_generation, labels):

    # Compute and plot dendrogram.
    plt.rcParams.update({'font.size': 20})
    fig = pylab.figure(figsize=(15,10))
    axdendro = fig.add_axes([0.09,0.1,0.2,0.8]) # 0.1, 0.1, 0.2, 0.8
    Y = sch.linkage(matrix, method='centroid')
    Z = sch.dendrogram(Y, orientation='right', labels=labels, leaf_rotation=360, leaf_font_size=5)
    # Z = sch.dendrogram(Y, orientation='right', leaf_rotation=360, leaf_font_size=18, no_labels=True)
    axdendro.set_xticks([])
    # axdendro.set_yticks([])

    # Plot distance matrix.
    axmatrix = fig.add_axes([0.3,0.1,0.6,0.8])
    index = Z['leaves']
    D = matrix[index,:]
    D = D[:,index]
    im = axmatrix.matshow(D, aspect='auto', origin='lower')
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])

    # Plot colorbar.
    axcolor = fig.add_axes([0.91,0.1,0.02,0.8])
    pylab.colorbar(im, cax=axcolor)

    plt.savefig(data_generation+distance_measure+"Dendrogram",bbox_inches="tight")
    plt.close(fig)

def generate_grid_tsr_sector(df):
    # Cluster assignment
    # Conditions
    conditions_tsr = [
        # 20 Income deciles
        (df["TSR"] >= np.percentile(df["TSR"], 0)) & (df["TSR"] <= np.percentile(df["TSR"], 10)),
        (df["TSR"] >= np.percentile(df["TSR"], 10)) & (df["TSR"] <= np.percentile(df["TSR"], 20)),
        (df["TSR"] >= np.percentile(df["TSR"], 20)) & (df["TSR"] <= np.percentile(df["TSR"], 30)),
        (df["TSR"] >= np.percentile(df["TSR"], 30)) & (df["TSR"] <= np.percentile(df["TSR"], 40)),
        (df["TSR"] >= np.percentile(df["TSR"], 40)) & (df["TSR"] <= np.percentile(df["TSR"], 50)),
        (df["TSR"] >= np.percentile(df["TSR"], 50)) & (df["TSR"] <= np.percentile(df["TSR"], 60)),
        (df["TSR"] >= np.percentile(df["TSR"], 60)) & (df["TSR"] <= np.percentile(df["TSR"], 70)),
        (df["TSR"] >= np.percentile(df["TSR"], 70)) & (df["TSR"] <= np.percentile(df["TSR"], 80)),
        (df["TSR"] >= np.percentile(df["TSR"], 80)) & (df["TSR"] <= np.percentile(df["TSR"], 90)),
        (df["TSR"] >= np.percentile(df["TSR"], 90)) & (df["TSR"] <= np.percentile(df["TSR"], 100))
    ]

    # Values to display
    values_tsr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    conditions_sector = [
        # Sector Classification
        (df["Sector"] == "Utilities"),
    (df["Sector"] == "Industrials"),
    (df["Sector"] == "Financials"),
    (df["Sector"] == "Materials"),
    (df["Sector"] == "Information Technology"),
    (df["Sector"] == "Energy"),
    (df["Sector"] == "Health Care"),
    (df["Sector"] == "Consumer Discretionary"),
    (df["Sector"] == "Real Estate"),
    (df["Sector"] == "Consumer Staples"),
    (df["Sector"] == "Communication Services"),
    (df["Sector"] == np.nan)]

    # Values to display
    values_sector = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    # TSR & Sector
    df["TSR_grid"] = np.select(conditions_tsr, values_tsr)
    df["Sector_grid"] = np.select(conditions_sector, values_sector)

    return df

def generate_market_cap_class(df):
    # Cluster assignment
    # Conditions
    conditions_mcap = [
        # 20 Income deciles
        (df["Market_capitalisation_f"] < 25000),
        (df["Market_capitalisation_f"] >= 25000) & (df["Market_capitalisation_f"] < 50000),
        (df["Market_capitalisation_f"] >= 50000) & (df["Market_capitalisation_f"] < 100000),
        (df["Market_capitalisation_f"] >= 100000)
    ]
    # Values to display
    values_mcap = ["<25b", "25b-50b", "50b-100b", ">100b"]

    df["mcap_grid"] = np.select(conditions_mcap, values_mcap)
    return df

def waterfall_value_plot(series, title):
    # Import dataframe
    df = pd.DataFrame({'Value_created':np.maximum(series,0),'Value_destroyed':np.minimum(series,0)})
    blank = series.cumsum().shift(1).fillna(0)
    df.plot(kind='bar', stacked=True, bottom=blank, color=['g','r'], rot=20, fontsize=6)
    step = blank.reset_index(drop=True).repeat(3).shift(-1)
    step[1::3] = np.nan

    # Plot Figure
    plt.plot(step.index, step.values,'k')
    plt.title(title)
    plt.xlabel("Levers")
    plt.ylabel("Value created / destroyed (%) per annum")
    plt.savefig("Waterfall_"+title, dpi=300, bbox_inches='tight')
    plt.show()

def compute_percentiles(values):
    percentile_5 = np.percentile(values, 5)
    percentile_25 = np.percentile(values, 25)
    percentile_50 = np.percentile(values, 50)
    percentile_75 = np.percentile(values, 75)
    percentile_95 = np.percentile(values, 95)
    return percentile_5, percentile_25, percentile_50, percentile_75, percentile_95

def geometric_return(last_value, first_value, n):
    return (last_value/first_value)**(1/n) - 1

def firefly_plot(years, revenue_vector, ep_fe_vector, cluster_label):

    x_axis_list = []
    y_axis_list = []
    labels_list = []

    # Engineer individual columns
    labels = np.array(years)
    x_axis = np.array(revenue_vector)
    y_axis = np.array(ep_fe_vector)

    # Append results from all companies in a list
    x_axis_list.append(list(x_axis))
    y_axis_list.append(list(y_axis))
    labels_list.append(list(labels))

    # Generate the grid for each axis
    x_pad = len(max(x_axis_list, key=len))
    x_fill_list = np.array([i + [0] * (x_pad - len(i)) for i in x_axis_list])
    y_pad = len(max(x_axis_list, key=len))
    y_fill_list = np.array([i + [0] * (y_pad - len(i)) for i in y_axis_list])
    labels_pad = len(max(labels_list, key=len))
    labels_fill_list = np.array([i + [0] * (labels_pad - len(i)) for i in labels_list])

    x = np.linspace(np.min(np.nan_to_num(x_fill_list)), np.max(np.nan_to_num(x_fill_list)), 100)
    y = np.linspace(np.min(np.nan_to_num(y_fill_list)), np.max(np.nan_to_num(y_fill_list)), 100)

    # Set automatic parameters for plotting
    x_lb = min(-.3, np.min(x))
    x_ub = max(.3, np.max(x))
    y_lb = min(-.3, np.min(y))
    y_ub = max(.3, np.max(y))

    x_segment_ranges = [(x_lb, 0), (0, .1), (.1, .2), (.2, x_ub)]
    y_segment_ranges = [(y_lb, 0), (0, y_ub)]
    label_counter = 0
    labels = ["Untenable", "Challenged", "Trapped", "Virtuous", "Brave", "Famous", "Fearless", "Legendary"]

    fig, ax = plt.subplots()
    for i in range(len(x_fill_list)):
        # Generate plots
        if i == 0:
            plt.plot(x_axis_list[i], y_axis_list[i], '-o', label=cluster_label, color="blue")
        # If we want a generic figure for all competitors, just do !=
        if i != 0:
            plt.plot(x_axis_list[i], y_axis_list[i], '-o', label=cluster_label, alpha=0.4, linestyle='--')
        for j, txt in enumerate(labels_list[i]):
            plt.annotate(labels_list[i][j], (x_axis_list[i][j], y_axis_list[i][j]), fontsize=6)

    for x_range in x_segment_ranges:
        for y_range in y_segment_ranges:
            rect = Rectangle((x_range[0], y_range[0]), x_range[1] - x_range[0], y_range[1] - y_range[0],
                             linewidth=0.3, edgecolor='black', facecolor='red', alpha=0.5)
            ax.add_patch(rect)
            # Add labels to each rectangle
            label = labels[label_counter]
            ax.text((x_range[0] + x_range[1]) / 2, (y_range[0] + y_range[1]) / 2, label,
                    ha='center', va='center', color='black', fontsize=8, fontweight='bold', rotation=15)
            label_counter += 1

    plt.title(cluster_label)
    plt.xlabel("Revenue growth (3 year moving average)")
    plt.ylabel("Economic profit / funds employed")
    plt.legend()
    plt.savefig("Firefly_plot_cluster_"+cluster_label)
    plt.show()

def get_even_clusters(X, cluster_size):
    n_clusters = int(np.ceil(len(X)/cluster_size))
    kmeans = KMeans(n_clusters)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_
    centers = centers.reshape(-1, 1, X.shape[-1]).repeat(cluster_size, 1).reshape(-1, X.shape[-1])
    distance_matrix = cdist(X, centers)
    clusters = linear_sum_assignment(distance_matrix)[1]//cluster_size
    return clusters

def value_creation_waterfall(df_initial, df_final, init_year, final_year):
    # Calculate metrics for FY Initial
    fy_final_market_cap = df_final["Market_Capitalisation"].sum()
    fy_final_revenue = df_final["Revenue"].sum()
    fy_final_npat = df_final["NPAT"].sum()

    # Ensure NPAT values are not zero or negative to avoid division errors
    if fy_final_npat <= 0 or df_initial["NPAT"].sum() <= 0:
        raise ValueError("Unable to compute valid waterfall due to negative aggregate market earnings")

    fy_final_margin = fy_final_npat / fy_final_revenue
    fy_final_market_pe = fy_final_market_cap / fy_final_npat

    # Calculate metrics for FY Beginning
    fy_initial_market_cap = df_initial["Market_Capitalisation"].sum()
    fy_initial_revenue = df_initial["Revenue"].sum()
    fy_initial_npat = df_initial["NPAT"].sum()
    fy_initial_margin = fy_initial_npat / fy_initial_revenue
    fy_initial_market_pe = fy_initial_market_cap / fy_initial_npat

    # Check for negative or zero P/E ratios before proceeding
    if fy_initial_market_pe <= 0 or fy_final_market_pe <= 0:
        raise ValueError("Unable to compute valid waterfall due to negative aggregate market earnings")

    # Calculate incremental values
    incremental_revenue = (fy_final_revenue - fy_initial_revenue) * fy_initial_margin * fy_initial_market_pe
    incremental_margin = fy_final_revenue * (fy_final_margin - fy_initial_margin) * fy_initial_market_pe
    incremental_pe = fy_final_revenue * fy_final_margin * (fy_final_market_pe - fy_initial_market_pe)

    # Total change in market value
    total_change = fy_final_market_cap - fy_initial_market_cap

    # Verify that the total incremental values add up to the total change
    assert round(incremental_revenue + incremental_margin + incremental_pe, 2) == round(total_change, 2), "Incremental values do not sum up to total change."

    # Create waterfall chart
    components = ['Initial Equity Market Value', 'Revenue Growth', 'Margin Change', 'Multiple Change', 'Final Equity Market Value']
    values = [fy_initial_market_cap, incremental_revenue, incremental_margin, incremental_pe, fy_final_market_cap]
    cumulative_values = [sum(values[:i + 1]) for i in range(len(values))]

    fig, ax = plt.subplots()
    bar_width = 0.4

    # Plot bars
    for i in range(len(components)):
        if i == 0:
            ax.bar(i, values[i], bar_width, color='black')
        elif i == len(components) - 1:
            ax.bar(i, values[i], bar_width, color='black')
        else:
            color = 'green' if values[i] > 0 else 'red'
            ax.bar(i, values[i], bar_width, bottom=cumulative_values[i - 1], color=color)

    # Add labels
    ax.set_xticks(range(len(components)))
    ax.set_xticklabels(components)
    ax.set_ylabel('Market Value')
    ax.set_title('Waterfall Chart of Market Value Changes: ' + str(init_year) + "-" + str(final_year))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("Market_Value_Waterfall")
    plt.show()
