import numpy as np
import pandas as pd
import matplotlib
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from sklearn.cluster import KMeans

# Import data & slice specific company
company_slice = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\platform_data_2025\_ORG.csv")
# Slice the sector
sector = company_slice["Sector"].values[0]
company_list = ["ORG"]
plot_label = "ORG_projected_firefly"
scenario = "N/A" # north, east, north_east
excel_output = True

# Define the list of financial sectors to use CROTE
crote_sectors = ["Banking", "Investment and Wealth", "Insurance", "Financials - other"]

if sector not in crote_sectors:

    # Current 2024 Financials
    fy_24_revenue = company_slice.loc[company_slice["Year"] == 2024]["Revenue"].iloc[0]
    fy_24_assets = company_slice.loc[company_slice["Year"] == 2024]["Total_assets"].iloc[0]
    fy_24_liabilities = company_slice.loc[company_slice["Year"] == 2024]["Total_liabilities"].iloc[0]

    # Next 3 year forecasts (actual values provided, no percentage growth)
    year_append = [2025, 2026, 2027]

    # THESE ARE BUTTONS TO FILL IN DIRECTLY ON THE PLATFORM
    revenue_growth_market = [-.4 / 100, -3.1/ 100, 1.9 / 100]  # Revenue growth still inferred
    assets_forecasts = [19841.5, 19085.5, 21238]  # Direct input for asset values
    liabilities_forecasts = [9511.5, 8882, 10759]  # Direct input for liability values
    npat_market = [1614, 1045, 1061]  # Direct input for NPAT
    cost_of_capital_forecasts = [0.07, 0.07, 0.07]  # Used for economic profit
    non_interest_debt_percentage_liabilities = [0.22, 0.22, 0.22]  # User-defined percentage #todo CHANGE THIS TO NON CURRENT LIABILITIES

    # Initialize projections lists with the first year's values
    revenue_projections = [fy_24_revenue]

    # Calculate the revenue projections based on growth rates
    for i in range(len(revenue_growth_market)):
        revenue_forecasts = revenue_projections[-1] * (1 + revenue_growth_market[i])
        revenue_projections.append(revenue_forecasts)

    # Remove first entry to align with future years
    revenue_projections = revenue_projections[1:]

    # Compute net asset value projections (Direct input for assets & liabilities)
    net_asset_value_projections = [asset - liability for asset, liability in
                                   zip(assets_forecasts, liabilities_forecasts)]

    # Compute non-interest bearing debt projections
    non_interest_debt_projections = [liabilities * percentage for liabilities, percentage in
                                     zip(liabilities_forecasts, non_interest_debt_percentage_liabilities)] #TODO CHANGE TO NON CURRENT LIABILITIES

    # Compute funds employed
    funds_employed = [non_interest_debt + net_asset_value for non_interest_debt, net_asset_value in #TODO CHANGE TO FE = NAV + NCL
                      zip(non_interest_debt_projections, net_asset_value_projections)]

    # Compute economic profit
    economic_profit = [npat - (funds * cost_of_capital) for npat, funds, cost_of_capital in
                       zip(npat_market, funds_employed, cost_of_capital_forecasts)]

    # Compute EP/FE (economic profit to funds employed ratio)
    ep_fe_market = [ep / funds for ep, funds in zip(economic_profit, funds_employed)]

    # Output the results
    print("Revenue Projections:", revenue_projections)
    print("Asset Forecasts:", assets_forecasts)
    print("Liability Forecasts:", liabilities_forecasts)
    print("Net Asset Value Projections:", net_asset_value_projections)
    print("Non-interest Debt Projections:", non_interest_debt_projections)
    print("Funds Employed:", funds_employed)
    print("Economic Profit:", economic_profit)
    print("EP/FE:", ep_fe_market)

    # Append predictions - Market
    year_predictions = pd.concat([company_slice["Year"], pd.Series(year_append)], ignore_index=True)
    ep_fe_predictions_market = pd.concat([company_slice["EVA_ratio_bespoke"], pd.Series(ep_fe_market)],ignore_index=True)
    revenue_growth_predictions_1_market = pd.concat(
        [company_slice["Revenue_growth_1_f"], pd.Series(revenue_growth_market)], ignore_index=True)
    revenue_growth_predictions_3_market = revenue_growth_predictions_1_market.rolling(window=3).mean()
    x_axis_list_market = [(list(revenue_growth_predictions_3_market))]
    y_axis_list_market = [(list(ep_fe_predictions_market))]
    labels_list = [(list(year_predictions))]

    # Generate the grid for each axis
    x_pad = len(max(x_axis_list_market + x_axis_list_market, key=len))
    x_fill_list_market = np.array([i + [0] * (x_pad - len(i)) for i in x_axis_list_market])

    y_pad = len(max(y_axis_list_market + y_axis_list_market, key=len))
    y_fill_list_market = np.array([i + [0] * (y_pad - len(i)) for i in y_axis_list_market])

    labels_pad = len(max(labels_list, key=len))
    labels_fill_list = np.array([i + [0] * (labels_pad - len(i)) for i in labels_list])

    x = np.linspace(np.min(np.nan_to_num(x_fill_list_market)), np.max(np.nan_to_num(x_fill_list_market)), 100)
    y = np.linspace(np.min(np.nan_to_num(y_fill_list_market)), np.max(np.nan_to_num(y_fill_list_market)), 100)

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

    # Define colors for actuals, market consensus, and scenario 1
    actual_color = 'blue'
    consensus_color = 'red'
    scenario_color = 'black'

    # Plot the actual data and market consensus data
    for i in range(len(x_fill_list_market)):
        for j in range(len(x_axis_list_market[i]) - 1):
            if labels_list[i][j + 1] in year_append:
                plt.plot(x_axis_list_market[i][j:j + 2], y_axis_list_market[i][j:j + 2], '-o', color=consensus_color)
            else:
                plt.plot(x_axis_list_market[i][j:j + 2], y_axis_list_market[i][j:j + 2], '-o', color=actual_color)
        for j, txt in enumerate(labels_list[i]):
            plt.annotate(labels_list[i][j], (x_axis_list_market[i][j], y_axis_list_market[i][j]), fontsize=6,
                         color='black')

    # Add placeholder lines for the legend
    plt.plot([], [], '-o', color=actual_color, label='Actuals')
    plt.plot([], [], '-o', color=consensus_color, label='Market Consensus')
    plt.plot([], [], '-o', color=scenario_color, label=scenario)

    # Draw the rectangles and labels for the segments
    for x_range in x_segment_ranges:
        for y_range in y_segment_ranges:
            rect = Rectangle((x_range[0], y_range[0]), x_range[1] - x_range[0], y_range[1] - y_range[0],
                             linewidth=0.3, edgecolor='black', facecolor='red', alpha=0.5)
            ax.add_patch(rect)
            label = labels[label_counter]
            ax.text((x_range[0] + x_range[1]) / 2, (y_range[0] + y_range[1]) / 2, label,
                    ha='center', va='center', color='black', fontsize=8, fontweight='bold', rotation=15)
            label_counter += 1

    plt.title(plot_label)
    plt.xlabel("Revenue growth (3 year moving average)")
    plt.ylabel("EVA Ratio")
    plt.legend()
    plt.savefig(plot_label)
    plt.show()

else:
    # **CROTE Calculation (for financial sectors)**
    fy_23_revenue = company_slice.loc[company_slice["Year"] == 2024, "Revenue"].iloc[0]
    fy_23_assets = company_slice.loc[company_slice["Year"] == 2024, "Total_assets"].iloc[0]
    fy_23_equity = company_slice.loc[company_slice["Year"] == 2024, "Total_equity"].iloc[0]
    fy_23_minority_interests = company_slice.loc[company_slice["Year"] == 2024, "Minority_interest"].iloc[0]
    fy_23_goodwill = company_slice.loc[company_slice["Year"] == 2024, "Goodwill"].iloc[0]
    fy_23_intangibles = company_slice.loc[company_slice["Year"] == 2024, "Other_intangibles"].iloc[0]

    year_append = [2025, 2026, 2027]
    revenue_growth_market = [7.45/100, 3.1/100, 3.8/100]
    npat_forecasts = [9962, 10106, 10432]
    total_equity_forecasts = [73884, 74803, 76900]
    total_assets_forecasts = [1280638.0, 1340585.0, 1409390.0]
    cost_of_equity = 0.0876

    tangible_equity_projections = []
    crote_projections = []

    for i in range(len(year_append)):
        tangible_equity = total_equity_forecasts[i] - (
                    (fy_23_minority_interests / fy_23_assets) * total_assets_forecasts[i]) - \
                          ((fy_23_goodwill / fy_23_assets) * total_assets_forecasts[i]) - \
                          ((fy_23_intangibles / fy_23_assets) * total_assets_forecasts[i])
        tangible_equity_projections.append(tangible_equity)

        cash_return = npat_forecasts[i] - (tangible_equity * cost_of_equity)
        crote = cash_return / tangible_equity
        crote_projections.append(crote)

    # **Firefly Plot**
    year_predictions = pd.concat([company_slice["Year"], pd.Series(year_append)], ignore_index=True)
    crote_predictions_market = pd.concat([company_slice["EVA_ratio_bespoke"], pd.Series(crote_projections)], ignore_index=True)
    revenue_growth_predictions_1_market = pd.concat([company_slice["Revenue_growth_1_f"], pd.Series(revenue_growth_market)], ignore_index=True)
    revenue_growth_predictions_3_market = revenue_growth_predictions_1_market.rolling(window=3).mean()

    x_axis_list_market = [(list(revenue_growth_predictions_3_market))]
    y_axis_list_market = [(list(crote_predictions_market))]
    labels_list = [(list(year_predictions))]

    # Generate the grid for each axis
    x_pad = len(max(x_axis_list_market + x_axis_list_market, key=len))
    x_fill_list_market = np.array([i + [0] * (x_pad - len(i)) for i in x_axis_list_market])

    y_pad = len(max(y_axis_list_market + y_axis_list_market, key=len))
    y_fill_list_market = np.array([i + [0] * (y_pad - len(i)) for i in y_axis_list_market])

    labels_pad = len(max(labels_list, key=len))
    labels_fill_list = np.array([i + [0] * (labels_pad - len(i)) for i in labels_list])

    x = np.linspace(np.min(np.nan_to_num(x_fill_list_market)), np.max(np.nan_to_num(x_fill_list_market)), 100)
    y = np.linspace(np.min(np.nan_to_num(y_fill_list_market)), np.max(np.nan_to_num(y_fill_list_market)), 100)

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

    # Define colors for actuals, market consensus, and scenario 1
    actual_color = 'blue'
    consensus_color = 'red'
    scenario_color = 'black'

    # Plot the actual data and market consensus data
    for i in range(len(x_fill_list_market)):
        for j in range(len(x_axis_list_market[i]) - 1):
            if labels_list[i][j + 1] in year_append:
                plt.plot(x_axis_list_market[i][j:j + 2], y_axis_list_market[i][j:j + 2], '-o', color=consensus_color)
            else:
                plt.plot(x_axis_list_market[i][j:j + 2], y_axis_list_market[i][j:j + 2], '-o', color=actual_color)
        for j, txt in enumerate(labels_list[i]):
            plt.annotate(labels_list[i][j], (x_axis_list_market[i][j], y_axis_list_market[i][j]), fontsize=6,
                         color='black')

    # Add placeholder lines for the legend
    plt.plot([], [], '-o', color=actual_color, label='Actuals')
    plt.plot([], [], '-o', color=consensus_color, label='Market Consensus')
    plt.plot([], [], '-o', color=scenario_color, label=scenario)

    # Draw the rectangles and labels for the segments
    for x_range in x_segment_ranges:
        for y_range in y_segment_ranges:
            rect = Rectangle((x_range[0], y_range[0]), x_range[1] - x_range[0], y_range[1] - y_range[0],
                             linewidth=0.3, edgecolor='black', facecolor='red', alpha=0.5)
            ax.add_patch(rect)
            label = labels[label_counter]
            ax.text((x_range[0] + x_range[1]) / 2, (y_range[0] + y_range[1]) / 2, label,
                    ha='center', va='center', color='black', fontsize=8, fontweight='bold', rotation=15)
            label_counter += 1

    plt.title(plot_label)
    plt.xlabel("Revenue growth (3 year moving average)")
    plt.ylabel("EVA Ratio")
    plt.legend()
    plt.savefig(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\_Firefly_projections\Firefly_"+plot_label+"_forecast_data_plot")
    plt.show()

if excel_output:
    # Write elements to csv
    # x_fill_df = pd.DataFrame(x_fill_list).transpose()
    x_stacked_array = np.vstack(x_fill_list_market)
    x_reshaped_array = x_stacked_array.reshape(-1, 1)
    y_stacked_array = np.vstack(y_fill_list_market)
    y_reshaped_array = y_stacked_array.reshape(-1, 1)
    labels_stacked_array = np.vstack(labels_list)
    labels_reshaped_array = labels_stacked_array.reshape(-1, 1)

    # Convert arrays to 1D arrays
    x_flat = x_reshaped_array.flatten()
    y_flat = y_reshaped_array.flatten()
    labels_flat = labels_reshaped_array.flatten()

    # Write a function for the marker array
    def create_marker_array(rows, cols):
        marker_array = np.zeros((rows, cols), dtype=int)
        for i in range(rows):
            marker_array[i, :] = i + 1
        return marker_array.reshape(-1, 1)


    # Get dimensions of labels list for marker array
    rows = len(labels_fill_list)
    cols = len(labels_fill_list[0])

    # Create the marker array
    marker_array = create_marker_array(rows, cols).flatten()

    # Create a DataFrame from the 1D arrays
    df = pd.DataFrame({
        'Series Labels': labels_flat,
        'X': x_flat,
        'Y': y_flat,
        'Marker and regression grouping': marker_array
    })

    # Write the DataFrame to a CSV file
    df.to_csv(
        r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\_Firefly_projections\_" + plot_label + "_firefly_projected_data_.csv",
        index=False)

