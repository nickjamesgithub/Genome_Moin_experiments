import numpy as np
import pandas as pd
from Utilities import compute_percentiles, firefly_plot, geometric_return
import matplotlib
from Utilities import generate_market_cap_class, waterfall_value_plot, geometric_return, dendrogram_plot, get_even_clusters
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
from matplotlib.patches import Rectangle
import matplotlib
from sklearn.metrics.pairwise import cosine_distances
matplotlib.use('TkAgg')

# Algorithms to run
run_similar = False
run_defined_trajectory = True
run_firefly_plot = False

if run_similar:
    # Read the data
    df_full = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\Global_data.csv")

    # Desired sectors and date range
    country_list = ["USA", 'AUS', 'INDIA', 'JAPAN', 'EURO', 'UK']
    unique_sectors = df_full["Sector"].unique()
    desired_sectors = unique_sectors
    company_ticker = "ASX:SGH"
    start_year = 2017
    end_year = 2024

    # 1. Filter by Country & Sector
    df_slice = df_full[(df_full['Country'].isin(country_list)) & (df_full["Sector"].isin(unique_sectors))]
    tickers = df_slice["Ticker_full"].unique()

    # Encode X and Y Values
    df_slice["Genome_encoding_x"] = df_slice["Revenue_growth_3_f"]
    df_slice["Genome_encoding_y"] = df_slice["EVA_ratio_bespoke"]

    # Company slice
    target_genome_slice = df_slice.loc[(df_slice["Year"]>=start_year) & (df_slice["Year"]<=end_year) & (df_slice["Ticker_full"]==company_ticker)][["Genome_encoding_x", "Genome_encoding_y"]]

    # Function to compute Cosine Distance
    def cosine_distance(df1, df2):
        return cosine_distances(df1.values.reshape(1, -1), df2.values.reshape(1, -1))[0][0]

    # Initialize list to store results
    distance_results = []

    # Define the fixed anchor window length
    time_window_length = end_year - start_year + 1  # Inclusive range

    # Get anchor company's available years
    anchor_years = df_slice[df_slice["Ticker_full"] == company_ticker]["Year"].unique()
    anchor_years.sort()

    # Get anchor company name
    anchor_company_name = df_slice[df_slice["Ticker_full"] == company_ticker]["Company_name"].iloc[0]

    # Get anchor company data for the fixed anchor window
    anchor_df = df_slice[(df_slice["Ticker_full"] == company_ticker) &
                         (df_slice["Year"].between(start_year, end_year))][
        ["Genome_encoding_x", "Genome_encoding_y", "Genome_classification_bespoke"]]

    anchor_genome_seq = list(anchor_df["Genome_classification_bespoke"].values)

    # Loop through all tickers for comparison
    for ticker in tickers:
        if ticker != company_ticker:  # Avoid self-comparison
            company_years = df_slice[df_slice["Ticker_full"] == ticker]["Year"].unique()
            company_years.sort()

            # Iterate over all valid sliding windows of `time_window_length`
            for i in range(len(company_years) - time_window_length + 1):
                compare_start = company_years[i]
                compare_end = compare_start + time_window_length - 1  # Ensure fixed-length window

                # Get data for this sliding window
                compare_df = df_slice[(df_slice["Ticker_full"] == ticker) &
                                      (df_slice["Year"].between(compare_start, compare_end))][
                    ["Genome_encoding_x", "Genome_encoding_y", "Genome_classification_bespoke"]]

                # **Skip the computation entirely if there's an "UNKNOWN"**
                try:
                    if "UNKNOWN" in anchor_genome_seq or "UNKNOWN" in compare_df["Genome_classification_bespoke"].values:
                        raise ValueError("Skipping due to UNKNOWN value")

                    # Ensure valid comparison (both have `time_window_length` data points)
                    if len(anchor_df) == len(compare_df) == time_window_length:
                        distance = cosine_distance(anchor_df[["Genome_encoding_x", "Genome_encoding_y"]],
                                               compare_df[["Genome_encoding_x", "Genome_encoding_y"]])

                        compare_company_name = df_slice[df_slice["Ticker_full"] == ticker]["Company_name"].iloc[0]
                        compare_genome_seq = list(compare_df["Genome_classification_bespoke"].values)

                        # Store results
                        distance_results.append([
                            company_ticker, anchor_company_name, start_year, end_year,  # Anchor details
                            ticker, compare_company_name, compare_start, compare_end,  # Comparison details
                            distance, anchor_genome_seq, compare_genome_seq
                        ])

                        print(f"Anchor: {company_ticker} ({anchor_company_name}) [{start_year}-{end_year}] | "
                              f"Comparing: {ticker} ({compare_company_name}) [{compare_start}-{compare_end}] | "
                              f"L1 Distance: {distance}")

                except ValueError:
                    print(f"Skipping comparison {ticker} due to UNKNOWN in sequence.")

    # Convert to DataFrame
    distance_df = pd.DataFrame(distance_results, columns=[
        "Anchor_Ticker", "Anchor_Company", "Anchor_Start_Year", "Anchor_End_Year",
        "Comparison_Ticker", "Comparison_Company", "Comparison_Start_Year", "Comparison_End_Year",
        "Distance", "Anchor_Genome_Sequence", "Comparison_Genome_Sequence"
    ])


if run_defined_trajectory:
    # Read the data
    df_full = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\Global_data.csv")

    # Parameters
    time_window_length = 6  # Define the time window length
    country_list = ["USA", 'AUS', 'INDIA', 'JAPAN', 'EURO', 'UK']
    unique_sectors = df_full["Sector"].unique()

    # Filter by Country and Sector
    df = df_full[(df_full['Country'].isin(country_list)) & (df_full["Sector"].isin(unique_sectors))]

    # Define potential Genome classifications for each year in the sequence
    # Example format: {year_offset: [possible_genomes]}
    # year_offset is 0 for the first year in the sequence, 1 for the second, etc.
    genome_options = {
        0: ["UNTENABLE", "UNTENABLE"],
        1: ["UNTENABLE" , "CHALLENGED"],
        2: ["CHALLENGED" , "VIRTUOUS"],
        3: ["VIRTUOUS", "FAMOUS"],
        4: ["FAMOUS", "LEGENDARY"],
        5: ["FAMOUS", "LEGENDARY"]
    }

    # Function to check if the sequence in the DataFrame matches any of the allowed sequences
    def matches_allowed_sequence(df_sequence):
        for year_offset, allowed_genomes in genome_options.items():
            if df_sequence.iloc[year_offset]["Genome_classification_bespoke"] not in allowed_genomes:
                return False
        return True


    # Scan over each ticker
    results = []
    tickers = df["Ticker_full"].unique()

    for ticker in tickers:
        print("Iteration ", ticker)
        ticker_data = df[df["Ticker_full"] == ticker]
        years = sorted(ticker_data["Year"].unique())

        # Check each possible starting year within the range that allows a full sequence
        for start_year in years:
            if start_year + time_window_length - 1 in years:  # Ensure the sequence can be complete
                sequence_df = ticker_data[
                    (ticker_data["Year"] >= start_year) & (ticker_data["Year"] < start_year + time_window_length)]
                if len(sequence_df) == time_window_length:  # Ensure no missing years in the sequence
                    if matches_allowed_sequence(sequence_df):
                        results.append({
                            "Ticker": ticker,
                            "Company Name": sequence_df["Company_name"].iloc[0],
                            "Start Year": start_year,
                            "End Year": start_year + time_window_length - 1,
                            "Genome Sequence": sequence_df["Genome_classification_bespoke"].tolist()
                        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    print(results_df)

if run_firefly_plot:
    # Import data
    data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\Global_data.csv")

    # Full ticker list and corresponding start/end years
    full_ticker_list = ["NasdaqGS:ADI", "ASX:PNI", "TSE:8031", "NYSE:CAG"]
    start_years = [2013, 2014, 2011, 2015]
    end_years = [2018, 2019, 2018, 2021]
    plot_label = "Aspirational_turnaround_firefly"

    # Extract company names before looping
    company_name_list = [data.loc[data["Ticker_full"] == ticker, "Company_name"].iloc[0] for ticker in full_ticker_list]

    x_axis_list = []
    y_axis_list = []
    labels_list = []

    for i, full_ticker in enumerate(full_ticker_list):
        idx_i, company_i = full_ticker.split(":")

        # Retrieve the correct country for the ticker
        country_i = data.loc[data["Ticker_full"] == full_ticker, "Country"].iloc[0]

        # Load company data from the correct country subfolder
        df = pd.read_csv(
            fr"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\{country_i}\_{company_i}.csv")

        # Filter for the selected time window
        company_slice = df[(df["Year"] >= start_years[i]) & (df["Year"] <= end_years[i])]

        # Extract relevant data
        labels = company_slice["Year"].values
        x_axis = company_slice["Revenue_growth_3_f"].values
        y_axis = company_slice["EVA_ratio_bespoke"].values

        # Append results
        x_axis_list.append(x_axis.tolist())
        y_axis_list.append(y_axis.tolist())
        labels_list.append(labels.tolist())

    # Generate the grid for each axis
    x_pad = max(map(len, x_axis_list))
    y_pad = max(map(len, y_axis_list))
    labels_pad = max(map(len, labels_list))

    x_fill_list = np.array([i + [np.nan] * (x_pad - len(i)) for i in x_axis_list])
    y_fill_list = np.array([i + [np.nan] * (y_pad - len(i)) for i in y_axis_list])
    labels_fill_list = np.array([i + [np.nan] * (labels_pad - len(i)) for i in labels_list])

    x = np.linspace(np.nanmin(x_fill_list), np.nanmax(x_fill_list), 100)
    y = np.linspace(np.nanmin(y_fill_list), np.nanmax(y_fill_list), 100)

    # Set automatic parameters for plotting
    x_lb = min(-.3, np.nanmin(x))
    x_ub = max(.3, np.nanmax(x))
    y_lb = min(-.3, np.nanmin(y))
    y_ub = max(.3, np.nanmax(y))

    x_segment_ranges = [(x_lb, 0), (0, .1), (.1, .2), (.2, x_ub)]
    y_segment_ranges = [(y_lb, 0), (0, y_ub)]
    label_counter = 0
    labels = ["Untenable", "Challenged", "Trapped", "Virtuous", "Brave", "Famous", "Fearless", "Legendary"]

    fig, ax = plt.subplots()

    # Plot each company with proper labels
    for i in range(len(x_fill_list)):
        if i == 0:
            plt.plot(x_axis_list[i], y_axis_list[i], '-o', label=company_name_list[i], color="blue")
        else:
            plt.plot(x_axis_list[i], y_axis_list[i], '-o', label=company_name_list[i], alpha=0.4, linestyle='--')

        # Annotate with year labels
        for j, txt in enumerate(labels_list[i]):
            plt.annotate(txt, (x_axis_list[i][j], y_axis_list[i][j]), fontsize=6)

    # Draw segmented rectangles and add labels
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
    plt.savefig(
        r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\USA_technology\Firefly_plot_CAPIQ_" + plot_label)
    plt.show()


