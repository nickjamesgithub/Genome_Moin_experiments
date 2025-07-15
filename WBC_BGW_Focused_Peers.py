import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Required tickers and plot label
tickers_ = ["ANZ:ASX", "NAB:ASX", "CBA:ASX", "WBC:ASX", "MQG", "ITAU", "RBC", "Santander", "DBS", "JPM"]
plot_label = "Focused_peer_group"

# Hardcoded data dictionary
company_data = {
    "ANZ": {"share_price": 30.08, "trailing_eps": 2.24, "forward_eps_1": 2.29, "forward_eps_2":2.14, "forward_eps_3":2.13, "cost_of_equity": 0.088},
    "NAB": {"share_price": 39.66, "trailing_eps": 2.26, "forward_eps_1": 2.26, "forward_eps_2": 2.14, "forward_eps_3": 2.13, "cost_of_equity": 0.088},
    "CBA": {"share_price": 178.8, "trailing_eps": 5.84, "forward_eps_1": 6.18, "forward_eps_2": 6.09,"forward_eps_3": 6.14, "cost_of_equity": 0.088},
    "WBC": {"share_price": 33.56, "trailing_eps": 1.95, "forward_eps_1": 2.00, "forward_eps_2": 1.89, "forward_eps_3": 1.86, "cost_of_equity": 0.088},
    "MQG": {"share_price": 221.28, "trailing_eps": 9.75, "forward_eps_1": 10.51, "forward_eps_2": 11.64,"forward_eps_3": 12.53, "cost_of_equity": 0.088},
    "Itau": {"share_price": 34.96, "trailing_eps":3.83, "forward_eps_1": 4.32, "forward_eps_2": 4.69,"forward_eps_3": 4.90, "cost_of_equity": 0.088},
    "RBC": {"share_price": 180.37, "trailing_eps": 12.10, "forward_eps_1": 13.15, "forward_eps_2": 14.00,"forward_eps_3": 15.15, "cost_of_equity": 0.088},
    "Santander": {"share_price": 7.20, "trailing_eps": .76, "forward_eps_1": .83, "forward_eps_2": .84,"forward_eps_3": .96, "cost_of_equity": 0.088},
    "DBS": {"share_price": 46.29, "trailing_eps": 3.98, "forward_eps_1": 3.76, "forward_eps_2": 3.84, "forward_eps_3": 4.05, "cost_of_equity": 0.088},
    "JPM": {"share_price": 286.86, "trailing_eps": 19.75, "forward_eps_1": 19.00, "forward_eps_2": 20.00,"forward_eps_3": 21.00, "cost_of_equity": 0.088},
}

# Store Black/Grey/White values
bgw_values_list = []

for company_i in company_data.keys():
    try:
        # Retrieve hardcoded values
        data = company_data[company_i]
        share_price = data["share_price"]
        trailing_eps = data["trailing_eps"]
        forward_eps_1 = data["forward_eps_1"]
        forward_eps_2 = data["forward_eps_2"]
        forward_eps_3 = data["forward_eps_3"]
        cost_of_equity = data["cost_of_equity"]

        # Compute Black space
        black_space = trailing_eps / cost_of_equity

        # Present value of EPS for Grey Space
        pv_eps_1 = (forward_eps_1 - trailing_eps) / (1 + cost_of_equity)
        pv_eps_2 = (forward_eps_2 - forward_eps_1) / (1 + cost_of_equity) ** 2 * cost_of_equity
        pv_eps_3 = (forward_eps_3 - forward_eps_2) / (1 + cost_of_equity) ** 3 * cost_of_equity
        grey_space = pv_eps_1 + pv_eps_2 + pv_eps_3

        # White space calculation
        white_space = share_price - grey_space - black_space

        # Append black/grey/white values to the master list
        bgw_values_list.append([black_space, grey_space, white_space])

    except Exception as e:
        print(f"Error with company {company_i}: {e}")

def bgw_percentage_calculator(bgw_values):
    black, grey, white = bgw_values

    black_adj = max(black, 0)
    grey_adj = max(grey, 0)
    white_adj = max(white, 0)

    bgw_values_adjusted = [black_adj, grey_adj, white_adj]
    bgw_values_percentages = np.round((bgw_values_adjusted / np.sum(bgw_values_adjusted)), 3)

    return list(bgw_values_percentages)

# BGW spat out to local script
bgw_raw_df = pd.DataFrame(bgw_values_list, columns=["Black", "Grey", "White"], index=tickers_)
bgw_raw_df.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\WBC\WBC_BGW_raw_data.csv", index=False)

# Calculate B/G/W percentages for each company
bgw_percentages_list = [bgw_percentage_calculator(values) for values in bgw_values_list]

# Black/Grey/White dataframe for plotting
bgw_percentages_df = pd.DataFrame(bgw_percentages_list, columns=["Black", "Grey", "White"], index=tickers_)

# Plot
bgw_percentages_df.plot.bar(stacked=True, rot=0, color={'Black': 'Black', "Grey": 'dimgrey', "White": 'lightgrey'})
plt.title(plot_label + " - Black/Grey/White space")
plt.savefig("BGW_" + plot_label)
plt.show()

# Create DataFrame with the black, grey, and white space values
bgw_values_df = pd.DataFrame(bgw_values_list, columns=["Black Space", "Grey Space", "White Space"], index=company_data.keys())
bgw_values_df.index.name = "Company"

# Reset index to have "Company" as a column, then save to CSV
bgw_values_df.reset_index().to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\WBC\WBC_BGW_Values.csv", index=False)

# Display the result (optional)
print(bgw_values_df)