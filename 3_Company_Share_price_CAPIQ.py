import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import string
import matplotlib.pyplot as pyplot
import matplotlib
matplotlib.use('TkAgg')

# Import data
data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\Global_data.csv")

# Full ticker list
full_ticker_list = ["XTRA:MBG", "BIT:RACE", "XTRA:VOW3"]
ticker_list = data.loc[data["Ticker_full"].isin(full_ticker_list)]["Ticker"].unique()
company_name_list = data.loc[data["Ticker_full"].isin(full_ticker_list)]["Company_name"].unique()
plot_label = "Rebased_share_price"

# Plot adjusted close having rebased
fig, ax = plt.subplots()
for i in range(len(full_ticker_list)):
    company, ticker = full_ticker_list[i].split(":")
    # Loop over companies & countries
    country_i = data.loc[data["Ticker_full"]==full_ticker_list[i]]["Country"].values[0]
    ticker_i = ticker_list[i]
    # Slice specific company
    stock_i = pd.read_csv(fr"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\share_price\{country_i}\_{ticker_i}_price.csv")
    # Get date and price
    stock_i_date = stock_i["Date"]
    stock_i_price = stock_i["Price"].fillna(method="bfill", limit=30)
    stock_i_price_adjusted = [float(i) for i in stock_i_price]

    # Plot adjusted close
    log_returns = (np.log(stock_i_price) - np.log(stock_i_price).shift(1))
    rebased = (100 * np.exp(np.nan_to_num(log_returns.cumsum())))

    # Plot date vs stock price
    plt.plot(stock_i_date, rebased, label = company_name_list[i])
    plt.title(plot_label)
    company_flat = plot_label.translate(str.maketrans('', '', string.punctuation))
    plt.xlabel("Year")
    plt.ylabel("Share price rebased to 100")
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
plt.legend()
plt.savefig(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\plots\Share_price_"+"_"+str(plot_label))
plt.show()