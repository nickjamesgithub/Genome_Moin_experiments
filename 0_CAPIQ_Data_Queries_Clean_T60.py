import requests, json
from requests.auth import HTTPBasicAuth
# import credentials
import json
import pandas as pd
import numpy as np
import os  # Ensure this is imported at the top

country_list = ["TS60"] # "INDIA", "JAPAN", "EURO", "UK", "USA", "AUS"
# Define the base directory where you want to save the files
# base_directory = r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data"
BASE_DIR = os.path.join(os.path.abspath(os.path.curdir))
GLOBAL_MARKET_DIR = os.path.join(BASE_DIR, "Global_market_constituents")
GLOBAL_DATA_DIR = os.path.join(BASE_DIR, "global_platform_data")

for c in range(len(country_list)):

    country_import = country_list[c] # AUS / USA / INDIA / JAPAN / euro / UK

    # Import mapping data
    if country_import == "TS60":
        # mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Global_market_constituents\Company_list_GPT_SP500.csv", encoding='cp1252')
        mapping_data = pd.read_csv(os.path.join(GLOBAL_MARKET_DIR, "Company_list_GPT_TS60_clean.csv"), encoding='cp1252')

    # WACC data
    # wacc_store = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\WACC_inputs\WACC_Store_2011_2024.csv")
    wacc_store = pd.read_csv(os.path.join(BASE_DIR, "WACC_Store_2011_2024.csv"))
    # Ticker data
    full_ticker_list = mapping_data["Full Ticker"].values
    ticker_slice_list = mapping_data["Ticker"].values

    for i in range(len(full_ticker_list)):
        try:
            ticker = full_ticker_list[i]
            ticker_slice = ticker_slice_list[i]
            # Split the string at ":"
            idx, company_ticker = ticker.split(":")

            # Sort out company & ticker pairing
            company = company_ticker +":"+idx # TICKER FIRST | EXCHANGE SECOND
            company_slice = idx +":"+company_ticker # EXCHANGE FIRST | TICKER SECOND
            print("Company data pull ", company_ticker)

            # Get company ticker
            company_label = company_ticker
            endpoint_url = "https://api-ciq.marketintelligence.spglobal.com/gdsapi/rest/v3/clientservice.json"

            labels = ["Beta", "Market_Capitalisation", "PE", "PBV", "Adjusted_Stock_Price", "Stock_Price",
                      "Dividends_Paid", "Book_Value_Equity", "Minority_interest", "Goodwill", "Other_intangibles",
                      "Shares_outstanding", "NPAT", "Diluted_EPS", "Basic_EPS", "Revenue", "EBIT", "EBITDA", "Total_assets",
                      "Total_liabilities", "Current_liabilities",
                      "Accounts_payable", "Accrued_expenses", "Unearned_revenue", "DTL", "Research_development",
                      "Effective_tax_rate", "Long_term_debt", "Short_term_debt",
                      "Stock_issued", "Stock_repurchased", "Total_debt", "Total_equity", "CAPEX", "Number_of_employees",
                      "Cash_acquisitions", "Gross_profit"]

            req_array = [
                # Beta
                {"function": "GDSP", "identifier": company, "mnemonic": "IQ_BETA","properties": {"AsOfDate": "01/01/2018"}},
                # Market Cap
                {"Function": "GDST", "Identifier": company, "Mnemonic": "IQ_MARKETCAP", "properties": {"frequency": "Yearly", "startDate": "01/01/2011", "endDate": "06/06/2025","CurrencyID": "USD"}},  # 2011
                # P/E
                {"Function": "GDST", "Identifier": company, "Mnemonic": "IQ_PE_NORMALIZED","properties": {"frequency": "Yearly", "startDate": "01/01/2011", "endDate": "06/06/2025",
                                "CurrencyID": "USD"}},
                # P/BV
                {"Function": "GDST", "Identifier": company, "Mnemonic": "IQ_PBV","properties": {"frequency": "Yearly", "startDate": "01/01/2011", "endDate": "06/06/2025","CurrencyID": "USD"}},
                # Adjusted Close Price
                {"Function": "GDST", "Identifier": company, "Mnemonic": "IQ_CLOSEPRICE_ADJ", "properties": {"frequency": "Yearly", "startDate": "01/01/2011", "endDate": "06/06/2025", "CurrencyID": "USD"}},
                # Close Price
                {"Function": "GDST", "Identifier": company, "Mnemonic": "IQ_CLOSEPRICE", "properties": {"frequency": "Yearly", "startDate": "01/01/2011", "endDate": "06/06/2025", "CurrencyID": "USD"}},

                # Total Dividends paid
                {"function": "GDSHE", "identifier": company, "mnemonic": "IQ_TOTAL_DIV_PAID_CF","Properties": {"PeriodType": "IQ_FY-15", "Metadatatag": "PeriodDate", "CurrencyID": "USD"}},
                # Tangible Book Value of Equity
                {"Function": "GDSHE", "Identifier": company, "Mnemonic": "IQ_TBV","Properties": {"PeriodType": "IQ_FY-15", "Metadatatag": "PeriodDate", "CurrencyID": "USD"}},
                # Minority interests
                {"Function": "GDSHE", "Identifier": company, "Mnemonic": "IQ_MINORITY_INTEREST", "Properties": {"PeriodType": "IQ_FY-15", "Metadatatag": "PeriodDate", "CurrencyID": "USD"}},
                # Goodwill
                {"Function": "GDSHE", "Identifier": company, "Mnemonic": "IQ_GW","Properties": {"PeriodType": "IQ_FY-15", "Metadatatag": "PeriodDate", "CurrencyID": "USD"}},
                # Other intangibles
                {"Function": "GDSHE", "Identifier": company, "Mnemonic": "IQ_OTHER_INTAN", "Properties": {"PeriodType": "IQ_FY-15", "Metadatatag": "PeriodDate", "CurrencyID": "USD"}},
                # Number of shares outstanding
                {"Function": "GDSHE", "Identifier": company, "Mnemonic": "IQ_TOTAL_OUTSTANDING_BS_DATE", "Properties": {"PeriodType": "IQ_FY-15", "Metadatatag": "PeriodDate", "CurrencyID": "USD"}},
                # Net Income Excluding extraordinary items
                {"Function": "GDSHE", "Identifier": company, "Mnemonic": "IQ_NI_AVAIL_EXCL","Properties": {"PeriodType": "IQ_FY-15", "Metadatatag": "PeriodDate", "CurrencyID": "USD"}},
                # Diluted EPS
                {"Function": "GDSHE", "Identifier": company, "Mnemonic": "IQ_DILUT_EPS_NORM","Properties": {"PeriodType": "IQ_FY-15", "Metadatatag": "PeriodDate", "CurrencyID": "USD"}},
                # Trailing EPS (including)
                {"Function": "GDSHE", "Identifier": company, "Mnemonic": "IQ_BASIC_EPS_INCL", "Properties": {"PeriodType": "IQ_FY-15", "Metadatatag": "PeriodDate", "CurrencyID": "USD"}},

                # Revenue
                {"Function": "GDSHE", "Identifier": company, "Mnemonic": "IQ_TOTAL_REV",             "Properties": {"PeriodType": "IQ_FY-15", "Metadatatag": "PeriodDate", "CurrencyID": "USD"}},
                # EBIT
                {"Function": "GDSHE", "Identifier": company, "Mnemonic": "IQ_EBIT",             "Properties": {"PeriodType": "IQ_FY-15", "Metadatatag": "PeriodDate", "CurrencyID": "USD"}},
                # EBITDA
                {"Function": "GDSHE", "Identifier": company, "Mnemonic": "IQ_EBITDA",             "Properties": {"PeriodType": "IQ_FY-15", "Metadatatag": "PeriodDate", "CurrencyID": "USD"}},
                # Total Assets
                {"Function": "GDSHE", "Identifier": company, "Mnemonic": "IQ_TOTAL_ASSETS",             "Properties": {"PeriodType": "IQ_FY-15", "Metadatatag": "PeriodDate", "CurrencyID": "USD"}},
                # Total liabilities
                {"Function": "GDSHE", "Identifier": company, "Mnemonic": "IQ_TOTAL_LIAB",             "Properties": {"PeriodType": "IQ_FY-15", "Metadatatag": "PeriodDate", "CurrencyID": "USD"}},
                # Total Current Liabilities
                {"Function": "GDSHE", "Identifier": company, "Mnemonic": "IQ_TOTAL_CL",             "Properties": {"PeriodType": "IQ_FY-15", "Metadatatag": "PeriodDate", "CurrencyID": "USD"}},
                # Total Accounts PAYABLE
                {"Function": "GDSHE", "Identifier": company, "Mnemonic": "IQ_AP",             "Properties": {"PeriodType": "IQ_FY-15", "Metadatatag": "PeriodDate", "CurrencyID": "USD"}},
                # Total Accrued expenses
                {"Function": "GDSHE", "Identifier": company, "Mnemonic": "IQ_AE",             "Properties": {"PeriodType": "IQ_FY-15", "Metadatatag": "PeriodDate", "CurrencyID": "USD"}},
                # Total Unearned Revenue
                {"Function": "GDSHE", "Identifier": company, "Mnemonic": "IQ_UNEARN_REV_LT",             "Properties": {"PeriodType": "IQ_FY-15", "Metadatatag": "PeriodDate", "CurrencyID": "USD"}},
                # Deferred Tax liability
                {"Function": "GDSHE", "Identifier": company, "Mnemonic": "IQ_DEF_TAX_LIAB_LT",             "Properties": {"PeriodType": "IQ_FY-15", "Metadatatag": "PeriodDate", "CurrencyID": "USD"}},
                # Research & Development
                {"Function": "GDSHE", "Identifier": company, "Mnemonic": "IQ_RD_EXP",             "Properties": {"PeriodType": "IQ_FY-15", "Metadatatag": "PeriodDate", "CurrencyID": "USD"}},
                # Effective Tax rate
                {"Function": "GDSHE", "Identifier": company, "Mnemonic": "IQ_EFFECT_TAX_RATE",             "Properties": {"PeriodType": "IQ_FY-15", "Metadatatag": "PeriodDate", "CurrencyID": "USD"}},
                # Long-term debt
                {"Function": "GDSHE", "Identifier": company, "Mnemonic": "IQ_LT_DEBT",             "Properties": {"PeriodType": "IQ_FY-15", "Metadatatag": "PeriodDate", "CurrencyID": "USD"}},
                # Short-term Borrowings
                {"Function": "GDSHE", "Identifier": company, "Mnemonic": "IQ_ST_DEBT",             "Properties": {"PeriodType": "IQ_FY-15", "Metadatatag": "PeriodDate", "CurrencyID": "USD"}},
                # Common stock issued
                {"function": "GDSHE", "identifier": company, "mnemonic": "IQ_COMMON_ISSUED",             "Properties": {"PeriodType": "IQ_FY-15", "Metadatatag": "PeriodDate", "CurrencyID": "USD"}},
                # Common stock repurchased
                {"function": "GDSHE", "identifier": company, "mnemonic": "IQ_COMMON_REP",             "Properties": {"PeriodType": "IQ_FY-15", "Metadatatag": "PeriodDate", "CurrencyID": "USD"}},
                # Total Debt
                {"function": "GDSHE", "identifier": company, "mnemonic": "IQ_TOTAL_DEBT",             "Properties": {"PeriodType": "IQ_FY-15", "Metadatatag": "PeriodDate", "CurrencyID": "USD"}},
                # Total Equity
                {"function": "GDSHE", "identifier": company, "mnemonic": "IQ_TOTAL_EQUITY",             "Properties": {"PeriodType": "IQ_FY-15", "Metadatatag": "PeriodDate", "CurrencyID": "USD"}},
                # CAPEX
                {"function": "GDSHE", "identifier": company, "mnemonic": "IQ_CAPEX",             "Properties": {"PeriodType": "IQ_FY-15", "Metadatatag": "PeriodDate", "CurrencyID": "USD"}},
                # Full-time employees
                {"function": "GDSHE", "identifier": company, "mnemonic": "IQ_FULL_TIME",             "Properties": {"PeriodType": "IQ_FY-15", "Metadatatag": "PeriodDate", "CurrencyID": "USD"}},
                # Cash Acquisitions
                {"function": "GDSHE", "identifier": company, "mnemonic": "IQ_CASH_ACQUIRE_CF",             "Properties": {"PeriodType": "IQ_FY-15", "Metadatatag": "PeriodDate", "CurrencyID": "USD"}},
                # Gross Profit
                {"function": "GDSHE", "identifier": company, "mnemonic": "IQ_GP",             "Properties": {"PeriodType": "IQ_FY-15", "Metadatatag": "PeriodDate", "CurrencyID": "USD"}},
            ]

            req = {"inputRequests": req_array}
            response = requests.post(endpoint_url,
                                     headers={'Content-Type': 'application/json'},
                                     data=json.dumps(req),
                                     auth=HTTPBasicAuth("apiadmin@bain.com", "Bain@1234"),
                                     verify=True)
            print(response)

            print(json.loads(response.text)['GDSSDKResponse'][0]['Rows'])

            # Lists to store values
            feature_date_lists = []
            feature_value_lists = []
            for f in range(len(req_array)):
                # Slice specific value
                ciq_data = json.loads(response.text)

                # Access the "Row" metric value
                if f <= 5:
                    row_value = ciq_data['GDSSDKResponse'][f]['Rows'][0]['Row']
                    dates = ciq_data['GDSSDKResponse'][f]['Headers']
                    feature_float_list = []
                    for item in row_value:
                        try:
                            float_value = float(item)
                            feature_float_list.append(float_value)
                        except ValueError:
                            # Handle the case where the string is not convertible to float
                            feature_float_list.append(None)

                else:
                    row_value = ciq_data['GDSSDKResponse'][f]['Rows']
                    dates = [row['Row'][1] for row in row_value]
                    feature_float_list = []
                    for row_dict in row_value:
                        try:
                            float_value = float(row_dict['Row'][0])
                            feature_float_list.append(float_value)
                        except (ValueError, IndexError):
                            # Handle the case where the value is not convertible to float or the list index is out of bounds
                            feature_float_list.append(None)

                # Append lists of features and dates
                feature_date_lists.append(dates)
                feature_value_lists.append(feature_float_list)

                print("Feature iteration ", f)
                print('Feature', labels[f])
                print("Feature values", feature_float_list)
                print("Feature dates", dates)

            # Feature value dataframe
            df = pd.DataFrame(feature_value_lists)
            years_set = set()

            for date_list in feature_date_lists[1:]:
                try:
                    for date_str in date_list:
                        date_parts = date_str.split('/')
                        year = date_parts[2]  # Extract year as string
                        years_set.add(year)
                except:
                    print("Issue with feature date list iter ", date_list)

            # Extract all unique valid years
            years_set = set()
            for date_list in feature_date_lists:
                for date_str in date_list:
                    try:
                        month, day, year = date_str.split('/')
                        if len(year) == 4:  # Check for valid year format
                            int(month)  # Ensure month is an integer
                            int(day)  # Ensure day is an integer
                            int(year)  # Ensure year is an integer
                            years_set.add(year)
                    except (ValueError, AttributeError):
                        continue

            unique_years = sorted(years_set)

            # Initialize an empty DataFrame with columns as the unique years
            df = pd.DataFrame(columns=unique_years, index=labels)

            # Populate the DataFrame with values aligned to the corresponding years
            for label, dates, values in zip(labels, feature_date_lists, feature_value_lists):
                year_value_dict = {}
                for date, value in zip(dates, values):
                    try:
                        month, day, year = date.split('/')
                        if len(year) == 4:
                            int(month)
                            int(day)
                            int(year)
                            year_value_dict[year] = value
                    except (ValueError, AttributeError):
                        continue
                df.loc[label] = [year_value_dict.get(year, None) for year in unique_years]

            # Drop any columns that contain all NaN values
            df.dropna(axis=1, how='all', inplace=True)

            print(df)

            # Clean the dataframe with a fill na with 0
            df_clean = df.fillna(0)
            df_transpose = df_clean.transpose()
            # Add company name, sector & ticker
            df_transpose["Company_name"] = mapping_data.loc[mapping_data["Full Ticker"] == full_ticker_list[i]]["Company name"].values[0]
            df_transpose["Sector"] = mapping_data.loc[mapping_data["Full Ticker"] == full_ticker_list[i]]["Sector_new"].values[0]
            df_transpose["Ticker_full"] = full_ticker_list[i]
            df_transpose["Ticker"] = ticker_slice_list[i]
            # Create a year column
            df_transpose["Year"] = pd.to_datetime(df_transpose.index).year
            # Dividend per share
            df_transpose["DPS"] = abs(df_transpose["Dividends_Paid"]) / df_transpose["Shares_outstanding"]
            # Implied P/E
            df_transpose["PE_Implied"] = np.nan_to_num(df_transpose["Stock_Price"] / df_transpose["Diluted_EPS"])
            # Buy back per share
            df_transpose["BBPS"] = np.nan_to_num(
                abs(df_transpose["Stock_repurchased"]) / df_transpose["Shares_outstanding"])
            # Dividends & buybacks per share
            df_transpose["DBBPS"] = abs(df_transpose["Dividends_Paid"]) / df_transpose[
                "Shares_outstanding"] + np.nan_to_num(
                abs(df_transpose["Stock_repurchased"]) / df_transpose["Shares_outstanding"])
            # Dividend Yield
            df_transpose["Dividend_Yield"] = np.nan_to_num(df_transpose["DPS"] / df_transpose["Stock_Price"])
            # Buyback yield
            df_transpose["Buyback_Yield"] = np.nan_to_num(df_transpose["BBPS"] / df_transpose["Stock_Price"])
            # Dividend + Buyback yield
            df_transpose["Dividend_Buyback_Yield"] = df_transpose["DBBPS"] / df_transpose["Stock_Price"]
            # Percentage of Debt
            df_transpose["Debt_percentage"] = df_transpose["Total_debt"] / (
                        df_transpose["Total_debt"] + df_transpose["Total_equity"])
            # Percentage of Equity
            df_transpose["Equity_percentage"] = df_transpose["Total_equity"] / (
                        df_transpose["Total_debt"] + df_transpose["Total_equity"])

            # Compute Price gain/loss
            df_transpose["Stock_gain_loss"] = (df_transpose["Stock_Price"] - df_transpose["Stock_Price"].shift(1)) / \
                                              df_transpose["Stock_Price"].shift(1)
            # Compute TSR - stock price/gain + DPS + BBPS
            df_transpose["TSR"] = ((df_transpose["Stock_Price"] - df_transpose["Stock_Price"].shift(1)) + df_transpose[
                "DBBPS"]) / df_transpose["Stock_Price"].shift(1)
            # Compute TSR (Capital_IQ)
            df_transpose["TSR_CIQ_no_buybacks"] = (
                        (df_transpose["Adjusted_Stock_Price"] - df_transpose["Adjusted_Stock_Price"].shift(1)) /
                        df_transpose["Adjusted_Stock_Price"].shift(1))
            # Net asset value
            df_transpose["Net_asset_value"] = df_transpose["Total_assets"] - df_transpose["Total_liabilities"]
            # Non-interest debt
            df_transpose["Non_interest_debt"] = df_transpose["Accounts_payable"] + df_transpose["Accrued_expenses"] + \
                                                df_transpose["Unearned_revenue"] + df_transpose["DTL"]
            # Funds employed
            df_transpose["Funds_employed"] = df_transpose["Net_asset_value"] + df_transpose["Non_interest_debt"]

            # Debt-to-equity ratio
            df_transpose["Debt_to_equity"] = df_transpose["Total_debt"] / df_transpose["Total_equity"]

            # Profit per employee
            df_transpose["NPAT_per_employee"] = df_transpose["NPAT"] / df_transpose["Number_of_employees"]

            # Include WACC data from Damodaran
            wacc_sector_i = mapping_data.loc[mapping_data["Full Ticker"] == ticker]["Sector_Damodaran"].values
            wacc_features = wacc_store.loc[wacc_store["Industry Name"] == wacc_sector_i[0]][["Cost of Debt", "Cost of Equity", "Tax Rate", "Year"]]
            # Join Cost of debt & Cost of equity sector estimates
            df_transpose = pd.merge(df_transpose, wacc_features, how='left', on='Year')

            # # Calculate the average of 'Effective_tax_rate' excluding NaN values
            # effective_tax_rate_mean = df_transpose.loc["Effective_tax_rate"].mean()
            # # Replace NaN values in 'Effective_tax_rate' with the calculated average
            # df_transpose.loc["Effective_tax_rate"].fillna(effective_tax_rate_mean, inplace=True)

            # Include new WACC column
            df_transpose["WACC_Debt_Component_Damodaran"] = ((df_transpose["Cost of Debt"] * (1 - df_transpose["Effective_tax_rate"] / 100)) * df_transpose["Debt_percentage"])
            df_transpose["WACC_Equity_Component_Damodaran"] = (df_transpose["Equity_percentage"] * df_transpose["Cost of Equity"])
            # Include debt & Equity Components
            df_transpose["WACC_Damodaran"] = df_transpose["WACC_Debt_Component_Damodaran"] + df_transpose["WACC_Equity_Component_Damodaran"]

            # Compute R&D % Revenue
            df_transpose["RD/Revenue"] = df_transpose["Research_development"] / df_transpose["Revenue"]

            # Compute Economic Profit
            df_transpose["Economic_profit"] = df_transpose["NPAT"] - df_transpose["Funds_employed"] * df_transpose[
                "WACC_Damodaran"]
            # Compute Economic Profit / Funds Employed
            df_transpose["EP/FE"] = df_transpose["Economic_profit"] / df_transpose["Funds_employed"]
            # Compute Gross margin
            df_transpose["Gross_margin"] = df_transpose["Gross_profit"] / df_transpose["Revenue"]
            # Compute profit margin
            df_transpose["Profit_margin"] = df_transpose["NPAT"] / df_transpose["Revenue"]
            # Compute return on tangible equity
            df_transpose["ROE"] = df_transpose["NPAT"] / df_transpose["Book_Value_Equity"]
            # Returns above Cost of Equity
            df_transpose["ROE_above_Cost_of_equity"] = df_transpose["ROE"] - df_transpose["Cost of Equity"]
            # Compute Return on Assets
            df_transpose["ROA"] = df_transpose["NPAT"] / df_transpose["Total_assets"]
            # Compute Return on Funds Employed
            if (df_transpose["EBIT"] == 0).any():
                df_transpose["ROFE"] = np.nan
                df_transpose["ROCE"] = np.nan
            else:
                df_transpose["ROFE"] = (df_transpose["EBIT"] / df_transpose["Revenue"]) * (
                            df_transpose["Revenue"] / df_transpose["Funds_employed"])
                # Compute Return on Capital Employed
                df_transpose["ROCE"] = df_transpose["ROFE"] * (1 - df_transpose["Effective_tax_rate"] / 100)
            # BVE/share
            df_transpose["BVE_per_share"] = df_transpose["Book_Value_Equity"] / df_transpose["Shares_outstanding"]
            # CAPEX/Sales
            df_transpose["CAPEX/Revenue"] = np.abs(df_transpose["CAPEX"]) / df_transpose["Revenue"]
            # Tangible equity
            df_transpose["Tangible_equity"] = df_transpose["Total_equity"] - df_transpose["Minority_interest"] - \
                                              df_transpose["Goodwill"] - df_transpose["Other_intangibles"]
            # CROTE
            df_transpose["CROTE"] = df_transpose["NPAT"] - (df_transpose["Tangible_equity"]) * df_transpose[
                "Cost of Equity"]
            # CROTE/TE
            df_transpose["CROTE_TE"] = df_transpose["CROTE"] / df_transpose["Tangible_equity"]
            # ROTE_
            df_transpose["ROTE"] = df_transpose["NPAT"] / df_transpose["Tangible_equity"]
            # Returns above Cost of Equity
            df_transpose["ROTE_above_Cost_of_equity"] = df_transpose["ROTE"] - df_transpose["Cost of Equity"]

            ### BESPOKE ECONOMIC PROFIT CALCULATIONS ###

            # Create Economic Profit Value
            df_transpose['EVA_bespoke'] = np.where(
                df_transpose['Sector'].isin(['Banking', 'Insurance', "Investment and Wealth", "Financials - other"]),
                df_transpose['CROTE'],
                df_transpose['Economic_profit']
            )

            # Create Economic Profit Ratio
            df_transpose['EVA_ratio_bespoke'] = np.where(
                df_transpose['Sector'].isin(['Banking', 'Insurance', "Investment and Wealth", "Financials - other"]),
                df_transpose['CROTE_TE'],
                df_transpose['EP/FE']
            )

            # Compute EVA Margin
            df_transpose["EVA_Margin"] = df_transpose["EVA_bespoke"] / df_transpose["Revenue"]
            # Calculate the first difference of the 'EVA_bespoke' column
            df_transpose['EVA_bespoke_diff'] = df_transpose['EVA_bespoke'].diff()
            # Create 'EVA_momentum' by dividing the difference by 'Revenue'
            df_transpose['EVA_momentum'] = df_transpose['EVA_bespoke_diff'] / df_transpose['Revenue'].shift(1)
            # Create 'EVA_shock' - Change in EVA Momentum
            df_transpose['EVA_shock'] = df_transpose['EVA_momentum'].diff()

            # Calculate Productivity Gains (Change in (EVA/Sales))
            df_transpose['EVA_Productivity_Gains'] = df_transpose['EVA_Margin'].diff()
            # Calculate Profitable Growth ((EVA/Sales) * Sales Growth)
            df_transpose['Sales_Growth'] = df_transpose['Revenue'].pct_change()  # Calculate sales growth as a percentage change
            df_transpose['EVA_Profitable_Growth'] = df_transpose['EVA_Margin'] * df_transpose['Sales_Growth']

            # Slice years for specific company
            unique_years = np.sort(np.unique(df_transpose["Year"]))
            # Generate grid of years
            years_grid = np.linspace(np.min(unique_years), np.max(unique_years), len(unique_years))

            list_dfs = []
            for j in range(len(years_grid)):
                # Slice on Company/Year
                slice_t1 = df_transpose.loc[df_transpose["Year"] == years_grid[j]]

                # Year t1
                pe_implied_t1 = df_transpose.loc[df_transpose["Year"] == years_grid[j]]["PE_Implied"]
                eps_t1 = df_transpose.loc[df_transpose["Year"] == years_grid[j]]["Diluted_EPS"]
                revenue_t1 = df_transpose.loc[df_transpose["Year"] == years_grid[j]]["Revenue"]
                nav_t1 = df_transpose.loc[df_transpose["Year"] == years_grid[j]]["Net_asset_value"]
                economic_profit_t1 = df_transpose.loc[df_transpose["Year"] == years_grid[j]]["Economic_profit"]
                ep_fe_t1 = df_transpose.loc[df_transpose["Year"] == years_grid[j]]["EP/FE"]
                profit_margin_t1 = df_transpose.loc[df_transpose["Year"] == years_grid[j]]["Profit_margin"]
                price_tbv_t1 = df_transpose.loc[df_transpose["Year"] == years_grid[j]]["PBV"]
                roe_t1 = df_transpose.loc[df_transpose["Year"] == years_grid[j]]["ROE"]
                roa_t1 = df_transpose.loc[df_transpose["Year"] == years_grid[j]]["ROA"]
                bve_t1 = df_transpose.loc[df_transpose["Year"] == years_grid[j]]["Book_Value_Equity"]
                bve_per_share_t1 = df_transpose.loc[df_transpose["Year"] == years_grid[j]]["BVE_per_share"]

                try:
                    # Year t0
                    pe_implied_t0 = df_transpose.loc[df_transpose["Year"] == years_grid[j] - 1]["PE_Implied"]
                    eps_t0 = df_transpose.loc[df_transpose["Year"] == years_grid[j] - 1]["Diluted_EPS"]
                    revenue_t0 = df_transpose.loc[df_transpose["Year"] == years_grid[j] - 1]["Revenue"]
                    nav_t0 = df_transpose.loc[df_transpose["Year"] == years_grid[j] - 1]["Net_asset_value"]
                    economic_profit_t0 = df_transpose.loc[df_transpose["Year"] == years_grid[j] - 1]["Economic_profit"]
                    ep_fe_t0 = df_transpose.loc[df_transpose["Year"] == years_grid[j] - 1]["EP/FE"]
                    profit_margin_t0 = df_transpose.loc[df_transpose["Year"] == years_grid[j] - 1]["Profit_margin"]
                    price_tbv_t0 = df_transpose.loc[df_transpose["Year"] == years_grid[j] - 1]["PBV"]
                    roe_t0 = df_transpose.loc[df_transpose["Year"] == years_grid[j] - 1]["ROE"]
                    roa_t0 = df_transpose.loc[df_transpose["Year"] == years_grid[j] - 1]["ROA"]
                    bve_t0 = df_transpose.loc[df_transpose["Year"] == years_grid[j] - 1]["Book_Value_Equity"]
                    bve_per_share_t0 = df_transpose.loc[df_transpose["Year"] == years_grid[j] - 1]["BVE_per_share"]

                    # Compute 1 year growth rates
                    slice_t1["PE_Implied_1_f"] = pe_implied_t1.iloc[0] / pe_implied_t0.iloc[0] - 1
                    slice_t1["Diluted_EPS_1_f"] = eps_t1.iloc[0] / eps_t0.iloc[0] - 1
                    slice_t1["Revenue_growth_1_f"] = revenue_t1.iloc[0] / revenue_t0.iloc[0] - 1
                    slice_t1["NAV_1_f"] = nav_t1.iloc[0] / nav_t0.iloc[0] - 1
                    slice_t1["Economic_profit_1_f"] = economic_profit_t1.iloc[0] / economic_profit_t0.iloc[0] - 1
                    slice_t1["EP/FE_1_f"] = ep_fe_t1.iloc[0] / ep_fe_t0.iloc[0] - 1
                    slice_t1["profit_margin_1_f"] = profit_margin_t1.iloc[0] / profit_margin_t0.iloc[0] - 1
                    slice_t1["price_tbv_1_f"] = price_tbv_t1.iloc[0] / price_tbv_t0.iloc[0] - 1
                    slice_t1["roe_1_f"] = roe_t1.iloc[0] / roe_t0.iloc[0] - 1
                    slice_t1["roa_1_f"] = roa_t1.iloc[0] / roa_t0.iloc[0] - 1
                    slice_t1["BVE_1_f"] = bve_t1.iloc[0] / bve_t0.iloc[0] - 1
                    slice_t1["BVE_per_share_1_f"] = bve_per_share_t1.iloc[0] / bve_per_share_t0.iloc[0] - 1
                except:
                    print("Cannot slice previous index")

                # Append slice to list_dfs
                list_dfs.append(slice_t1)

            # Concatenate into complete view
            merged_df = pd.concat(list_dfs)

            # 2 year growth rates
            merged_df["Revenue_growth_2_f"] = merged_df["Revenue_growth_1_f"].rolling(window=2).mean()
            merged_df["NAV_growth_2_f"] = merged_df["NAV_1_f"].rolling(window=2).mean()
            merged_df["EP_growth_2_f"] = merged_df["Economic_profit_1_f"].rolling(window=2).mean()
            merged_df["EP/FE_growth_2_f"] = merged_df["EP/FE_1_f"].rolling(window=2).mean()
            merged_df["profit_margin_growth_2_f"] = merged_df["profit_margin_1_f"].rolling(window=2).mean()
            merged_df["price_tbv_growth_2_f"] = merged_df["price_tbv_1_f"].rolling(window=2).mean()
            merged_df["roe_growth_2_f"] = merged_df["roe_1_f"].rolling(window=2).mean()
            merged_df["roa_growth_2_f"] = merged_df["roa_1_f"].rolling(window=2).mean()
            merged_df["BVE_growth_2_f"] = merged_df["BVE_1_f"].rolling(window=2).mean()
            merged_df["BVE_per_share_growth_2_f"] = merged_df["BVE_per_share_1_f"].rolling(window=2).mean()

            # 3 Year growth rates
            merged_df["Revenue_growth_3_f"] = merged_df["Revenue_growth_1_f"].rolling(window=3).mean()
            merged_df["NAV_growth_3_f"] = merged_df["NAV_1_f"].rolling(window=3).mean()
            merged_df["EP_growth_3_f"] = merged_df["Economic_profit_1_f"].rolling(window=3).mean()
            merged_df["EP/FE_growth_3_f"] = merged_df["EP/FE_1_f"].rolling(window=3).mean()
            merged_df["profit_margin_growth_3_f"] = merged_df["profit_margin_1_f"].rolling(window=3).mean()
            merged_df["price_tbv_growth_3_f"] = merged_df["price_tbv_1_f"].rolling(window=3).mean()
            merged_df["roe_growth_3_f"] = merged_df["roe_1_f"].rolling(window=3).mean()
            merged_df["roa_growth_3_f"] = merged_df["roa_1_f"].rolling(window=3).mean()
            merged_df["BVE_growth_3_f"] = merged_df["BVE_1_f"].rolling(window=3).mean()
            merged_df["BVE_per_share_growth_3_f"] = merged_df["BVE_per_share_1_f"].rolling(window=3).mean()

            # Write to CSV
            # Create the folder if it doesn't exist
            # os.makedirs(rf"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\{country_import}", exist_ok=True)
            folder_path = os.path.join(GLOBAL_DATA_DIR, f"{country_import}")
            os.makedirs(folder_path, exist_ok=True)
            # Save the file
            # merged_df.to_csv(rf"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\{country_import}\_" + company_label + ".csv")
            merged_df.to_csv(os.path.join(folder_path, f"_{company_label}.csv"))

            # Now get Adjusted Close Price & save separately
            req_array = [
                {"Function": "GDST", "Identifier": company, "Mnemonic": "IQ_CLOSEPRICE_ADJ",
                 "properties": {"frequency": "Daily", "startDate": "02/02/2019", "endDate": "02/02/2025","CurrencyID": "USD"}}]

            req = {"inputRequests": req_array}
            response = requests.post(endpoint_url,
                                     headers={'Content-Type': 'application/json'},
                                     data=json.dumps(req),
                                     auth=HTTPBasicAuth("apiadmin@bain.com", "Bain@1234"),
                                     verify=True)
            print(response)

            # Slice date and share price
            dates = pd.DataFrame(json.loads(response.text)['GDSSDKResponse'][0]["Headers"])
            share_price = json.loads(response.text)['GDSSDKResponse'][0]['Rows'][0]['Row']
            share_price_adjusted = pd.DataFrame([float(i) for i in share_price])

            # Write prices to local directory
            price_df = pd.concat([dates, share_price_adjusted], axis=1)
            price_df.columns = ["Date", "Price"]
            # os.makedirs(rf"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\share_price\{country_import}", exist_ok=True)
            folder_path = os.path.join(GLOBAL_DATA_DIR, "share_price", f"{country_import}")
            os.makedirs(folder_path, exist_ok=True)
            # price_df.to_csv(rf"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\share_price\{country_import}\_" + company_label +"_price.csv")
            price_df.to_csv(os.path.join(folder_path, f"_{company_label}_price.csv"))

        except:
            print("Issue with ticker ", full_ticker_list[i])