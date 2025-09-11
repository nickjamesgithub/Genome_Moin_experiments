import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import math
import matplotlib
matplotlib.use('TkAgg')

# ----------------- Switches -----------------
make_plots = True         # set True to draw plots
genome_filtering = True    # set False to skip bounds filtering
# -------------------------------------------

# Years / window
beginning_year = 2011
end_year = 2025
year_grid = np.linspace(beginning_year, end_year, end_year - beginning_year + 1, dtype=int)
rolling_window = 3

# ===== Load INSURANCE file =====
stacked = pd.read_csv(
    r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\Clean\Clean_insurance_data_genome_country.csv"
)

# Filter scope
countries_to_include = [
    "Australia","Belgium","Canada","Chile","China","Denmark","France","Germany","Hong_Kong",
    "India","Italy","Japan","Luxembourg","Malaysia","Netherlands","Philippines","Saudi_Arabia",
    "Singapore","South_Korea","Switzerland","Sweden","Thailand","UAE","USA","United_Kingdom",
]
sectors_to_include = [
    "Banking","Communication Services","Consumer Discretionary","Consumer Staples","Diversified",
    "Energy","Financials","Health Care","Healthcare","Industrials","Insurance","Investment and Wealth",
    "Materials","Real Estate","Technology","Telecommunication","Transportation","Utilities",
]

# Use the same variable name as your original script after filtering
data = stacked.loc[
    stacked["Country"].isin(countries_to_include) & stacked["Sector"].isin(sectors_to_include)
].copy()

# Derived feature
data["Price_to_Book"] = data["Market_Capitalisation"] / data["Book_Value_Equity"]

# Features to slice
features = [
    "Company_name","Country","Sector","Year","TSR","Revenue_growth_3_f","Stock_Price","Adjusted_Stock_Price",
    "DPS","BBPS","DBBPS","Genome_classification_bespoke","Price_to_Book","PE_Implied","Market_Capitalisation",
    "Revenue","Debt_to_equity","RD/Revenue","EVA_ratio_bespoke","ROE_above_Cost_of_equity","CROTE_TE",
    "Cash_acquisitions","NPAT_per_employee","CAPEX/Revenue","Gross_margin"
]

# Functions (unchanged logic)
def sector_functions_mobility_matrix(df):
    unique_companies = df["Company_name"].unique()
    feats = []
    for name in unique_companies:
        s = df.loc[df["Company_name"] == name]
        rev_avg = s["Revenue"].iloc[1:].mean()
        if len(s) > 1 and s["Revenue"].iloc[0] != 0:
            rev_cagr = (s["Revenue"].iloc[-1] / s["Revenue"].iloc[0]) ** (1/(len(s)-1)) - 1
        else:
            rev_cagr = np.nan
        if len(s) > 1:
            tsr = (s["Adjusted_Stock_Price"].iloc[-1] / s["Adjusted_Stock_Price"].iloc[0]) ** (1/(len(s)-1)) - 1
        else:
            tsr = np.nan
        leverage = s["Debt_to_equity"].iloc[1:].mean()
        invest = s["RD/Revenue"].iloc[1:].mean()
        eva_avg = s["EVA_ratio_bespoke"].iloc[1:].mean()
        acq_prop = abs(s["Cash_acquisitions"].iloc[1:]).sum() / s["Market_Capitalisation"].iloc[1:].mean()
        capex_rev = s["CAPEX/Revenue"].iloc[1:].mean()
        npat_emp = s["NPAT_per_employee"].iloc[1:].mean()
        gm = s["Gross_margin"].iloc[1:].mean()
        feats.append([rev_avg, rev_cagr, leverage, invest, eva_avg, acq_prop, capex_rev, npat_emp, gm, tsr])
    feats_df = pd.DataFrame(feats)
    feats_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return feats_df.mean(skipna=True)

def company_functions_mobility_matrix(df):
    rev_avg = df["Revenue"].iloc[1:].mean()
    rev_cagr = (df["Revenue"].iloc[-1] / df["Revenue"].iloc[0]) ** (1/(len(df)-1)) - 1
    tsr = (df["Adjusted_Stock_Price"].iloc[-1] / df["Adjusted_Stock_Price"].iloc[0]) ** (1/(len(df)-1)) - 1
    leverage = df["Debt_to_equity"].iloc[1:].mean()
    invest = df["RD/Revenue"].iloc[1:].mean()
    eva_avg = df["EVA_ratio_bespoke"].iloc[1:].mean()
    acq_prop = abs(df["Cash_acquisitions"].iloc[1:]).sum() / df["Market_Capitalisation"].iloc[1:].mean()
    capex_rev = df["CAPEX/Revenue"].iloc[1:].mean()
    npat_emp = df["NPAT_per_employee"].iloc[1:].mean()
    gm = df["Gross_margin"].iloc[1:].mean()
    return rev_avg, rev_cagr, leverage, invest, eva_avg, acq_prop, capex_rev, npat_emp, gm, tsr

# Prep
unique_tickers = data["Ticker_full"].dropna().unique()
df_list, df_issue_list = [], []

# Main loop
for i in range(len(year_grid) - rolling_window):
    year_i = int(year_grid[i])
    year_i_2 = int(year_grid[i + rolling_window])
    for j in range(len(unique_tickers)):
        unique_ticker_j = unique_tickers[j]
        company_name_j = None  # ensure defined for exception logging

        try:
            # Slices at endpoints
            df_slice_i = data.loc[(data["Year"] == year_i) & (data["Ticker_full"] == unique_ticker_j), features]
            df_slice_i_2 = data.loc[(data["Year"] == year_i_2) & (data["Ticker_full"] == unique_ticker_j), features]

            if df_slice_i.empty or df_slice_i_2.empty:
                df_issue_list.append([unique_ticker_j, year_i, year_i_2, "missing endpoint"])
                continue

            # Company info from end year
            company_info = df_slice_i_2
            company_name_j = company_info["Company_name"].values[0]
            sector_j = df_slice_i["Sector"].values[0]
            country_j = df_slice_i["Country"].values[0]

            # Ranges
            df_slice_company_range_i = data.loc[
                (data["Year"] >= year_i) & (data["Year"] <= year_i_2) & (data["Ticker_full"] == unique_ticker_j),
                features
            ]
            df_slice_sector_range_i = data.loc[
                (data["Year"] >= year_i) & (data["Year"] <= year_i_2) & (data["Sector"] == sector_j),
                features
            ]
            if df_slice_company_range_i.empty or df_slice_sector_range_i.empty:
                df_issue_list.append([unique_ticker_j, year_i, year_i_2, "missing range"])
                continue

            # Sector stats
            (sector_revenue_avg, sector_revenue_cagr, sector_leverage, sector_investment, sector_eva_avg,
             sector_acquisition_propensity, sector_capex_per_revenue, sector_npat_per_employee,
             sector_gross_margin, sector_tsr) = sector_functions_mobility_matrix(df_slice_sector_range_i)

            # Company stats
            (company_revenue_avg, company_revenue_cagr, company_leverage, company_investment, company_eva_avg,
             company_acquisition_propensity, company_capex_per_revenue, company_npat_per_employee,
             company_gross_margin, company_tsr) = company_functions_mobility_matrix(df_slice_company_range_i)

            # Deltas
            delta_revenue_avg = company_revenue_avg - sector_revenue_avg
            delta_revenue_cagr = company_revenue_cagr - sector_revenue_cagr
            delta_leverage = company_leverage - sector_leverage
            delta_investment = company_investment - sector_investment
            delta_eva_avg = company_eva_avg - sector_eva_avg
            delta_acquisition_propensity = company_acquisition_propensity - sector_acquisition_propensity
            delta_capex_per_revenue = company_capex_per_revenue - sector_capex_per_revenue
            delta_npat_per_employee = company_npat_per_employee - sector_npat_per_employee
            delta_gross_margin = company_gross_margin - sector_gross_margin

            # Labels / values
            genome_classification_bespoke_beginning = df_slice_i["Genome_classification_bespoke"].values[0]
            genome_classification_bespoke_end = df_slice_i_2["Genome_classification_bespoke"].values[0]
            pe_implied_beginning = df_slice_i["PE_Implied"].values[0]
            pe_implied_end = df_slice_i_2["PE_Implied"].values[0]
            market_capitalisation_beginning = df_slice_i["Market_Capitalisation"].values[0]
            market_capitalisation_end = df_slice_i_2["Market_Capitalisation"].values[0]
            firefly_y_beginning = df_slice_i["EVA_ratio_bespoke"].values[0]
            firefly_y_end = df_slice_i_2["EVA_ratio_bespoke"].values[0]
            firefly_x_beginning = df_slice_i["Revenue_growth_3_f"].values[0]
            firefly_x_end = df_slice_i_2["Revenue_growth_3_f"].values[0]

            # Angle
            radians = math.atan2(firefly_y_end - firefly_y_beginning, firefly_x_end - firefly_x_beginning)
            degree_angle = math.degrees(radians)
            degrees = (degree_angle + 360) % 360

            # DBBPS total
            dbbps_slice = data.loc[
                (data["Year"] >= year_i) & (data["Year"] <= year_i_2) & (data["Ticker_full"] == unique_ticker_j), "DBBPS"
            ]
            dbbps_total = dbbps_slice.sum(skipna=True)

            # Cumulative & annualized TSRs
            def _tsr(stock_end, stock_begin, add_cash):
                cum_tsr = (stock_end - stock_begin + add_cash) / stock_begin
                return cum_tsr, (1 + cum_tsr) ** (1/rolling_window) - 1

            cum_tsr, annualized_tsr = _tsr(
                df_slice_i_2["Stock_Price"].values[0], df_slice_i["Stock_Price"].values[0], dbbps_total
            )
            cum_tsr_cq = df_slice_i_2["Adjusted_Stock_Price"].values[0] / df_slice_i["Adjusted_Stock_Price"].values[0] - 1
            annualized_tsr_capiq = (1 + cum_tsr_cq) ** (1/rolling_window) - 1

            # Scenario bucket
            eva_beg, eva_end = df_slice_i["EVA_ratio_bespoke"].values[0], df_slice_i_2["EVA_ratio_bespoke"].values[0]
            if eva_beg < 0 and eva_end < 0:
                journey = "Remain_negative"
            elif eva_beg < 0 and eva_end >= 0:
                journey = "Move_up"
            elif eva_beg >= 0 and eva_end >= 0:
                journey = "Remain_positive"
            else:
                journey = "Move_down"

            # Append row
            df_list.append([
                journey, genome_classification_bespoke_beginning, genome_classification_bespoke_end,
                company_name_j, unique_ticker_j, country_j, sector_j, int(year_i), int(year_i_2),
                firefly_x_beginning, firefly_x_end, firefly_y_beginning, firefly_y_end,
                df_slice_i["ROE_above_Cost_of_equity"].values[0], df_slice_i_2["ROE_above_Cost_of_equity"].values[0],
                df_slice_i["CROTE_TE"].values[0], df_slice_i_2["CROTE_TE"].values[0],
                df_slice_i["Stock_Price"].values[0], df_slice_i_2["Stock_Price"].values[0],
                df_slice_i["Price_to_Book"].values[0], dbbps_total, degrees, radians,
                annualized_tsr, annualized_tsr_capiq, pe_implied_beginning, pe_implied_end,
                market_capitalisation_beginning, market_capitalisation_end,
                company_revenue_avg, sector_revenue_avg, delta_revenue_avg,
                company_revenue_cagr, sector_revenue_cagr, delta_revenue_cagr,
                company_leverage, sector_leverage, delta_leverage,
                company_investment, sector_investment, delta_investment,
                company_eva_avg, sector_eva_avg, delta_eva_avg,
                company_acquisition_propensity, sector_acquisition_propensity, delta_acquisition_propensity,
                company_capex_per_revenue, sector_capex_per_revenue, delta_capex_per_revenue,
                company_npat_per_employee, sector_npat_per_employee, delta_npat_per_employee,
                company_gross_margin, sector_gross_margin, delta_gross_margin, sector_tsr, company_tsr
            ])

        except Exception as e:
            print(f"Issue during processing: Ticker {unique_ticker_j} Years: {year_i}-{year_i_2} | {e}")
            df_issue_list.append([unique_ticker_j, year_i, year_i_2, "exception"])

# Collapsed journeys
df_journey_collapsed = pd.DataFrame(df_list)
df_journey_collapsed.columns = [
    "Journey","Genome_classification_bespoke_beginning","Genome_classification_bespoke_end",
    "Company_name","Ticker","Country","Sector","Year_beginning","Year_final",
    "Revenue_growth_beginning","Revenue_growth_end","EVA_beginning","EVA_end",
    "ROE_above_Cost_of_equity_beginning","ROE_above_Cost_of_equity_end",
    "CROTE_TE_beginning","CROTE_TE_end",
    "Stock_price_beginning","Stock_price_final","Price_to_book","DBBPS_total","Angle","Radians",
    "Annualized_TSR","Annualized_TSR_Capiq","PE_beginning","PE_end","Market_Capitalisation_beginning","Market_Capitalisation_end",
    "Company_revenue_avg","Sector_revenue_avg","Delta_revenue_avg",
    "Company_revenue_cagr","Sector_revenue_cagr","Delta_revenue_cagr",
    "Company_leverage","Sector_leverage","Delta_leverage",
    "Company_investment","Sector_investment","Delta_investment",
    "Company_eva_avg","Sector_eva_avg","delta_eva_avg",
    "Company_acquisition_propensity","Sector_acquisition_propensity","delta_acquisition_propensity",
    "Company_capex/revenue","Sector_capex/revenue","Delta_capex/revenue",
    "Company_npat_per_employee","Sector_npat_per_employee","Delta_npat_per_employee",
    "Company_gross_margin","Sector_gross_margin","Delta_gross_margin","Sector_TSR","Company_TSR"
]

# Clean & optional filter
df_journey_collapsed = df_journey_collapsed.replace([np.inf, -np.inf], np.nan)
df_journey_collapsed = df_journey_collapsed.dropna(subset=["Annualized_TSR_Capiq"])

# Bounds filter (same as your code)
if genome_filtering:
    df_journey_collapsed = df_journey_collapsed.loc[
        (df_journey_collapsed["EVA_end"] >= -0.3) & (df_journey_collapsed["EVA_end"] <= 0.5) &
        (df_journey_collapsed["Revenue_growth_end"] >= -0.3) & (df_journey_collapsed["Revenue_growth_end"] <= 1.5) &
        (df_journey_collapsed["Annualized_TSR_Capiq"] >= -0.4) & (df_journey_collapsed["Annualized_TSR_Capiq"] <= 1) &
        (df_journey_collapsed["Price_to_book"] > -200)
    ]

# Add delta X/Y (as in your code)
df_journey_collapsed["X_change"] = (
    df_journey_collapsed["Revenue_growth_end"] - df_journey_collapsed["Revenue_growth_beginning"]
)
df_journey_collapsed["Y_change"] = (
    df_journey_collapsed["EVA_end"] - df_journey_collapsed["EVA_beginning"]
)

print("Built df_journey_collapsed:", df_journey_collapsed.shape)
print(df_journey_collapsed.head(3))

# Write out csv file locally
df_journey_collapsed.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\Clean\Journeys_summary_Global_FE_Update.csv")

# ----------------- Plots (optional) -----------------
if make_plots:
    # Scatter + KDE contours by Journey
    for journey in df_journey_collapsed["Journey"].dropna().unique():
        df_f = df_journey_collapsed[df_journey_collapsed["Journey"] == journey].copy()
        df_f = df_f.replace([np.inf, -np.inf], np.nan).dropna(subset=["Angle","Annualized_TSR_Capiq"])
        if df_f.empty:
            continue
        plt.figure(figsize=(8,6))
        plt.scatter(df_f["Angle"], df_f["Annualized_TSR_Capiq"], alpha=0.5)
        kde = gaussian_kde(df_f[["Angle","Annualized_TSR_Capiq"]].T, bw_method=0.3)
        xi, yi = np.mgrid[df_f["Angle"].min():df_f["Angle"].max():100j,
                          df_f["Annualized_TSR_Capiq"].min():df_f["Annualized_TSR_Capiq"].max():100j]
        zi = kde(np.vstack([xi.flatten(), yi.flatten()]))
        plt.contour(xi, yi, zi.reshape(xi.shape), colors='k')
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Annualized TSR (CapIQ)")
        plt.title(f"Annualized TSR vs Angle — Journey: {journey}")
        plt.grid(True)
        plt.show()

    # Density curves with medians by Journey
    plt.figure(figsize=(10,8))
    journey_colors, journey_medians = {}, {}
    for journey in df_journey_collapsed["Journey"].dropna().unique():
        df_f = df_journey_collapsed[df_journey_collapsed["Journey"] == journey].copy()
        df_f = df_f.replace([np.inf, -np.inf], np.nan).dropna(subset=["Annualized_TSR_Capiq"])
        if df_f.empty:
            continue
        tsr_vector = df_f["Annualized_TSR_Capiq"] * 100
        kde = gaussian_kde(tsr_vector, bw_method=0.3)
        x = np.linspace(tsr_vector.min(), tsr_vector.max(), 200)
        y = kde(x)
        plt.plot(x, y, label=journey)
        color = plt.gca().lines[-1].get_color()
        journey_colors[journey] = color
        median_tsr = np.median(tsr_vector)
        journey_medians[journey] = median_tsr
    plt.xlabel("Annualized TSR % (CapIQ)")
    plt.ylabel("Density")
    plt.title("AIA_Insurance_Distribution_TSR_by_Journey")
    plt.grid(True)
    for journey, color in journey_colors.items():
        plt.axvline(journey_medians[journey], linestyle='--', color=color, alpha=0.7,
                    label=f"{journey} median: {journey_medians[journey]:.2f}%")
    plt.legend()
    plt.show()
