import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

# Load the main data
df_full = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\Global_data.csv")

# Load the inflation data
inflation_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\WACC_inputs\Global_inflation_data.csv")

# Reshape the inflation data to facilitate merging
inflation_data_melted = inflation_data.melt(id_vars='Year', var_name='Country', value_name='Inflation')

# Map the country names in the inflation data to match those in the main DataFrame
country_map = {'Australia': 'AUS', 'USA': 'USA', 'Europe': 'EURO', 'UK': 'UK', 'India': 'INDIA', 'Japan': 'JAPAN'}
inflation_data_melted['Country'] = inflation_data_melted['Country'].map(country_map)

# Merge the main DataFrame with the inflation data
df_full = pd.merge(df_full, inflation_data_melted, how='left', on=['Year', 'Country'])

# Filter data for the required years
start_year = 2014
end_year = 2024
df = df_full[(df_full['Year'] >= start_year) & (df_full['Year'] <= end_year)]

# Drop duplicates by considering only the first occurrence of each company for each year
df = df.drop_duplicates(subset=['Company_name', 'Year'])

# Apply the criteria and create a new column for whether criteria are met each year
df['SVC_Criteria_Met'] = (df['EVA_ratio_bespoke'] > 0) & (df['Revenue_growth_1_f'] > df['Inflation'])

# Group by company and count the number of years criteria is met
criteria_count = df.groupby('Company_name')['SVC_Criteria_Met'].sum()

# Create a dictionary to collect companies by the number of years they meet the criteria
svc_summary_dict = {i: [] for i in range(0, end_year - start_year + 2)}  # from 0 to 11 years

for company, count in criteria_count.items():
    svc_summary_dict[count].append(company)

# Convert dictionary into a DataFrame for display
svc_summary = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in svc_summary_dict.items()])).fillna('')
svc_summary.columns = ["SVC_0", "SVC_1", "SVC_2", "SVC_3", "SVC_4", "SVC_5", "SVC_6", "SVC_7", "SVC_8", "SVC_9", "SVC_10", "SVC_11"]
print(svc_summary)

# Write out to local file
svc_summary.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\svc_summary_global.csv")

### Test if valuation is associated with SVC ###
# Initialize dictionary to store median PBV values for each SVC category
pbv_medians = {}
# Iterate through SVC categories (columns) in svc_summary
for column in svc_summary.columns:
    # Get the list of companies for the current SVC category
    companies = svc_summary[column].dropna().tolist()
    # Filter df for these companies
    filtered_df = df[df["Company_name"].isin(companies)]
    # Compute the median PBV and store it in the dictionary
    pbv_medians[column] = filtered_df["PBV"].median()
# Convert the results dictionary to a DataFrame
pbv_summary = pd.DataFrame(list(pbv_medians.items()), columns=['SVC_Category', 'Median_PBV'])
# Print the resulting DataFrame
print(pbv_summary)

# Plot SVC vs Median PBV
plt.plot(pbv_summary["SVC_Category"][:-1], pbv_summary["Median_PBV"][:-1])
plt.ylabel("Average P:BV")
plt.xlabel("SVC Years")
plt.title("SVC Criteria met vs P:BV")
plt.show()

x=1
y=2