import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Load data
df = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\Clean\AIA_Journeys_summary_Global_FE_Update.csv")

# Step 1: Filter for Banking sector
df_insurance = df[(df["Sector"] == "Insurance") & (df["Year_final"]<=2025)]
# df_banking = df_insurance.loc[df_insurance["Year_final"]<2023]

# Step 3: Define "above the line" segments
above_line_segments = ["CHALLENGED", "VIRTUOUS", "FAMOUS", "LEGENDARY"]

# Step 4: Create stay_above and fall_below groups
stay_above = df_insurance[
    (df_insurance["Genome_classification_bespoke_beginning"].isin(above_line_segments)) &
    (df_insurance["Genome_classification_bespoke_end"].isin(above_line_segments))
]

fall_below = df_insurance[
    (df_insurance["Genome_classification_bespoke_beginning"].isin(above_line_segments)) &
    (~df_insurance["Genome_classification_bespoke_end"].isin(above_line_segments))
]

# Step 5: Function to plot distributions with % formatting
def plot_distribution(data1, data2, label1, label2, column, title, xlabel):
    plt.figure(figsize=(10, 6))
    plt.hist(data1[column].dropna(), bins=20, alpha=0.6, label=label1, density=True, color='blue')
    plt.hist(data2[column].dropna(), bins=20, alpha=0.6, label=label2, density=True, color='orange')

    mean1 = data1[column].mean()
    mean2 = data2[column].mean()
    plt.axvline(mean1, color='blue', linestyle='--', linewidth=2, label=f'{label1} Mean: {mean1 * 100:.2f}%')
    plt.axvline(mean2, color='orange', linestyle='--', linewidth=2, label=f'{label2} Mean: {mean2 * 100:.2f}%')

    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x * 100:.0f}%'))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("AIA_Clean_TSR_Transition.png")
    plt.show()

# Step 6: Plot TSR distribution
plot_distribution(
    stay_above, fall_below,
    "Stayed Above the Line", "Fell Below the Line",
    "Annualized_TSR_Capiq",
    "Distribution of Annualized TSR: Stay Above vs Fall Below",
    "Annualized TSR"
)

# Step 7: Plot average TSR by country with % y-axis
# Combine with group labels
combined = pd.concat([
    stay_above.assign(Group='Stayed Above'),
    fall_below.assign(Group='Fell Below')
])

# Group and calculate mean TSR by country
tsr_by_country = (
    combined
    .groupby(['Country_label', 'Group'])['Annualized_TSR_Capiq']
    .mean()
    .unstack()
    .sort_index()
)

# Plot country comparison
ax = tsr_by_country.plot(kind='bar', figsize=(12, 6))
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y * 100:.0f}%'))
plt.title("Average TSR by Country: Stay Above vs Fall Below")
plt.ylabel("Average Annualized TSR (CIQ)")
plt.xlabel("Country")
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("AIA_Clean_TSR_Transition_by_Country.png")
plt.show()
