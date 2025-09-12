import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import math
import matplotlib
matplotlib.use('TkAgg')

### Study of valuation over time ###

df = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\Clean\Journeys_summary_Global_FE_Update.csv")

x=1
y=2