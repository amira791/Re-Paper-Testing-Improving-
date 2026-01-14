import pandas as pd
import numpy as np
import os
import glob
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Set paths
data_path = r"C:\Users\admin\Desktop\DR2\3 Coding\Re-Paper-Testing-Improving-\Paper 05 (10 EVs)\dataset_10EVs"
output_path = r"C:\Users\admin\Desktop\DR2\3 Coding\Re-Paper-Testing-Improving-\Paper 05 (10 EVs)\processed_data"

# Create output directory
os.makedirs(output_path, exist_ok=True)

# Vehicle metadata from paper (Table 1)
vehicle_metadata = {
    'vehicle#1': {'type': 'passenger', 'chemistry': 'NCM', 'capacity': 150, 'sampling_freq': 0.1},
    'vehicle#2': {'type': 'passenger', 'chemistry': 'NCM', 'capacity': 150, 'sampling_freq': 0.1},
    'vehicle#3': {'type': 'passenger', 'chemistry': 'NCM', 'capacity': 160, 'sampling_freq': 0.1},
    'vehicle#4': {'type': 'passenger', 'chemistry': 'NCM', 'capacity': 160, 'sampling_freq': 0.1},
    'vehicle#5': {'type': 'passenger', 'chemistry': 'NCM', 'capacity': 160, 'sampling_freq': 0.1},
    'vehicle#6': {'type': 'passenger', 'chemistry': 'NCM', 'capacity': 160, 'sampling_freq': 0.1},
    'vehicle#7': {'type': 'passenger', 'chemistry': 'LFP', 'capacity': 120, 'sampling_freq': 0.5},
    'vehicle#8': {'type': 'bus', 'chemistry': 'LFP', 'capacity': 645, 'sampling_freq': 0.1},
    'vehicle#9': {'type': 'bus', 'chemistry': 'LFP', 'capacity': 505, 'sampling_freq': 0.1},
    'vehicle#10': {'type': 'bus', 'chemistry': 'LFP', 'capacity': 505, 'sampling_freq': 0.1}
}