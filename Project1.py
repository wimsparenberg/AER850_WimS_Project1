import pandas as pd
import numpy as np
import matplotlib as plt

#1 - Data Processing
df = pd.read_csv("Data/Project 1 Data.csv")

#2 - Data Visualization
desc_stat = df[['X', 'Y', 'Z',]].describe()
desc_stat_step = df['Step'].value_counts().sort_index()
df.hist()

