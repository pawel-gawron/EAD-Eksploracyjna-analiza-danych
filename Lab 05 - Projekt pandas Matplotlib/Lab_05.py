import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

folder_path = './data/names'

file_list = glob.glob(os.path.join(folder_path, '*.txt'))

names_df = pd.DataFrame()
names_df_list = []

for file in file_list:
    data = pd.read_csv(file)
    names_df_list.append(data)

names_df = pd.concat(names_df_list, ignore_index=True)

print(names_df)