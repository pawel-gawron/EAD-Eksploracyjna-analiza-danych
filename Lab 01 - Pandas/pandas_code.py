# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import re

# df_population = pd.read_csv("/home/pawel/Documents/RISA/sem2/EAD/pandas/population_by_country_2019_2020.csv")
# # print(df_population)

# df_population_summary = df_population.describe()

# # print(df_population_summary)

# # Bezwzgledna
# df_population["Net population change"] = df_population["Population (2020)"] - df_population["Population (2019)"]

# # Wzgledna
# df_population["Population change [%]"] = (df_population["Population (2020)"] - df_population["Population (2019)"])/((df_population["Population (2019)"] + df_population["Population (2020)"])/2)

# # Najwiekszy, bezwzgledny przyrost naturalny
# print(df_population["Net population change"].max())
# countryLowestNetPopulationChange = df_population.loc[df_population["Net population change"] == df_population["Net population change"].max()]["Country (or dependency)"]
# print(countryLowestNetPopulationChange)

# # Najwiekszy wzgledny ujemny przyrost naturalny
# print(df_population["Population change [%]"].min())
# countryLowestNetPopulationChange = df_population.loc[df_population["Population change [%]"] == df_population["Population change [%]"].min()]["Country (or dependency)"]
# print(countryLowestNetPopulationChange)

# df_population_sorted = df_population.sort_values(by=["Population change [%]"], axis=0, ascending=False)
# print(df_population_sorted)

# # print("adwadawd: ", df_population.loc[:, ["Population change [%]"]])

# # df_population_sorted.iloc[:10, df_population["Population change [%]"]].plot(kind='bar')

# df_population_sorted.dropna(subset=["Population change [%]"], inplace=True)

# # Extract columns for population in 2019 and 2020 using regex
# pop_cols = df_population.filter(regex='Population \(2019|2020\)', axis=1).columns

# # Plot two bars for every country, one for population from 2019 and one for population from 2020
# df_population_sorted.iloc[:10].plot(x="Country (or dependency)", y=pop_cols, kind='bar')
# plt.show()

# df_population["Density (2020)"] = "Low"

# # if df_population[]

# df_population["Density (2020)"] = np.where(df_population["Population (2020)"]/df_population["Land Area (Km²)"] > 500, "High", "Low")

# df_population.iloc[::2].to_csv("population_output.csv")

# # print(df_population["Density (2020)"].values)
# # import re

# # pattern = r"Population \((2019|2020)\)"

import requests
url = "https://api.openweathermap.org/data/2.5/weather"
api_key = "cf9b530bfb0d75e9c836f808cac693b3"
latitude = 37.2431
longitude = -115.7930
req = requests.get(f"{url}?lat={latitude}&lon={longitude}&exclude=minutely&appid={api_key}")
print(req.text)