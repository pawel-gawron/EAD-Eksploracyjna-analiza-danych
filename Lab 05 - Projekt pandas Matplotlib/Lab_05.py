import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import sqlite3

#######################################################
# Funkcja wykorzystywana w zadaniu 10 do automatyzacji
# wykonywania tych samych operacji na zmiennych
# zakresach lat
#######################################################
def aggregated_trend(years, selected_years, selected_years_female_names, selected_years_male_names):
    common_names_table = []

    for year in selected_years['Year'].unique():
        female_data = selected_years_female_names.loc[year]
        male_data = selected_years_male_names.loc[year]
        common_names = female_data.index[female_data.index.isin(male_data.index)].values
        year_data = pd.DataFrame({'Year': [year] * len(common_names), 'Name': common_names})
        year_data = pd.concat([year_data], axis=1)
        common_names_table.append(year_data)

    common_names_table = pd.concat(common_names_table, ignore_index=True)


    filtered_names = [common_names_table[common_names_table['Year'] == year]['Name'].values for year in years]

    filtered_names_df_female = names_df_pivot.loc[(years, 'F'), :]
    filtered_names_df_male = names_df_pivot.loc[(years, 'M'), :]

    column_names_male = ['Quanity male']
    column_names_female = ['Quanity female']

    common_female = pd.concat([filtered_names_df_female.loc[(year, 'F', list(filtered_names[i])), :] for i, year in enumerate(years)]).drop(columns=['frequency_female', 'frequency_male'])
    common_male = pd.concat([filtered_names_df_male.loc[(year, 'M', list(filtered_names[i])), :] for i, year in enumerate(years)]).drop(columns=['frequency_female', 'frequency_male'])

    common_female.columns = column_names_female
    common_male.columns = column_names_male

    common_female = common_female.reset_index()
    common_male = common_male.reset_index()

    merged_pivot = common_female.combine_first(common_male)

    plot_merged_pivot = merged_pivot.copy()

    merged_pivot = merged_pivot.pivot_table(index=['Name'], values=['Quanity female', 'Quanity male'], aggfunc='sum')

    merged_pivot['frequency_male'] = merged_pivot['Quanity male'] / (merged_pivot['Quanity male'].sum() + merged_pivot['Quanity female'].sum())
    merged_pivot['frequency_female'] = merged_pivot['Quanity female'] / (merged_pivot['Quanity male'].sum() + merged_pivot['Quanity female'].sum())

    merged_pivot['Ratio'] = abs(merged_pivot['frequency_male'] - merged_pivot['frequency_female'])

    max_2_diff = merged_pivot['Ratio'].nlargest(2)

    name = max_2_diff.index

    print('######################Zad10#######################')
    print(f'Names with biggest difference between {years[0]} and {years[-1]}')
    print(name.values)
    print('#################################################\n')

    diff = [plot_merged_pivot[plot_merged_pivot['Name'] == name[0]]['Quanity female'] + plot_merged_pivot[plot_merged_pivot['Name'] == name[0]]['Quanity male'],
            plot_merged_pivot[plot_merged_pivot['Name'] == name[1]]['Quanity female'] + plot_merged_pivot[plot_merged_pivot['Name'] == name[1]]['Quanity male']]

    fig, ax = plt.subplots(figsize=(16, 6))

    ax.plot(years, diff[0], label=name[0])
    ax.plot(years, diff[1], label=name[1])
    ax.set_xlabel('Year')
    ax.set_ylabel('Popularity')
    ax.set_title(f'Name with biggest difference between male and female ratio aggregated {years[0]} - {years[-1]}')
    ax.grid(True)

    ax.legend(loc='best')

    return fig, ax

#######################################################
# Zadanie 1 Wczytywanie danych
#######################################################

folder_path = './data/names'

file_list = glob.glob(os.path.join(folder_path, '*.txt'))

names_df_list = []

for file in file_list:
    data = pd.read_csv(file, delimiter=',', header=None)
    year = int(os.path.basename(file).split('.')[0][-4:])
    data = pd.concat([data, pd.Series([year] * len(data))], axis=1)
    names_df_list.append(data)

names_df = pd.concat(names_df_list, ignore_index=True)

column_names = ['Name', 'Sex', 'Quanity', 'Year']

names_df.columns = column_names

#######################################################
# Zadanie 2/3 Unikalne imiona z rozroznieniem płci i bez rozroznienia płci
#######################################################

num_uniq_names = len(names_df['Name'].unique())
num_uniq_names_female = len(names_df.loc[names_df['Sex'] == 'F']['Name'].unique())
num_uniq_names_male = len(names_df.loc[names_df['Sex'] == 'M']['Name'].unique())

print('######################Zad2/3#######################')
print("Number of unique names across data:", num_uniq_names)
print("Number of unique Female names across data:", num_uniq_names_female)
print("Number of unique Male names across data:", num_uniq_names_male)
print('#################################################\n')

#######################################################
# Zadanie 4 Obliczenie czestotliwosci dla kazdego imienia
#######################################################

names_df_pivot = names_df.pivot_table(index=['Year', 'Sex', 'Name'], values='Quanity')

births_by_year = names_df_pivot.groupby(['Year', 'Sex']).sum()

names_male_freq = names_df_pivot.loc[(slice(None), 'M'), :] / births_by_year.loc[(slice(None), 'M'), :]
names_female_freq = names_df_pivot.loc[(slice(None), 'F'), :] / births_by_year.loc[(slice(None), 'F'), :]


names_df_pivot['frequency_male'] = names_male_freq
names_df_pivot['frequency_female'] = names_female_freq

#######################################################
# Zadanie 5 Wykresy pokazujace liczbe urodzin i stosunek urodzin kobiet do mezczyzn
#######################################################
fig, ax = plt.subplots(2, 1, figsize=(20, 8))

years = names_df_pivot.index.get_level_values('Year').unique()
sum_by_year = names_df.groupby(['Year'])['Quanity'].sum()

ax[0].plot(years, sum_by_year.values, label='Birth quanity')
ax[0].set_xlabel('Year')
ax[0].set_ylabel('Births')
ax[0].set_title('Number of births by year')
ax[0].legend(loc='best')
ax[0].grid(True)

sum_by_year_female = names_df.groupby(['Year', 'Sex'])['Quanity'].sum().loc[(slice(None), 'F')]
sum_by_year_male = names_df.groupby(['Year', 'Sex'])['Quanity'].sum().loc[(slice(None), 'M')]
ratio = sum_by_year_female/sum_by_year_male

min_diff_abs = ratio.min().round(5)
max_diff_abs = ratio.max().round(5)
min_year = ratio.idxmin()
max_year = ratio.idxmax()

print('######################Zad5#######################')
print('Min difference:', min_diff_abs, 'at year:', min_year)
print('Max difference:', max_diff_abs, 'at year:', max_year)
print('#################################################\n')

ax[1].plot(years, ratio, label='Ratio')
ax[1].set_xlabel('Year')
ax[1].set_ylabel('Ratio')
ax[1].set_title('Ratio number of births female/male by year')
ax[1].legend(loc='best')
ax[1].grid(True)

#######################################################
# Zadanie 6 Top1000 najpopularniejsze imiona jako suma wazona uwzględniając czestotliwosc i ilosc urodzen
#######################################################

top_names_df = names_df_pivot.copy().reset_index()

top_names_group_filtered = top_names_df.groupby(['Year', 'Sex']).apply(lambda x: x.nlargest(1000, 'frequency_male' if x['Sex'].iloc[0] == 'M' else 'frequency_female')).reset_index(drop=True)

top_names_group_filtered['weighted_sum_female'] = top_names_group_filtered['frequency_female'] * top_names_group_filtered['Quanity']
top_names_group_filtered['weighted_sum_male'] = top_names_group_filtered['frequency_male'] * top_names_group_filtered['Quanity']

top_names_pivot_female = top_names_group_filtered.pivot_table(index='Name', values='weighted_sum_female', aggfunc='sum')
top_names_pivot_male = top_names_group_filtered.pivot_table(index='Name', values='weighted_sum_male', aggfunc='sum')

name_counts_male = top_names_pivot_male.nlargest(1000, 'weighted_sum_male')
name_counts_female = top_names_pivot_female.nlargest(1000, 'weighted_sum_female')

#######################################################
# Zadanie 7 Wykres zmiany dla imienia John i top1 imienia kobiecego
#######################################################

fig1, ax1 = plt.subplots(figsize=(10, 6))

John_data = names_df_pivot.loc[([1934, 1980, 2022], 'M', 'John'), :]
top_female = name_counts_female.index[0]
top_female_data = names_df_pivot.loc[([1934, 1980, 2022], 'F', top_female), :]
years_3 = John_data.index.get_level_values('Year')

width = 8

bar1 = ax1.bar(years_3-1.5*width, John_data['Quanity'], label='John birth quanity', width=width)
bar2 = ax1.bar(years_3-width/2, top_female_data['Quanity'], label=f'{top_female} birth quanity', width=width)
ax1.set_xlabel('Year')
ax1.set_ylabel('Births')
ax1.set_title('Name comparison')
ax1.grid(True)
ax1.set_xticks(years_3)
ax1.set_xticklabels(years_3)

ax2 = ax1.twinx()
bar3 = ax2.bar(years_3+width/2, John_data['frequency_male'], label='John popularity', color='black', width=width)
bar4 = ax2.bar(years_3+1.5*width, top_female_data['frequency_female'], label=f'{top_female} popularity', color='red', width=width)
ax2.set_ylabel('Popularity')
ax2.grid(True)

bars = [bar1, bar2, bar3, bar4]
labels = [bar.get_label() for bar in bars]
ax1.legend(bars, labels, loc='best')

#######################################################
# Zadanie 8 Wykres roznorodnosci imion
#######################################################

names_top_df_pivot = names_df.pivot_table(index=['Year', 'Sex'], values='Name', aggfunc=','.join)

same_names_percentage = {}

for year in years:
    names_year_female = set(names_top_df_pivot.loc[(year, 'F')].values[0].split(','))
    names_year_male = set(names_top_df_pivot.loc[(year, 'M')].values[0].split(','))

    female = (len(names_year_female.intersection(set(name_counts_female.index)))/len(names_year_female))*100
    male = (len(names_year_male.intersection(set(name_counts_male.index)))/len(names_year_male))*100
    
    same_names_percentage[year] = {
        'female': female,
        'male': male,
        'difference': abs(female - male)
    }

fig3, ax3 = plt.subplots(figsize=(10, 6))

years = list(same_names_percentage.keys())
female_percentages = [same_names_percentage[year]['female'] for year in years]
male_percentages = [same_names_percentage[year]['male'] for year in years]
max_diff, max_diff_year = max((same_names_percentage[year]['difference'], year) for year in years)

print('######################Zad8#######################')
print(f'year of the  maximum difference between number of female and male names: {max_diff_year}')
print('#################################################\n')

ax3.plot(years, female_percentages, label='Female percentage')
ax3.plot(years, male_percentages, label='Male percentage')
ax3.set_xlabel('Year')
ax3.set_ylabel('Percentage [%]')
ax3.set_title('Percentage of names by year in top 1000 names')
ax3.grid(True)
ax3.legend(loc='best')

#######################################################
# Zadanie 9 Analiza hipotezy o ostatniej literze imienia
#######################################################

last_letter_df_pivot = names_df.copy()
last_letter_df_pivot['Last letter'] = last_letter_df_pivot['Name'].str[-1]
last_letter_df_pivot = last_letter_df_pivot.pivot_table(index=['Year', 'Sex', 'Last letter'], values='Quanity', aggfunc='sum')

sum_by_year_sex = last_letter_df_pivot.groupby(['Year', 'Sex']).sum()
last_letter_df_pivot['Quanity normalization'] = last_letter_df_pivot['Quanity'] / sum_by_year_sex['Quanity']

extract = last_letter_df_pivot.loc[([1917, 1967, 2022], 'M'), :]

years_extract = extract.index.get_level_values('Year').unique()

extract = extract.reindex(pd.MultiIndex.from_product([years_extract, ['M'], extract.index.get_level_values('Last letter').unique()], names=['Year', 'Sex', 'Last letter']), fill_value=0)

letters_difference = abs(extract.loc[(2022, 'M'), :]['Quanity normalization'] - extract.loc[(1917, 'M'), :]['Quanity normalization'])

max_difference_letter = letters_difference.nlargest(1)

letter = max_difference_letter.index.get_level_values('Last letter')[0]

print('######################Zad9#######################')
print(f'Biggest drop/rise in last letter popularity between 1917 and 2022:\n letter: {letter}, value: {max_difference_letter.values[0].round(5)}')
print('#################################################\n')

width = 0.2

data = extract.groupby(['Year'])['Quanity normalization']

fig4, ax4 = plt.subplots(figsize=(16, 6))

for i, year in enumerate(years_extract):
    year_data = data.get_group(year)
    letters = year_data.index.get_level_values('Last letter')
    quantities = year_data.values

    x = np.arange(0, len(letters))
    ax4.bar(x + (i - len(letters) / 2) * width - width, quantities, width=width, label=str(year))

ax4.set_xlabel('Last Letter')
ax4.set_ylabel('Popularity')
ax4.set_title(f'Popularity of last letters by year')
ax4.grid(True)

ax4.set_xticks(x - (len(letters) / 2) * width)
ax4.set_xticklabels(letters)

ax4.legend(loc='best')

difference = abs(extract.loc[(1917, 'M'), :]['Quanity normalization'] - extract.loc[(2022, 'M'), :]['Quanity normalization'])

max_difference_letter = difference.nlargest(1)
max_difference_letter

top_3_max_values = difference.nlargest(3)
top_3_letters = list(top_3_max_values.index.get_level_values('Last letter'))

first_letter = last_letter_df_pivot.loc[(slice(None), 'M', top_3_letters[0]), :]['Quanity normalization']
second_letter = last_letter_df_pivot.loc[(slice(None), 'M', top_3_letters[1]), :]['Quanity normalization']
third_letter = last_letter_df_pivot.loc[(slice(None), 'M', top_3_letters[2]), :]['Quanity normalization']

fig5, ax5 = plt.subplots(figsize=(16, 6))


ax5.plot(years, first_letter.values, label=top_3_letters[0])
ax5.plot(years, second_letter.values, label=top_3_letters[1])
ax5.plot(years, third_letter.values, label=top_3_letters[2])

ax5.set_xlabel('Year')
ax5.set_ylabel('Popularity')
ax5.set_title(f'Three most popular of last letters by full range of years')
ax5.grid(True)

ax5.legend(loc='best')
plt.xlim(years[0], years[-1])

#######################################################
# Zadanie 10 Najwieksza zmiana wsrod wspolnych imion, bylo to
# obliczane na podstawie rankingu top1000 dla poszczegolnych lat
# nastepnie wybierane byly imiona ktore znajdowaly sie w rankingu oraz byly wspolne male/female
# dla podanych zakresow lat dane byly agregowane i wybierane imiona o najwiekszej roznicy
#######################################################

before_1930 = top_names_group_filtered[top_names_group_filtered['Year'] <= 1930]
since_2000 = top_names_group_filtered[top_names_group_filtered['Year'] >= 2000]

before_1930_female_names = before_1930[before_1930['Sex'] == 'F']
before_1930_male_names = before_1930[before_1930['Sex'] == 'M']
since_2000_female_names = since_2000[since_2000['Sex'] == 'F']
since_2000_male_names = since_2000[since_2000['Sex'] == 'M']

before_1930_female_names = before_1930_female_names.pivot_table(index=['Year', 'Name'], values='Quanity')
before_1930_male_names = before_1930_male_names.pivot_table(index=['Year', 'Name'], values='Quanity')
since_2000_female_names = since_2000_female_names.pivot_table(index=['Year', 'Name'], values='Quanity')
since_2000_male_names = since_2000_male_names.pivot_table(index=['Year', 'Name'], values='Quanity')

years_1930 = list(before_1930_female_names.index.get_level_values('Year').unique().values)
years_2000 = list(since_2000_female_names.index.get_level_values('Year').unique().values)

fig6, ax6 = aggregated_trend(years_1930, before_1930, before_1930_female_names, before_1930_male_names)
fig7, ax7 = aggregated_trend(years_2000, since_2000, since_2000_female_names, since_2000_male_names)

#######################################################
# Zadanie 11 Wczytywanie danych SQLite
#######################################################

conn = sqlite3.connect("./data/demography_us_2023.sqlite3")
c = conn.cursor()

df_births = pd.read_sql_query("SELECT * FROM births", conn)
df_deaths = pd.read_sql_query("SELECT * FROM deaths", conn)
df_population = pd.read_sql_query("SELECT * FROM population", conn)

conn.close()

#######################################################
# Zadanie 12 Wykrs przyrostu naturalnego
#######################################################

df_deaths_by_year = df_deaths.pivot_table(index=['Year'], values='Total', aggfunc='sum')
df_births_by_year = df_births.pivot_table(index=['Year'], values='Total', aggfunc='sum')

natural_increase = df_births_by_year - df_deaths_by_year

fig8, ax8 = plt.subplots(figsize=(16, 6))


ax8.plot(df_deaths_by_year.index, natural_increase.values, label='Births - Deaths')
ax8.set_xlabel('Year')
ax8.set_ylabel('Value of natural increase')
ax8.set_title(f'Natural increase by year')
ax8.grid(True)

ax8.legend(loc='best')

#######################################################
# Zadanie 13 Wykres wspolczynnika przezywalnosci dzieci w pierwszym roku zycia
#######################################################

df_first_year_deaths = df_deaths[df_deaths['Age'] == '0']['Total']
df_first_year_births = df_births['Total']

ratio = ((df_first_year_births - df_first_year_deaths.values) / df_first_year_births.values) * 100

fig9, ax9 = plt.subplots(figsize=(16, 6))

ax9.plot(df_deaths_by_year.index, ratio, label='Ratio')
ax9.set_xlabel('Year')
ax9.set_ylabel('Ratio')
ax9.set_title(f'Survival rate of children in the first year of life')
ax9.grid(True)
ax9.legend(loc='best')

#######################################################
# Zadanie 14 Porownanie bazy danych imion oraz bazy danych SQLite
#######################################################

df_births_by_year.columns = ['Quanity']
range_year = df_births_by_year.index.get_level_values('Year')
names_df_by_year = names_df.pivot_table(index=['Year'], values='Quanity', aggfunc='sum')
names_df_by_year = names_df_by_year.loc[range_year[0]:range_year[-1]]


relative_error = ((abs(names_df_by_year.values - df_births_by_year.values)) / abs(df_births_by_year.values)) * 100
absolute_error = abs(names_df_by_year - df_births_by_year)

relative_error

fig10, ax10 = plt.subplots(figsize=(16, 6))

ax10.plot(range_year, relative_error, label='Error')
ax10.set_xlabel('Year')
ax10.set_ylabel('Error [%]')
ax10.set_title(f'Relative error between births in names dataset and births in demography dataset')
ax10.grid(True)
ax10.legend(loc='best')


max_absolute_error = absolute_error.nlargest(1, 'Quanity')
min_absolute_error = absolute_error.nsmallest(1, 'Quanity')

print('######################Zad14#######################')
print(f"Max relative error:\n Year: {max_absolute_error.index.values[0]}, Value: {max_absolute_error.values[0][0]}")
print(f"Min relative error:\n Year: {min_absolute_error.index.values[0]}, Value: {min_absolute_error.values[0][0]}")
print('#################################################\n')


figures = [fig, fig1, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10]
titles = ['Zad 5', 'Zad 7', 'Zad 8', 'Zad 9', 'Zad 9', 'Zad 10', 'Zad 10', 'Zad 12', 'Zad 13', 'Zad 14']

for i in range(len(figures)):
    figures[i].canvas.manager.set_window_title(titles[i])

plt.tight_layout()
plt.show()
