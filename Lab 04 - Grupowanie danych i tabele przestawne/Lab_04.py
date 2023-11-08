import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_temperature = pd.read_csv('city_temperature.csv', low_memory=False)
df_temperature = df_temperature.drop(df_temperature.loc[df_temperature['Year'] < 1995].index)

def Ex1():
    # df_statistics_region1 = df_temperature[['Region', 'AvgTemperature']]
    # print(df_statistics_region1)

    # df_statistics_region2 = df_temperature.groupby('Region').agg({'AvgTemperature': ['mean', 'median', 'sum', 'min', 'max']})
    # print(df_statistics_region2)

    # df_statistics_month = df_temperature.groupby('Month').agg({'AvgTemperature': ['mean', 'median', 'sum', 'min', 'max']})
    # print(df_statistics_month)
    df = df_temperature.drop(columns=['Day']).pivot_table(columns='Region', index=['Year', 'Month'],
                                                          aggfunc=['min', 'max', 'mean'], values='AvgTemperature')
    print(df)
    # print(df.index)
    print(df.loc[(slice(None), slice(None)), (['min', 'max'], 'Africa')])

def Ex2():

    ### MONTH 6
    df_month_6 = df_temperature[df_temperature['Month'] == 6]
    df_temperature_subplot_month_6 = df_month_6.groupby(['Region', 'Year']).agg({'AvgTemperature': ['mean']})

    df_temperature_dict_month_6 = df_temperature_subplot_month_6.groupby(level=0).apply(lambda x: x.droplevel(0).to_dict()).to_dict()
    df_keys_month_6 = list(df_temperature_dict_month_6.keys())

    ### MONTH 12
    df_month_12 = df_temperature[df_temperature['Month'] == 12]
    df_temperature_subplot_month_12  = df_month_12.groupby(['Region', 'Year']).agg({'AvgTemperature': ['mean']})

    df_temperature_dict_month_12  = df_temperature_subplot_month_12.groupby(level=0).apply(lambda x: x.droplevel(0).to_dict()).to_dict()
    df_keys_month_12  = list(df_temperature_dict_month_12.keys())
    ### END

    print(df_temperature_dict_month_12)

    fig, ax = plt.subplots(2, 1, figsize=(20, 8))
    fig.suptitle('Temperature through the years')

    for i in range(len(df_keys_month_6)):
        month6_xaxis = list(df_temperature_dict_month_6[df_keys_month_6[i]]['AvgTemperature', 'mean'].keys())
        month6_yaxis = list(df_temperature_dict_month_6[df_keys_month_6[i]]['AvgTemperature', 'mean'].values())

        month12_xaxis = list(df_temperature_dict_month_12[df_keys_month_12[i]]['AvgTemperature', 'mean'].keys())
        month12_yaxis = list(df_temperature_dict_month_12[df_keys_month_12[i]]['AvgTemperature', 'mean'].values())

        ax[0].plot(month6_xaxis, month6_yaxis)
        ax[1].plot(month12_xaxis, month12_yaxis)

    ax[0].legend(df_keys_month_6, loc='lower right', fontsize=14)
    ax[0].grid()
    ax[0].tick_params(axis='both', which='major', labelsize=24)
    ax[0].set_xticks(month6_xaxis)
    ax[0].set_xlim(month6_xaxis[0], month6_xaxis[-1])
    ax[0].set_xlabel('Year')
    ax[0].set_ylabel('Temperature')
    ax[0].set_title('June')

    ax[1].legend(df_keys_month_12, loc='lower right', fontsize=14)
    ax[1].grid()
    ax[1].tick_params(axis='both', which='major', labelsize=24)
    ax[1].set_xticks(month6_xaxis)
    ax[1].set_xlim(month12_xaxis[0], month12_xaxis[-1])
    ax[1].set_xlabel('Year')
    ax[1].set_ylabel('Temperature')
    ax[1].set_title('December')

    plt.show()

def Ex2_pivot_table():
    df = df_temperature[(df_temperature['Month'] == 6) | (df_temperature['Month'] == 12)]
    df = df.pivot_table(columns='Region',
                        index=['Year','Month'],
                        aggfunc=['mean'],
                        values='AvgTemperature')
    print(df)

    temp_6month = df.loc[(slice(None), 6), :]
    temp_12month = df.loc[(slice(None), 12), :]

    # Przygotuj wykresy dla każdego regionu dla miesiąca 6
    fig, ax = plt.subplots(2, 1, figsize=(20, 8))
    fig.suptitle('Temperature through the years')

    for region in df.columns.get_level_values('Region'):
        region_data = temp_6month[('mean', region)]
        years = region_data.index.get_level_values('Year')
        values = region_data.values
        ax[0].plot(years, values, label=region)

        region_data = temp_12month[('mean', region)]
        years = region_data.index.get_level_values('Year')
        values = region_data.values
        ax[1].plot(years, values, label=region)

    # Ustawienia wykresu dla miesiąca 6
    ax[0].set_xlabel('Year')
    ax[0].set_ylabel('Temperature')
    ax[0].set_title('June')
    ax[0].legend(loc='best')
    ax[0].grid(True)

    # Ustawienia wykresu dla miesiąca 12
    ax[1].set_xlabel('Year')
    ax[1].set_ylabel('Temperature')
    ax[1].set_title('December')
    ax[1].legend(loc='best')
    ax[1].grid(True)

    plt.show()


def Ex3():
    # Import CSV files to pandas dataframes
    raw_ecg_df = pd.read_csv('raw_ecg.csv')
    ecg_beats_df = pd.read_csv('ecg_beats.csv')

    # Create plot
    fig, ax = plt.subplots(2, 1, figsize=(20, 8))
    ax[0].set_title('ECG Signal with Detected Beats')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Amplitude (mV)')

    sample_rate = 500  # Example sampling rate (500 Hz)

    raw_timestamps = np.array(raw_ecg_df.index)
    raw_amplitudes = np.array(raw_ecg_df['LeadI'])
    # Plot ECG signal from raw_ecg dataframe
    ax[0].plot(raw_timestamps, raw_amplitudes)

    # Plot points on raw_ecg plot in places that are indicated in dataframe ecg_beats
    beat_timestamps = np.array(ecg_beats_df['BeatTimestamp'].values)
    temp = np.array(ecg_beats_df['BeatTimestamp'].values)
    print(temp)
    beat_amplitudes = raw_amplitudes[temp]
    ax[0].scatter(beat_timestamps, beat_amplitudes, color='red')

    # Print the first beat amplitude and all beat timestamps
    print(f"First beat amplitude: {beat_amplitudes} mV")
    print("All beat timestamps:")
    print(beat_timestamps)
    

    window_start = -0.55  # Start 550 ms before the event
    window_end = 0.4  # End 400 ms after the event

    # Create a time vector representing the time points for the analysis
    time_vector = np.linspace(window_start, window_end, int((window_end - window_start) * sample_rate) + 1)

    # Use NumPy vectorized operations to extract segments around each event
    range = (beat_timestamps[:, np.newaxis] + sample_rate * time_vector).astype(int).clip(0, 4999)
    segments = raw_amplitudes[range]

    # Compute the average signal across all segments using vectorized operations
    average_signal = np.mean(segments, axis=0)
    ax[1].plot(average_signal)
    plt.show()

def Ex4():
    df_titanic = pd.read_csv('titanic_train.csv')
    df_titanic_pivot_table = df_titanic.pivot_table(columns='Sex',
                                                    index=['Pclass'],
                                                    aggfunc=['sum', lambda x: (sum(x)/len(x))*100, lambda x: len(x)],
                                                    values='Survived')
    df_titanic_pivot_table.columns = ['Female_Survived', 'Male_Survived', 'Female_Survival_Rate', 'Male_Survival_Rate', 'Sum_Female_Passengers', 'Sum_Male_Passengers']
    print(df_titanic_pivot_table)

    first_class = df_titanic_pivot_table.loc[1, :][2:-2]
    second_class = df_titanic_pivot_table.loc[2, :][2:-2]
    third_class = df_titanic_pivot_table.loc[3, :][2:-2]

    fig, ax = plt.subplots(figsize=(20, 8))

    width = 0.2

    x = np.arange(len(first_class))
    labels = df_titanic_pivot_table.columns[2:-2]

    ax.bar(x-width, first_class, width, label='First class')
    ax.bar(x, second_class, width, label='Second class')
    ax.bar(x+width, third_class, width, label='Third class')

    ax.legend(fontsize = 24)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.grid(axis = 'y')
    ax.set_ylim(0, 100)
    plt.show()

if __name__ == '__main__':
    # Ex1()
    # Ex2()
    # Ex2_pivot_table()
    Ex3()
    # Ex4()
