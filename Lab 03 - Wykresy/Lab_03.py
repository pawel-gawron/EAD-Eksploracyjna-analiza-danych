import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def Ex2():
    x = np.arange(-5.0, 5.0, 0.2)
    std_dev = [2, 1, 3, 4]
    mean = [-2, 0, 3, 4]
    types = ['ob', 'or', '--g', 'xk']

    fig, ax = plt.subplots()
    fig.suptitle('Rozklad')

    for i in range(len(std_dev)):
        f = (1/(std_dev[i]*np.sqrt(np.pi)))*np.exp((-(x-mean[i])**2)/(2*std_dev[i]))
        ax.plot(x, f, types[i])

    legend_labels = [rf'$\sigma = {std_dev[i]}, \mu = {mean[i]}$' for i in range(len(std_dev))]
    ax.legend(legend_labels, loc='upper left')
    ax.grid()
    ax.set_xlim(-5, 5)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(-5, 6, 1))
    ax.set_xticks(np.arange(-5, 5, 0.5), minor=True)
    ax.tick_params(labelrotation=45)
    plt.show()

from matplotlib.ticker import FuncFormatter

def to_percent(y, position):
    return str(int(y)) + '%'

def Ex3():
    fig, ax = plt.subplots()

    cancer = pd.read_json('cancer.json')

    age = [cancer['age_groups'][obj]['age'] for obj in range(len(cancer))]
    male_survivors = [cancer['age_groups'][obj]['male_survivors'] for obj in range(len(cancer))]
    female_survivors = [cancer['age_groups'][obj]['female_survivors'] for obj in range(len(cancer))]

    width = 0.3
    x = np.arange(len(male_survivors))
    ax.bar(x-width/2, male_survivors, width, label='Value 1')
    ax.bar(x+width/2, female_survivors, width, label='Value 2')

    ax.legend(['Man', 'Woman'], loc='upper left')

    ax.set_ylim(0, 40)
    ax.set_xticks(x)
    ax.set_yticks(np.arange(0, 41, 10))
    ax.set_xticklabels(age)

    ax.tick_params(axis='x', labelrotation=90)

    formatter = FuncFormatter(to_percent)
    ax.yaxis.set_major_formatter(formatter)

    ax.grid(axis='y') # add horizontal grid

    plt.show()

if __name__ == '__main__':
    # Ex2()
    Ex3()