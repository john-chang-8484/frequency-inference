import numpy as np
from matplotlib import pyplot as plt
from sys import argv
from util import load_data, Bunch

def plot_measure_time(b):
    plt.errorbar(b.tlist, b.mle, yerr=np.sqrt(b.mle_var), color=(0., 1., 0.), capsize=2)
    plt.errorbar(b.tlist, b.mpe, yerr=np.sqrt(b.mpe_var), color=(0., 0., 1.), capsize=2)
    plt.errorbar(b.tlist, b.mmse, yerr=np.sqrt(b.mmse_var), color=(0.5, 0., 1.), capsize=2)
    plt.ylim(bottom=0.0)
    plt.show()

plotfns = {
    'measure_time': plot_measure_time
}

def plot(data):
    plotfns[data['plottype']](Bunch(data))

def main():
    for filename in argv[1:]:
        plot(load_data(filename))

if __name__ == '__main__':
    main()

