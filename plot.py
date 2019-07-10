import numpy as np
from matplotlib import pyplot as plt
from sys import argv
from util import load_data, Bunch

def plot_measure_time(b):
    for loss, var, nm in zip(b.avg_losses, b.avg_loss_vars, b.estimator_names):
        plt.errorbar(b.tlist, loss, yerr=np.sqrt(var), capsize=2, label=nm)
    plt.ylim(bottom=0.0)
    plt.legend()
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

