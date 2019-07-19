import numpy as np
from matplotlib import pyplot as plt
from sys import argv
from util import load_data, Bunch


hyperparams = ['omega_min', 'omega_max', 'v_0', 'var_omega', 'NUM_PARTICLES', 'dynamic_params']


# a line on a plot
class Trace:
    plottype2xlistnm = {'measure_time': 'tlist', 'measurement_performance': 'N_list'}
    def __init__(self, b, loss, var, nm):
        self.b = b
        self.loss = loss
        self.var = var
        self.nm = nm
    def __str__(self):
        return self.nm
    def plot(self):
        plt.errorbar(
            vars(self.b)[self.plottype2xlistnm[self.b.plottype]],
            self.loss, yerr=np.sqrt(self.var),
            capsize=2, label=self.nm
        )


# each bunch may contribute multiple traces
def get_traces(b):
    return [ Trace(b, loss, var, nm)
        for loss, var, nm
        in zip(b.avg_losses, b.avg_loss_vars, b.estimator_names) ]


# mutates trace names to contain relevant hyperparam info
def expand_names(traces):
    for param in hyperparams:
        differs = False # does this param differ for any traces we are looking at?
        for i in range(1, len(traces)):
            if vars(traces[i].b)[param] != vars(traces[i-1].b)[param]:
                differs = True
                break
        if differs:
            for t in traces:
                t.nm += ', %s=%s' % (param, str(vars(t.b)[param]))


def plot_measure_time():
    plt.yscale('log')
    plt.xlabel('measurement time')
    plt.ylabel('loss (mean squared error)')

def plot_measurement_performance():
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('number of measurements')
    plt.ylabel('loss (mean squared error)')


plotfns = {
    'measure_time': plot_measure_time,
    'measurement_performance': plot_measurement_performance
}


def main():
    traces = []
    for filename in argv[1:]:
        traces.extend(get_traces(Bunch(load_data(filename))))
    plottype = traces[0].b.plottype
    expand_names(traces)
    for t in traces:
        t.plot()
    plotfns[plottype]()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

