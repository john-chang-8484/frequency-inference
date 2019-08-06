import numpy as np
from matplotlib import pyplot as plt
from sys import argv
from util import load_data, Bunch, diff


hyperparams = ['omega_min', 'omega_max', 'v_0', 'dist_name', 'chooser_name', 'dist_params', 'chooser_params']
differfns = ['get_v1', 'get_omega_list', 'get_estimator']


# a line on a plot
class Trace:
    def __init__(self, b):
        vars(self).update(vars(b))
        self.nm = ''
    def __str__(self):
        return self.nm
    def plot_omega_loss(self):
        y = np.mean(self.loss_omegas, axis=1)
        u_y = np.std(self.loss_omegas, axis=1) / np.sqrt(self.loss_omegas.shape[1])
        plt.errorbar(self.x_list, y, yerr=u_y, capsize=2,
            label=('omega_loss' + self.nm), color=self.colour1)
        plt.plot(self.x_list, np.median(self.loss_omegas, axis=1),
            linestyle='--', color=self.colour1)
    def plot_v1_loss(self):
        y = np.mean(self.loss_v1s, axis=1)
        u_y = np.std(self.loss_v1s, axis=1) / np.sqrt(self.loss_v1s.shape[1])
        plt.errorbar(self.x_list, y, yerr=u_y, capsize=2,
            label=('v1_loss' + self.nm), color=self.colour2)
        plt.plot(self.x_list, np.median(self.loss_v1s, axis=1),
            linestyle='--', color=self.colour2)


def fnrep(fn):
    """ return a representation of the source code of a function,
        ignoring its signature """
    return fn.split('\n')[1:]

def fn_eq(frep1, frep2):
    """ check for equality between 2 function representations """
    for line1, line2 in zip(frep1, frep2):
        if line1 != line2:
            return False
    return True
    

# mutates trace names to contain relevant hyperparam info
def expand_names(traces):
    for param in hyperparams:
        differs = False # does this param differ for any traces we are looking at?
        for i in range(1, len(traces)):
            if vars(traces[i])[param] != vars(traces[i-1])[param]:
                differs = True
                break
        if differs:
            for t in traces:
                t.nm += ', \n%s=%s' % (param, str(vars(t)[param]))
    for param in differfns:
        differs = False # does this param differ for any traces we are looking at?
        for i in range(1, len(traces)):
            if not fn_eq(
            fnrep(vars(traces[i])[param]),
            fnrep(vars(traces[i-1])[param]) ):
                differs = True
                break
        if differs:
            for i in range(-1, len(traces)-1):
                print(param, diff(fnrep(vars(traces[i])[param]), fnrep(vars(traces[i+1])[param]), []))
                dlines = diff( fnrep(vars(traces[i])[param]),
                               fnrep(vars(traces[i+1])[param]), [] )
                dlines = [line.strip() for line in dlines] # remove tabs
                traces[i].nm += ', \n%s: %s' % (param, '; '.join(dlines))


def v1_true():
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('v1_true')

def n_measurements():
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('n_measurements')


plotfns = {
    'est_var_omega_v1_true': v1_true,
    'est_var_omega_n_measurements': n_measurements,
    'x_trace_n_ms': n_measurements,
}


# note: this version of the program does not have the property that all
#   estimators see the same data
def main():
    # get filenames and options
    options = set()
    filenames = []
    for arg in argv[1:]:
        if len(arg) >= 2 and arg[0] == '-':
            options.add(arg[1:])
        else:
            filenames.append(arg)
    if not ('o' in options or 'v' in options):
        options.add('o') # default is the distplay omega_loss
    
    # make traces
    traces = [Trace(Bunch(load_data(filename))) for filename in filenames]
    colourmap = plt.get_cmap('jet')
    for i, t in enumerate(traces):
        t.colour1 = colourmap(i / len(traces))
        t.colour2 = colourmap((2*i + 1) / (2*len(traces)))
    plottype = traces[0].plottype
    expand_names(traces)
    
    # plot
    for t in traces:
        if 'o' in options:
            t.plot_omega_loss()
        if 'v' in options:
            t.plot_v1_loss()
    
    # label
    if 'o' in options and 'v' in options:
        plt.ylabel('omega_loss = $\\langle(\\hat\\Omega - \\Omega)^2\\rangle$, v1_loss = $\\langle(\\log\\hat v_1 - \\log v_1)^2\\rangle$')
    elif 'o' in options:
        plt.ylabel('omega loss $\\langle(\\hat\\Omega - \\Omega)^2\\rangle$')
    else:
        plt.ylabel('v1 loss $\\langle(\\log\\hat v_1 - \\log v_1)^2\\rangle$')
    
    # customize based on type of plot
    plotfns[plottype]()
    
    # plot theoretical bounds
    # TODO: clean up the bounds plotting
    b = traces[0]
    plt.plot(b.x_list, np.array(b.x_list)*0 + ((b.omega_max - b.omega_min) / b.omegas.size)**2 / 12, label='grid bound')

    if 'l' in options:
        plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

