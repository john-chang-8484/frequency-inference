import numpy as np
from matplotlib import pyplot as plt
from sys import argv
from util import load_data, Bunch, diff, fn_from_source


# constants
EST_BND_GAMMA = 0.78 # the gamma constant for the estimated bound
# (run compute_est_bnd_gamma.py to compute this value)

P_BND_GAMMA = 0.998


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
        means = np.mean(self.loss_omegas, axis=1)
        u_rel = np.std(self.loss_omegas, axis=1) / (np.sqrt(self.loss_omegas.shape[1]) * means)
        y = np.sqrt(means) / self.mu_omega
        u_y = 0.5 * u_rel * y
        plt.errorbar(self.x_list, y, yerr=u_y, capsize=2,
            label=('omega_loss' + self.nm), color=self.colour1)
        y_med = np.sqrt(np.median(self.loss_omegas, axis=1)) / self.mu_omega
        plt.plot(self.x_list, y_med,
            linestyle='--', color=self.colour1)
    def plot_v1_loss(self):
        means = np.mean(self.loss_v1s, axis=1)
        u_rel = np.std(self.loss_v1s, axis=1) / (np.sqrt(self.loss_v1s.shape[1]) * means)
        y = np.sqrt(means)
        u_y = 0.5 * u_rel * y
        plt.errorbar(self.x_list, y, yerr=u_y, capsize=2,
            label=('v1_loss' + self.nm), color=self.colour2)
        plt.plot(self.x_list, np.sqrt(np.median(self.loss_v1s, axis=1)),
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
                dlines = diff( fnrep(vars(traces[i])[param]),
                               fnrep(vars(traces[i+1])[param]), [] )
                dlines = [line.strip() for line in dlines] # remove tabs
                traces[i].nm += ', \n%s: %s' % (param, '; '.join(dlines))


def v1_true(traces):
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('v1_true')
def n_measurements(traces):
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('n_measurements')
def t_ms(traces):
    plt.yscale('log')
    #plt.plot(np.linspace(0., 0.0002, 100), 1e5*np.sin(70000*np.linspace(0., 0.0002, 100))**2)
    plt.xlabel('time of measurement')
def v1_nom(traces):
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('nominal v1')
    xmin, xmax, ymin, ymax = plt.axis()
    plt.plot([traces[0].v1_true] * 2, [ymin, ymax], label='v1_true')


plotfns = {
    'est_var_omega_v1_true': v1_true,
    'est_var_omega_n_measurements': n_measurements,
    'x_trace_n_ms': n_measurements,
    'x_trace_v1_true': v1_true,
    'x_trace_t_ms': t_ms,
    'x_trace_fit_shots': n_measurements,
    'x_trace_nominal_v1': v1_nom,
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
    
    if 'l' in options:
        expand_names(traces)
    if 'p' in options:
        expand_names(traces)
        for i, t in enumerate(traces):
            print('Trace %d:' % i, t.nm)
            t.nm = ', trace %d' % i
    
    # plot
    for t in traces:
        if 'o' in options:
            t.plot_omega_loss()
        if 'v' in options:
            t.plot_v1_loss()
    
    # label
    if 'o' in options and 'v' in options:
        plt.ylabel('omega_loss = $\\\\sqrt{langle(\\hat\\Omega - \\Omega)^2\\rangle} / \\mu_\\Omega$, v1_loss = $\\sqrt{\\langle(\\log\\hat v_1 - \\log v_1)^2\\rangle}$')
    elif 'o' in options:
        plt.ylabel('omega loss $\\frac{\\sqrt{\\langle(\\hat\\Omega - \\Omega)^2\\rangle}}{\\mu_\\Omega}$')
    else:
        plt.ylabel('v1 loss $\\sqrt{\\langle(\\log\\hat v_1 - \\log v_1)^2\\rangle}$')
    
    # customize based on type of plot
    try:
        plotfns[plottype](traces)
    except KeyError:
        pass
    
    # plot theoretical bounds
    if 'v' in options:
        if 'gb' in options:
            for i, t in enumerate(traces):
                if t.dist_name in ['grid', 'grid_dist']:
                    plt.plot(t.x_list, np.array(t.x_list)*0 + 
                        np.sqrt((np.log(t.v1s[1]) - np.log(t.v1s[0]))**2 / 12),
                        label='grid bound, trace %d' % i)
    if 'o' in options:
        if 'gb' in options: # grid bound
            for i, t in enumerate(traces):
                if t.dist_name in ['grid', 'grid_dist']:
                    plt.plot(t.x_list,
                        np.sqrt( np.array(t.x_list)*0 +
                            ((t.omega_max - t.omega_min) / t.omegas.size)**2
                                 / 12) / t.mu_omega, 
                        label='grid bound, trace %d' % i)
        if 'eb' in options: # estimated bound
            if plottype == 'x_trace_n_ms':
                delta0 = np.sum(traces[0].omega_prior * traces[0].omegas**2) - np.sum(traces[0].omega_prior * traces[0].omegas)**2
                try:
                    bnd = (delta0 * np.power(EST_BND_GAMMA, traces[0].x_list) + 
                        (fn_from_source(traces[0].get_v1)(0, 0) * EST_BND_GAMMA / (1 - EST_BND_GAMMA)))
                except NameError: # backwards compatibility for some data files:
                    bnd = (delta0 * np.power(EST_BND_GAMMA, traces[0].x_list) + 
                        (traces[0].v1s[0] * EST_BND_GAMMA / (1 - EST_BND_GAMMA)))
                plt.plot(traces[0].x_list, bnd, label='estimated bound')
            elif plottype == 'x_trace_v1_true':
                bnd = traces[0].x_list * EST_BND_GAMMA / (1 - EST_BND_GAMMA)
                plt.plot(traces[0].x_list, bnd, label='estimated bound')
        if 'pb' in options: # pessimistic bound
            if plottype == 'x_trace_n_ms':
                delta0 = np.sum(traces[0].omega_prior * traces[0].omegas**2) - np.sum(traces[0].omega_prior * traces[0].omegas)**2
                try:
                    bnd = (delta0 * np.power(P_BND_GAMMA, traces[0].x_list) + 
                        (fn_from_source(traces[0].get_v1)(0, 0) * P_BND_GAMMA / (1 - P_BND_GAMMA)))
                except NameError: # backwards compatibility for some data files:
                    bnd = (delta0 * np.power(P_BND_GAMMA, traces[0].x_list) + 
                        (traces[0].v1s[0] * P_BND_GAMMA / (1 - P_BND_GAMMA)))
                plt.plot(traces[0].x_list, bnd, label='pessimistic bound')
        if 'crb' in options: # bayesian Cramer-Rao bound
            init_cov = np.cov(traces[0].omegas, aweights=traces[0].omega_prior)
            try:
                v1_true = fn_from_source(traces[0].get_v1)(0, 0)
            except NameError: # backwards compatibility for some data files:
                v1_true = traces[0].v1s[0]
            if plottype == 'x_trace_t_ms':
                tlist = traces[0].x_list
                min_cov = [init_cov * np.ones_like(tlist)]
                length = int(input('How many measurements were taken for these traces? > '))
                for i in range(1, length):
                    min_cov.append(1 / (tlist**2 + 1 / (v1_true + min_cov[-1])))
                if 'ncrf' not in options: # option to remove floor
                    plt.plot(tlist, np.sqrt(v1_true * (np.sqrt(1 + 4/(tlist**2*v1_true)) - 1) / 2) / traces[0].mu_omega, label='Cramer Rao bound, infinite measurement floor')
                plt.plot(tlist, np.sqrt(min_cov[-1]) / traces[0].mu_omega, label='Cramer Rao Bound')
            if plottype == 'x_trace_n_ms':
                nlist = traces[0].x_list
                length = max(nlist)
                min_cov = [init_cov, init_cov]
                try:
                    t_max = traces[0].t_max
                except AttributeError: # backwards compatibility for some data files
                    t_max = float(input('What was t_max? > '))
                for i in range(1, length):
                    min_cov.append(1 / (t_max**2 + 1 / (v1_true + min_cov[-1])))
                plt.plot(nlist, np.sqrt(np.array(min_cov)[nlist]) / traces[0].mu_omega, label='Cramer Rao bound')
                

    if 'l' in options or 'p' in options:
        plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

