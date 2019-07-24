import numpy as np
from matplotlib import pyplot as plt
from sys import argv
from util import load_data, Bunch


def plot_measure_time(b):
    print(b.run_hists.shape)
    for loss, var, med, nm in zip(b.avg_losses, b.avg_loss_vars, b.med_losses, b.estimator_names):
        plt.errorbar(b.tlist, loss, yerr=np.sqrt(var), capsize=2, label=nm)
        plt.plot(b.tlist, med, marker='o', linestyle='None', label=nm)
    #plt.ylim(bottom=0.0)
    plt.plot(b.tlist, b.var_omega * (np.sqrt(1 + 4/(b.tlist**2*b.var_omega)) - 1) / 2) # Bayesian Cramer Rao Bound
    plt.plot(b.tlist, b.tlist*0 + ((b.omega_max - b.omega_min) / b.NUM_PARTICLES)**2 / 12, label='grid bound')
    plt.plot(b.tlist, b.tlist*0 + 3*b.var_omega, label='an estimated bound')
    plt.yscale('log')

def plot_measure_number(b):
    for loss, var, nm in zip(b.avg_losses, b.avg_loss_vars, b.estimator_names):
        plt.errorbar(b.nlist, loss, yerr=np.sqrt(var), capsize=2, label=nm)
    #plt.ylim(bottom=0.0)
    plt.yscale('log')
    plt.xscale('log')

def plot_shot_number(b):
    for loss, var, nm in zip(b.avg_losses, b.avg_loss_vars, b.estimator_names):
        plt.errorbar(b.nshots_list, loss, yerr=np.sqrt(var), capsize=2, label=nm)
    #plt.ylim(bottom=0.0)
    plt.yscale('log')
    plt.xscale('log')

def plot_t_theta_loss(b):
    for loss, var, nm in zip(b.avg_losses, b.avg_loss_vars, b.t_estimator_names):
        plt.errorbar(b.theta_list, loss, yerr=np.sqrt(var), capsize=2, label=nm)
    #plt.ylim(bottom=0.0)
    plt.yscale('log')

def plot_measurement_performance(b):
    for loss, var, nm in zip(b.avg_losses, b.avg_loss_vars, b.estimator_names):
        plt.errorbar(b.N_list, loss, yerr=np.sqrt(var), capsize=2, label=nm)
    #plt.ylim(bottom=0.0)
    plt.plot(b.N_list, b.N_list*0 + ((b.omega_max - b.omega_min) / b.NUM_PARTICLES)**2 / 12, label='grid bound')
    plt.plot(b.N_list, b.N_list*0 + 3*b.var_omega, label='an estimated bound')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('number of measurements')
    plt.ylabel('loss (MSE)')

plotfns = {
    'measure_time': plot_measure_time,
    'measure_number': plot_measure_number,
    'shot_number': plot_shot_number,
    't_theta_loss': plot_t_theta_loss,
    'measurement_performance': plot_measurement_performance
}


def plot(data):
    plotfns[data['plottype']](Bunch(data))


def main():
    for filename in argv[1:]:
        plot(load_data(filename))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()


