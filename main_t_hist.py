import numpy as np
from estimators import *
import matplotlib.pyplot as plt


def main():
    omegas = np.linspace(omega_min, omega_max, 60)
    #omega_prior = normalize(1. + 0.*omegas) # uniform prior
    omega_prior = normalize(np.exp(-1e-7 * (omegas-mu_omega)**2)) # normal prior

    def get_v1(x, r):
        return 0.
    def get_omega_list(x, r, v1, t_u_list=None):
        random_seed(x, r)
        if t_u_list is None:
            t_u_list = np.arange(100)
        ans = sample_omega_list(omegas, omega_prior, v1, t_u_list)
        random_reseed()
        return ans
    def get_est(x, r, v1):
        return Estimator(GridDist1D(omegas, omega_prior, 0.), OptimizingChooser(20, 20))
    
    sim = Simulator(get_v1, get_omega_list, get_est)
    t_hists = sim.get_t_hist(-1, 1000)
    dist, bins = np.histogram(t_hists[:, :20].flatten(), bins=100)
    plt.bar(bins[:-1] / np.pi, normalize(dist), align='edge', width=(bins[1] - bins[0]))
    plt.xlabel('$t \hat \Omega / \pi $'); plt.ylabel('relative frequency')
    plt.show()
    
    dist, bins = np.histogram(t_hists[:, 80:].flatten(), bins=100)
    plt.bar(bins[:-1] / np.pi, normalize(dist), align='edge', width=(bins[1] - bins[0]))
    plt.xlabel('$t \hat \Omega / \pi $'); plt.ylabel('relative frequency')
    plt.show()


if __name__ == '__main__':
    main()
