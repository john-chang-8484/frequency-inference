import numpy as np
from scipy.fftpack import dct, idct
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sim import *



def perturb_omega(omega, v1):
    return clip_omega(omega + np.random.normal(0., np.sqrt(v1)))


def sample_omega_list(omegas, prior, v1, length):
    omega0 = sample_dist(omegas, prior)
    omega_list = [omega0]
    for i in range(1, length):
        omega_list.append(perturb_omega(omega_list[-1], v1))
    return omega_list


# RULE: all fn calls should preserve normalization 
class ParticleDist:
    def normalize(self):
        self.dist = normalize(self.dist)
    def mean_omega(self):
        return np.sum(self.dist * self.omegas)
    def mean_v1(self):
        return np.sum(self.dist * self.v1s)
    def many_update(self, ts, ms):
        for t, m in zip(ts, ms):
            self.wait_u()
            self.update(t, m)


class GridDist(ParticleDist):
    def __init__(self, omegas, v1s, prior):
        assert omegas.shape + v1s.shape == prior.shape
        self.shape = prior.shape
        self.omegas = np.copy(omegas).reshape((omegas.size, 1))
        self.v1s = np.copy(v1s).reshape((1, v1s.size))
        self.dist = np.copy(prior)
    def wait_u(self):
        ''' given a posterior distribution for omega at time t,
            we find the dist for omega at time t+u '''
        diff = self.omegas[-1] - self.omegas[0]
        fact = ((self.v1s * np.pi**2) / (2. * diff**2))
        cos_coeffs = dct(self.dist, axis=0) # switch to fourier space, in terms of cosines to get Neumann BC
        n = np.outer(np.arange(self.shape[0]), np.ones(self.shape[1]))
        cos_coeffs *= np.exp( - fact * n**2 ) # heat eq update
        self.dist = idct(cos_coeffs, axis=0) / (2 * self.shape[0]) # switch back to the usual representation
    def update(self, t, m):
        self.dist *= get_likelihood(self.omegas, t, m)
        self.normalize()



def get_measurements(v1s, v1_prior, omegas, omega_prior, get_ts, n_ms):
    ts = get_ts(n_ms)
    
    v1_true = sample_dist(v1s, v1_prior)
    omega_list_true = sample_omega_list(omegas, omega_prior, v1_true, n_ms)
    ms = many_measure(omega_list_true, ts)
    
    return v1_true, omega_list_true, ts, ms


def main():
    log_v1s = np.linspace(-16., -2., 103)
    v1s = np.exp(log_v1s)
    v1_prior = normalize(1. + 0.*v1s)
    omegas = np.linspace(omega_min, omega_max, 400)
    omega_prior = normalize(1. + 0.*omegas)
    
    def get_ts(n_ms):
        return np.random.uniform(0., 4.*np.pi, n_ms)
    
    v1_true, omega_list_true, ts, ms = get_measurements(v1s, v1_prior, omegas,
        omega_prior, get_ts, 300)
    
    ####
    
    grid = GridDist(omegas, v1s, np.outer(omega_prior, v1_prior))
    grid.many_update(ts, ms)
    
    ####
    
    print(grid.mean_omega(), omega_list_true[-1])
    print(grid.mean_v1(), v1_true)
    
    ####
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(log_v1s, omegas)
    ax.plot_surface(X, Y, grid.dist, cmap=plt.get_cmap('copper'))
    plt.show()


if __name__ == '__main__':
    main()

