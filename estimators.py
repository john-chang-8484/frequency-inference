import numpy as np
import random
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
from util import save_data, get_filepath, deterministic_sample, get_numeric_class_vars, gini
from plot_util import pin_plot
import inspect
import math
import qinfer
from qinfer import SimplePrecessionModel, Distribution, LiuWestResampler


# constants:
omega_min = 0.1     # [1/s]
omega_max = 1.9     # [1/s]
v_0       = 0.      # [1/s^2]   # the noise in omega (essentially a decoherence rate)
t_max     = 4. * np.pi # [s]    # the maximum time at which we can make a measurement


################################################################################
##                                                                            ##

def prob_excited(t, omega):
    """ probability of excitation at time t for a given value of omega """
    return 0.5 * (1. - (np.exp(- 0.5 * v_0 * t) * np.cos(omega * t)))

def normalize(dist):
    """ normalize a discrete probability distribution """
    return dist / np.sum(dist)

def clip_omega(omegas):
    return np.clip(omegas, omega_min, omega_max)

def perturb_omega(omega, v1):
    return clip_omega(omega + np.random.normal(0., np.sqrt(v1)))

def sample_dist(values, dist, size=None):
    """ Randomly samples from a distribution, interpolating between points.
        Values should be in sorted order.
        dist is a probability distribution on values. """
    i = 1 + np.random.choice(np.arange(0, len(values)), p=dist, size=size)
    extvals = np.concatenate((values[0:1], values, values[-1:]))
    return np.random.uniform(
        (extvals[i-1] + extvals[i])/2,
        (extvals[i] + extvals[i+1])/2 )

def sample_omega_list(omegas, prior, v1, length):
    """ sample omega from the prior, then simulate random drift over time """
    omega0 = sample_dist(omegas, prior)
    omega_list = [omega0]
    for i in range(1, length):
        omega_list.append(perturb_omega(omega_list[-1], v1))
    return omega_list

def measure(omega, t):
    """ Measurement at time t. Returns 0 for ground state, 1 for excited. """
    return np.random.binomial(1, prob_excited(t, omega))

def likelihood(omega, t, m):
    """ Returns likelihood, where m is the result of a measurement. """
    pe = prob_excited(t, omega)
    return (pe * m) + (1 - pe) * (1 - m)

def random_seed(x, run, randomizer=0):
    seed = (randomizer * 10000000) + (x * 10000) + run
    random.seed(seed)
    np.random.seed(seed)

##                                                                            ##
################################################################################
##                                                                            ##

class ParticleDist1D:
    """ Represents a probability distribution using weighted particles.
        RULE: all method calls should leave the distribution normalized. """
    def normalize(self):
        self.dist = normalize(self.dist)
    def mean_omega(self):
        return np.sum(self.dist * self.omegas)
    def mean_log_v1(self):
        return np.log(self.v1)
    def sample_omega(self, n):
        """ Take n samples of omega from this distribution. """
        return np.random.choice(self.omegas, p=np.abs(self.dist), size=n)

class GridDist1D(ParticleDist1D):
    """ particles are in a regularly spaced grid """
    name = 'grid'
    def __init__(self, omegas, prior, v1):
        assert omegas.size == prior.size
        self.size = omegas.size
        self.omegas = np.copy(omegas)
        self.dist = np.copy(prior)
        self.v1 = v1
        
    def wait_u(self):
        """ given a posterior distribution for omega at time T,
            we find the dist for omega at time T+u """
        diff = omega_max - omega_min
        fact = (self.v1 * np.pi**2) / (2. * diff**2) # update factor
        cos_coeffs = dct(self.dist) # switch to fourier space. (in terms of cosines to get Neumann BC)
        n = np.arange(cos_coeffs.size)
        cos_coeffs *= np.exp( - fact * n**2 ) # heat eq update
        self.dist = idct(cos_coeffs) / (2 * cos_coeffs.size) # switch back to the usual representation
    def update(self, t, m):
        self.dist *= likelihood(self.omegas, t, m)
        self.normalize()

class DynamicDist1D(ParticleDist1D):
    """ particles are not regularly spaced and can move around over time """
    n_ess_limit = 0.5
    a = 0.1 # a fudge factor to prevent degeneracy
    b = 1.2 # additional fudge factor for resampling
    name = 'dynamic'
    def __init__(self, omegas, prior, v1, size):
        self.v1, self.size = v1, size
        self.omegas = omegas[deterministic_sample(size, prior)]
        self.dist = np.ones(size) / size
        self.target_cov = self.cov() # initialize target covariance to actual covariance
    def wait_u(self):
        self.omegas = perturb_omega(self.omegas, self.v1)
        self.target_cov += self.v1
    def update(self, t, m):
        old_cov = self.cov()
        self.dist *= likelihood(self.omegas, t, m)
        self.normalize()
        new_cov = self.cov()
        self.target_cov *= max(self.a, new_cov / old_cov)
        """ assume target covariance changes by same amount,
            but max with fudge factor to prevent degeneracy """
        if self.n_ess() < self.n_ess_limit * self.size:
            self.resample() # resample only if necessary
    def cov(self):
        return np.cov(self.omegas, ddof=0, aweights=self.dist)
    def n_ess(self):
        """ returns the effective sample size """
        return 1. / np.sum(self.dist**2)
    def resample(self):
        self.omegas = self.omegas[deterministic_sample(self.size, self.dist)]
        self.dist = np.ones(self.size) / self.size
        cov_new = self.cov()
        if cov_new > self.target_cov:
            # we ended up with more variance than expected, that's cool
            self.target_cov = cov_new
        else:
            # fudge factor b multiplies the amount of variance we add
            add_var = self.b * (self.target_cov - cov_new)
            epsilon = ( # double tailed exponential distribution
                np.random.exponential(scale=np.sqrt(add_var/2), size=self.size)-
                np.random.exponential(scale=np.sqrt(add_var/2), size=self.size))
            self.omegas = clip_omega(self.omegas + epsilon)

##                                                                            ##
################################################################################
##                                                                            ##

class TimeChooser:
    """ class that chooses the time at which we should make a measurement,
        given a distribution """
    pass

class RandomChooser(TimeChooser):
    name = 'random'
    def __init__(self): pass
    def get_t(self, dist):
        return np.random.uniform(0., t_max)

class TwoPointChooser(TimeChooser):
    name = 'two_point'
    def __init__(self, search_depth):
        self.search_depth = search_depth
    def tau_n(self, n):
        return np.pi * (2 * n + 1) / abs(self.omega1 - self.omega2)
    def tau_m(self, m):
        return np.pi * (2 * m + 1) / (self.omega1 + self.omega2)
    def get_t(self, dist):
        """ choose a good t for the next experiment. """
        if np.random.binomial(1, 0.2): # some chance of just picking t randomly
            return np.random.uniform(0., t_max)
        self.omega1, self.omega2 = np.sort(dist.sample_omega(16))[np.array([0, -1])]
        if self.omega1 == self.omega2: # just choose a large time
            return np.random.uniform(0.7 * t_max, t_max)
        if self.tau_m(0) > t_max:
            return t_max
        if self.tau_n(0) > t_max:
            return self.tau_m(np.floor(
                (t_max * (self.omega1 + self.omega2) / np.pi - 1) / 2 ))
        n, m = 0, 0
        nb, mb = 0, 0 # best n, best m
        min_diff = abs(self.tau_n(n) - self.tau_m(m))
        for i in range(self.search_depth):
            if self.tau_n(n) < self.tau_m(m):
                n += 1
            else:
                m += 1
            if max(self.tau_n(n), self.tau_m(m)) > t_max:
                break
            diff = abs(self.tau_n(n) - self.tau_m(m))
            if diff < min_diff:
                min_diff = diff
                nb, mb = n, m
        if self.tau_n(nb) > t_max:
            return self.tau_m(mb)
        else:
            return (self.tau_n(nb) + self.tau_m(mb)) / 2

##                                                                            ##
################################################################################
##                                                                            ##

class Estimator:
    """ An estimator is a combination of a distribution and a time chooser """
    def __init__(self, dist, chooser):
        self.dist = dist
        self.chooser = chooser
    def mean_omega(self):
        return self.dist.mean_omega()
    def mean_log_v1(self):
        return self.dist.mean_log_v1()
    def many_measure(self, omega_list):
        length = len(omega_list)
        t_hist = []
        for i in range(length):
            t = self.chooser.get_t(self.dist)
            t_hist.append(t)
            m = measure(omega_list[i], t)
            self.dist.wait_u()
            self.dist.update(t, m)
        return t_hist

class Simulator:
    def __init__(self, get_v1, get_omega_list, get_estimator):
        self.get_v1 = get_v1
        self.get_omega_list = get_omega_list
        self.get_estimator = get_estimator
    def do_runs(self, x, n_runs):
        loss_omega_list, loss_v1_list = np.zeros(n_runs), np.zeros(n_runs)
        for r in range(n_runs):
            v1 = self.get_v1(x, r) # [1/s^2/u] (u is the time between measurements)
            omega_list = self.get_omega_list(x, r, v1)
            estimator = self.get_estimator(x, r, v1)
            estimator.many_measure(omega_list)
            loss_omega_list[r] = (omega_list[-1] - estimator.mean_omega())**2
            loss_v1_list[r] = (np.log(v1) - estimator.mean_log_v1())**2
        return loss_omega_list, loss_v1_list
    def x_trace(self, n_runs, x_list, x_list_nm):
        loss_omegas = np.zeros((len(x_list), n_runs))
        loss_v1s    = np.zeros((len(x_list), n_runs))
        for i, x in enumerate(x_list):
            print(i, '\t', x)
            loss_omegas[i], loss_v1s[i] = self.do_runs(x, n_runs)
        dummy_est = self.get_estimator(0, 0, 0.)
        return {
            'omega_min': omega_min,
            'omega_max': omega_max,
            'v_0': v_0,
            'x_list_nm': x_list_nm,
            'x_list': x_list,
            'dist_name': dummy_est.dist.name,
            'chooser_name': dummy_est.chooser.name,
            'get_v1': inspect.getsource(self.get_v1),
            'get_omega_list': inspect.getsource(self.get_omega_list),
            'loss_omegas': loss_omegas,
            'loss_v1s': loss_v1s,
            'plottype': 'x_trace_%s' % x_list_nm,
            'dist_params': get_numeric_class_vars(type(dummy_est.dist)),
            'chooser_params': get_numeric_class_vars(type(dummy_est.chooser)),
        }

##                                                                            ##
################################################################################


def main():
    omegas = np.linspace(omega_min, omega_max, 600)
    prior = normalize(1. + 0.*omegas)
    def get_v1(x, r):
        return 0.00001
    def get_omega_list(x, r, v1):
        random_seed(x, r)
        return sample_omega_list(omegas, prior, v1, x)
    def get_estimator(x, r, v1):
        return Estimator(GridDist1D(omegas, prior, v1), TwoPointChooser(32))
    sim = Simulator(get_v1, get_omega_list, get_estimator)
    data = sim.x_trace(500, [1, 2, 3, 6, 10], 'n_ms')
    data['omegas'], data['prior'] = omegas, prior
    save_data(data, get_filepath(data['plottype']))


if __name__ == '__main__':
    main()


