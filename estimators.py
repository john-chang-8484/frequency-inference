import numpy as np
import random
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
from util import save_data, get_filepath, deterministic_sample, get_numeric_class_vars, gini
from plot_util import pin_plot
import inspect
import math
import qinfer
from qinfer import SimplePrecessionModel, Distribution, LiuWestResampler, FiniteOutcomeModel


# constants:
omega_min = 130000. # [1/s]
omega_max = 150000. # [1/s]
v_0       = 0.      # [1/s^2]   # the noise in omega (essentially a decoherence rate)
t_max     = 0.0002  # [s]       # the maximum time at which we can make a measurement
mu_omega  = 140000. # [1/s]     # the mean of the initial distribution on omega

q_g1      = 0.043   # P(m=1 | g)
q_g0      = 1-q_g1  # P(m=0 | g)
q_e0      = 0.009   # P(m=0 | e)
q_e1      = 1-q_e0  # P(m=1 | e)


################################################################################
##                            Basic Functions                                 ##

def prob_excited(t, omega):
    """ probability of excitation at time t for a given value of omega
        -> assumes zero detuning
    """
    return 0.5 * (1. - np.cos(omega * t) * np.exp(- 0.5 * v_0 * t))

def likelihood(omega, t, m):
    """ Returns likelihood, where m is the result of a measurement. """
    pe = prob_excited(t, omega)
    return pe*(m*q_e1 + (1 - m)*q_e0) + (1 - pe)*(m*q_g1 + (1 - m)*q_g0)

def normalize(dist):
    """ normalize a discrete probability distribution """
    return dist / np.sum(dist)

def clip_omega(omegas):
    return np.clip(omegas, omega_min, omega_max)

def perturb_omega(omega, v1):
    if hasattr(omega, 'size') and omega.shape != ():
        return clip_omega(omega + np.sqrt(v1) * np.random.randn(omega.size))
    else:
        return clip_omega(omega + np.sqrt(v1) * np.random.randn())

def sample_dist(values, dist, size=None):
    """ Randomly samples from a distribution, interpolating between points.
        Values should be in sorted order.
        dist is a probability distribution on values. """
    i = 1 + np.random.choice(np.arange(0, len(values)), p=dist, size=size)
    extvals = np.concatenate((values[0:1], values, values[-1:]))
    return np.random.uniform(
        (extvals[i-1] + extvals[i])/2,
        (extvals[i] + extvals[i+1])/2 )

def sample_omega_list(omegas, prior, v1, t_u_list):
    """ sample omega from the prior, then simulate random drift over time """
    omega0 = sample_dist(omegas, prior)
    omega_list = [omega0]
    for i in range(1, len(t_u_list)):
        omega_list.append(perturb_omega(
            omega_list[-1], v1 * (t_u_list[i] - t_u_list[i-1]) ))
    return omega_list

def measure(omega, t):
    """ Measurement at time t. Returns 0 for ground state, 1 for excited. """
    return np.random.binomial(1, likelihood(omega, t, 1))

def random_seed(x, run, randomizer=0):
    seed = hash((randomizer * 100000000000) + (x * 1000000) + run)
    random.seed(seed)
    np.random.seed(seed % 2**32)
def random_reseed():
    random.seed()
    np.random.seed()

def heat_evolve(omegas, dist, v_tot):
    """ evolve a probability distribution over a set of evenly spaced
        particles according to the heat equation. v_tot is the amount 
        of variance to add. """
    delta_omega = omegas[1] - omegas[0]
    if v_tot == 0:
        return dist
    elif v_tot / (delta_omega**2) < 0.4: # stability condition with wiggle room
        dist_new = np.copy(dist)
        # heat eq evolution
        dist_new[1:-1] += (0.5 * v_tot / (delta_omega**2)) * (
            dist[2:] + dist[:-2] - 2.*dist[1:-1] )
        # boundary conditions
        dist_new[0] += (0.5 * v_tot / (delta_omega**2)) * (dist[1] - dist[0])
        dist_new[-1] += (0.5 * v_tot / (delta_omega**2)) * (dist[-2] - dist[-1])
        return dist_new
    else: # if v_tot is too big, we use eigenfunction decomposition instead
        diff = omega_max - omega_min
        fact = (v_tot * np.pi**2) / (2. * diff**2) # update factor
        cos_coeffs = dct(dist) # switch to fourier space. (in terms of cosines to get Neumann BC)
        n = np.arange(cos_coeffs.size)
        cos_coeffs *= np.exp( - fact * n**2 ) # heat eq update
        return np.abs(idct(cos_coeffs) / (2 * cos_coeffs.size)) # switch back to the usual representation

##                                                                            ##
################################################################################
##                           1D Distributions                                 ##

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
        
    def wait_u(self, n_u=1.):
        """ given a posterior distribution for omega at time T,
            we find the dist for omega at time T+u
            or, if the optional n_u argument is not 1, at time T + n_u*u """
        v_tot = self.v1 * n_u
        self.dist = heat_evolve(self.omegas, self.dist, v_tot)
        self.normalize()
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
    def wait_u(self, n_u=1.):
        self.omegas = perturb_omega(self.omegas, self.v1 * n_u)
        self.target_cov += self.v1 * n_u
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
            epsilon = np.random.normal(0., np.sqrt(add_var), size=self.size)
            self.omegas = clip_omega(self.omegas + epsilon)

#   helper classes for QinferDist1D:
# http://docs.qinfer.org/en/latest/guide/timedep.html#specifying-custom-time-step-updates
class DiffusivePrecessionModel(SimplePrecessionModel):
    def __init__(self, v1, **kwargs):
        self.v1 = v1
        super().__init__(**kwargs)
    def update_timestep(self, modelparams, expparams, n_u=0.):
        assert expparams.shape[0] == 1
        steps = np.random.normal(0., np.sqrt(self.v1 * n_u), 
            size=(modelparams.shape[0], 1, 1))
        return clip_omega(modelparams[:, :, np.newaxis] + steps)
    def likelihood(self, outcomes, modelparams, expparams):
        super(DiffusivePrecessionModel, self).likelihood(outcomes, modelparams, expparams)
        return likelihood(modelparams[:,0], expparams, outcomes).reshape(1, modelparams.shape[0], 1)
class PriorSample(Distribution):
    n_rvs = 1
    def __init__(self, values, dist):
        self.values = values
        self.dist = dist
    def sample(self, n=1):
        return np.reshape(sample_dist(self.values, self.dist, size=n), (n, 1))
# simple wrapper class for the qinfer implementation
class QinferDist1D(ParticleDist1D):
    a = 1.
    h = 0.005
    name = 'qinfer'
    def __init__(self, omegas, prior, v1, size):
        self.v1, self.size = v1, size
        self.qinfer_model = DiffusivePrecessionModel(self.v1,
            min_freq=omega_min)
        self.qinfer_prior = PriorSample(omegas, prior)
        self.qinfer_updater = qinfer.SMCUpdater( self.qinfer_model, self.size,
            self.qinfer_prior,
            resampler=LiuWestResampler(self.a, self.h, debug=False) )
    def update(self, t, m):
        self.qinfer_updater.update(np.array([m]), np.array([t]))
    def mean_omega(self):
        return self.qinfer_updater.est_mean()[0]
    def mean_v1(self):
        return self.v1
    def posterior_marginal(self, *args, **kwargs):
        return self.qinfer_updater.posterior_marginal(*args, **kwargs)
    def wait_u(self, n_u=1.):
        self.qinfer_updater.particle_locations = self.qinfer_model.update_timestep(
            self.qinfer_updater.particle_locations, np.zeros((1,)), n_u )[:,:,0]
    def sample_omega(self, n):
        """ Take n samples of omega from this distribution. """
        return self.qinfer_updater.particle_locations[np.random.choice(
            self.qinfer_updater.particle_locations.shape[0],
            p=self.qinfer_updater.particle_weights, size=n ), 0]

##                                                                            ##
################################################################################
##                           2D Distributions                                 ##

class ParticleDist2D:
    """ Represents a 2d probability dist using weighted particles
        RULE: all fn calls should leave the dist normalized """
    def normalize(self):
        self.dist = normalize(self.dist)
    def mean_omega(self):
        return np.sum(self.dist * self.omegas)
    def mean_log_v1(self):
        return np.sum(self.dist * np.log(self.v1s))
    def sample_omega(self, n):
        """ Take n samples of omega from this distribution. """
        return np.random.choice(self.omegas.flatten(), 
            p=normalize(np.abs(np.sum(self.dist, axis=1))), size=n)

class GridDist2D(ParticleDist2D):
    """ evenly spaced grid """
    name = 'grid'
    def __init__(self, omegas, v1s, prior):
        assert omegas.shape + v1s.shape == prior.shape
        self.size = prior.size
        self.shape = prior.shape
        self.omegas = np.copy(omegas).reshape((omegas.size, 1))
        self.v1s = np.copy(v1s).reshape((1, v1s.size))
        self.dist = np.copy(prior)
    def wait_u(self, n_u=1.):
        """ given a posterior distribution for omega at time t,
            we find the dist for omega at time t+u """
        v_tots = self.v1s.flatten() * n_u
        for i, v_tot in enumerate(v_tots):
            self.dist[:,i] = heat_evolve(self.omegas.flatten(), self.dist[:,i], v_tot)
    def update(self, t, m):
        self.dist *= likelihood(self.omegas, t, m)
        self.normalize()

def covmax2d(cov1, cov2):
    """ a helper function for DynamicDist2D, this function
        takes the "maximum" of two 2d covariance matrices """
    a1, b1, c1 = cov1[0,0], cov1[1,0], cov1[1, 1]
    a2, b2, c2 = cov2[0,0], cov2[1,0], cov2[1, 1]
    a = max(a1, a2)
    c = max(c1, c2)
    b = ((b1 / np.sqrt(a1*c1)) + (b2 / np.sqrt(a2*c2))) * np.sqrt(0.5 * a * c)
    return np.array([[a, b], [b, c]])
def semidefify2d(cov):
    """ make a covariance matrix positive semidefinite """
    a, b, c = cov[0, 0], cov[1, 0], cov[1, 1]
    a, c = max(a, 0), max(c, 0)
    b = np.clip(b, -np.sqrt(a*c), np.sqrt(a*c))
    return np.array([[a, b], [b, c]])
def cov_ratios2d(cov1, cov2):
    a1, b1, c1 = cov1[0,0], cov1[1,0], cov1[1, 1]
    a2, b2, c2 = cov2[0,0], cov2[1,0], cov2[1, 1]
    return np.array([a1/a2, b1/b2])
class DynamicDist2D(ParticleDist2D, DynamicDist1D):
    """ A 2d particle distribution that dynamically adapts the locations of
        the particles in order to do inference with fewer particles than the
        grid method, but to a finer resolution. """
    name = 'dynamic'
    a = np.array([[0.1, 0.], [0., 0.9]])
    b = 1.
    def __init__(self, omegas, v1s, prior, size):
        self.size = size
        self.log_v1_min, self.log_v1_max = np.log(v1s[0]), np.log(v1s[-1])
        log_v1s, omegas = np.meshgrid(np.log(v1s), omegas)
        indices = np.random.choice(np.arange(prior.size), p=prior.flatten(), size=self.size)
        selected_omegas = omegas.flatten()[indices]
        selected_log_v1s = log_v1s.flatten()[indices]
        self.vals = np.stack([selected_omegas, selected_log_v1s], axis=0)
        self.dist = normalize(np.ones(self.size))
        self.target_cov = self.cov()
    def cov(self):
        return np.cov(self.vals, ddof=0, aweights=self.dist)
    def wait_u(self, n_u=1.):
        self.vals[0] = perturb_omega(self.vals[0], np.exp(self.vals[1]) * n_u)
        self.target_cov += np.array([[self.mean_v1() * n_u, 0.], [0., 0.]])
    def update(self, t, m):
        old_cov = self.cov()
        self.dist *= likelihood(self.vals[0], t, m)
        self.normalize()
        new_cov = self.cov()
        ratio = new_cov / old_cov
        ratio[1 - np.isfinite(ratio)] = np.array([[1., 0.], [0., 1.]])[1 - np.isfinite(ratio)]
        self.target_cov = semidefify2d(self.target_cov * ratio)
        if self.n_ess() < self.n_ess_limit * self.size:
            self.resample() # resample only if necessary
    def resample(self):
        self.vals = self.vals[:, deterministic_sample(self.size, self.dist)]
        self.dist = np.ones(self.size) / self.size
        cov_new = self.cov()
        self.target_cov = covmax2d(self.target_cov, cov_new)
        # fudge factor b multiplies the amount of variance we add
        add_var = self.b * semidefify2d((self.target_cov - cov_new))
        epsilon = np.random.multivariate_normal(np.zeros(2), add_var, size=self.size).T
        # hack to preserve the weird scaling properties here
        epsilon[0] *= np.exp(self.vals[1]) / np.sum(np.exp(self.vals[1]))
        self.vals += epsilon
        self.vals[0] = clip_omega(self.vals[0])
        self.vals[1] = np.clip(self.vals[1], self.log_v1_min, self.log_v1_max)
    def mean_omega(self):
        return np.sum(self.vals[0] * self.dist)
    def mean_v1(self):
        return np.sum(np.exp(self.vals[1]) * self.dist)
    def mean_log_v1(self):
        return np.sum(self.vals[1] * self.dist)
    def sample_omega(self, n):
        return np.random.choice(self.vals[0], p=self.dist, size=n)


#   helper classes for QinferDist2D:
# http://docs.qinfer.org/en/latest/guide/timedep.html#specifying-custom-time-step-updates
# http://docs.qinfer.org/en/latest/guide/models.html
class DiffusivePrecessionModel2D(FiniteOutcomeModel):
    @property
    def n_modelparams(self):
        return 2
    @property
    def is_n_outcomes_constant(self):
        return True
    def n_outcomes(self, expparams):
        return 2
    def are_models_valid(self, modelparams):
        return np.logical_and(
            modelparams[:,0] >= omega_min, 
            modelparams[:,0] <= omega_max,
            modelparams[:,1] >= 0. )
    @property
    def expparams_dtype(self):
        return [('ts', 'float', 1)]
    def likelihood(self, outcomes, modelparams, expparams):
        super(DiffusivePrecessionModel2D, self).likelihood(outcomes, modelparams, expparams)
        return likelihood(modelparams[:,0], expparams, outcomes).reshape(1, modelparams.shape[0], 1)
    def update_timestep(self, modelparams, expparams, n_u=0.):
        assert expparams.shape[0] == 1
        modelparams_new = np.copy(modelparams)
        modelparams_new[:,1] = np.clip(modelparams[:,1], 0., np.inf)
        steps = np.random.normal(0., np.sqrt(modelparams_new[:,1] * n_u), 
            size=modelparams.shape[0])
        modelparams_new[:,0] = clip_omega(modelparams[:,0] + steps)
        return modelparams_new.reshape(modelparams.shape + (1,))
class PriorSample2D(Distribution):
    n_rvs = 2
    def __init__(self, omegas, v1s, dist):
        self.omegas, self.v1s = np.meshgrid(omegas, v1s)
        self.values = np.stack([self.omegas, self.v1s], axis=-1)
        self.dist = dist
    def sample(self, n=1):
        """ fancy 2D distribution sampling
            Essentially, this function calculates the midpoints between the
            grid points, then, if a grid point is selected, it samples
            uniformly between the midpoints on either side of that gridpoint. """
        sz = self.values.size // 2
        cat0 = np.concatenate((self.values[0:1,:], self.values, self.values[-1:,:]), axis=0)
        cat1 = np.concatenate((cat0[:,0:1], cat0, cat0[:,-1:]), axis=1)
        choice = np.random.choice(sz, p=self.dist.flatten(), size=n)
        ind0, ind1 = np.unravel_index(choice, self.dist.shape)
        upper = (cat1[ind1 + 1, ind0 + 1] + cat1[ind1 + 2, ind0 + 2]) / 2
        lower = (cat1[ind1 + 1, ind0 + 1] + cat1[ind1, ind0]) / 2
        return np.random.uniform(lower, upper)
# simple wrapper class for the qinfer implementation
class QinferDist2D(ParticleDist2D):
    name = 'qinfer'
    a = 1.
    h = 0.005
    size = ...
    def __init__(self, omegas, v1s, prior, size):
        self.size = size
        self.qinfer_model = DiffusivePrecessionModel2D()
        self.qinfer_prior = PriorSample2D(omegas, v1s, prior)
        self.qinfer_updater = qinfer.SMCUpdater( self.qinfer_model, self.size,
            self.qinfer_prior, resampler=LiuWestResampler(self.a, self.h, debug=False) )
    def update(self, t, m):
        self.qinfer_updater.update(np.array([m]), np.array([t]))
    def wait_u(self, n_u=1.):
        self.qinfer_updater.particle_locations = self.qinfer_model.update_timestep(
            self.qinfer_updater.particle_locations, np.zeros((1,)), n_u )[:,:,0]
    def mean_omega(self):
        return self.qinfer_updater.est_mean()[0]
    def mean_log_v1(self):
        log_v1s = np.log(self.qinfer_updater.particle_locations[:,1])
        return np.sum( log_v1s[np.isfinite(log_v1s)] *
            self.qinfer_updater.particle_weights[np.isfinite(log_v1s)] )
    def posterior_marginal(self, *args, **kwargs):
        return self.qinfer_updater.posterior_marginal(*args, **kwargs)
    def sample_omega(self, n):
        """ Take n samples of omega from this distribution. """
        return self.qinfer_updater.particle_locations[np.random.choice(
            self.qinfer_updater.particle_locations.shape[0],
            p=self.qinfer_updater.particle_weights, size=n ), 0]

##                                                                            ##
################################################################################
##                              Choosers                                      ##

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

class OptimizingChooser(TimeChooser):
    name = 'optimizing'
    def __init__(self, n_omegas, n_ts):
        self.n_omegas, self.n_ts = n_omegas, n_ts
    def get_potential_ts(self):
        return np.random.uniform(0., t_max, self.n_ts).reshape(1, self.n_ts)
    def get_t(self, dist):
        omegas = dist.sample_omega(self.n_omegas).reshape(self.n_omegas, 1)
        ts = self.get_potential_ts()
        pe = likelihood(omegas, ts, 1)
        pg = 1 - pe
        mean_e = omegas * pe / np.sum(pe, axis = 0)
        mean_g = omegas * pg / np.sum(pg, axis = 0)
        se = ( pe * (omegas - mean_e)**2 +
               pg * (omegas - mean_g)**2 )
        mse = np.sum(se, axis=0) / self.n_omegas
        t = ts[0, np.argmin(mse)]
        return t

##                                                                            ##
################################################################################
##                   Estimator and Simulator Classes                          ##

class Estimator:
    """ An estimator is a combination of a distribution and a time chooser """
    def __init__(self, dist, chooser):
        self.dist = dist
        self.chooser = chooser
    def mean_omega(self):
        return self.dist.mean_omega()
    def mean_log_v1(self):
        return self.dist.mean_log_v1()
    def many_measure(self, omega_list, t_u_list=None):
        """ make many measurements and update the distribution accordingly,
            where omega_list contains the true value of omega at the time
            of each measurement.
            Optional argument t_u_list shows the time at which each measurement
            was taken, this time being measured in u's. If omitted, they are
            assumed to be spaced evenly, with a separation of 1u. """
        length = len(omega_list)
        t_hist, t_omega_hat_hist = [], []
        for i in range(length):
            t = self.chooser.get_t(self.dist)
            t_hist.append(t)
            t_omega_hat_hist.append(t * self.mean_omega())
            m = measure(omega_list[i], t)
            if i > 0:
                if t_u_list is None:
                    self.dist.wait_u()
                else:
                    self.dist.wait_u(t_u_list[i] - t_u_list[i-1])
            self.dist.update(t, m)
        return t_hist, t_omega_hat_hist

class Simulator:
    def __init__(self, get_v1, get_omega_list, get_estimator, get_t_u_list=None):
        """ all arguments are functions that take the numbers (x, r, v1) as arguments
            ( except that get_v1 only takes the numbers (x, r), and
              get_omega_list takes (x, r, v1, t_u_list=None). ) """
        self.get_v1 = get_v1
        self.get_omega_list = get_omega_list
        self.get_estimator = get_estimator
        if get_t_u_list is None:
            self.get_t_u_list = (lambda x, r, v1: None)
        else:
            self.get_t_u_list = get_t_u_list
    def do_runs(self, x, n_runs):
        loss_omega_list, loss_v1_list = np.zeros(n_runs), np.zeros(n_runs)
        for r in range(n_runs):
            v1 = self.get_v1(x, r) # [1/s^2/u] (u is the time between measurements)
            estimator = self.get_estimator(x, r, v1)
            t_u_list = self.get_t_u_list(x, r, v1)
            omega_list = self.get_omega_list(x, r, v1, t_u_list)
            estimator.many_measure(omega_list, t_u_list)
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
            't_max': t_max,
            'mu_omega': mu_omega,
            'q_g1': q_g1, 'q_g0': q_g0, 'q_e0': q_e0, 'q_e1': q_e1,
            'x_list_nm': x_list_nm,
            'x_list': x_list,
            'dist_name': dummy_est.dist.name,
            'chooser_name': dummy_est.chooser.name,
            'get_v1': inspect.getsource(self.get_v1),
            'get_omega_list': inspect.getsource(self.get_omega_list),
            'get_estimator': inspect.getsource(self.get_estimator),
            'get_t_u_list': inspect.getsource(self.get_t_u_list),
            'loss_omegas': loss_omegas,
            'loss_v1s': loss_v1s,
            'plottype': 'x_trace_%s' % x_list_nm,
            'dist_params': get_numeric_class_vars(type(dummy_est.dist)),
            'chooser_params': get_numeric_class_vars(type(dummy_est.chooser)),
        }
    def get_t_hist(self, x, n_runs):
        t_hists = []
        t_omega_hat_hists = []
        for r in range(n_runs):
            v1 = self.get_v1(x, r) # [1/s^2/u]
            estimator = self.get_estimator(x, r, v1)
            t_u_list = self.get_t_u_list(x, r, v1)
            omega_list = self.get_omega_list(x, r, v1, t_u_list)
            t_hist, t_omega_hat_hist = estimator.many_measure(omega_list, t_u_list)
            t_hists.append(t_hist)
            t_omega_hat_hists.append(t_omega_hat_hist)
        return np.array(t_hists), np.array(t_omega_hat_hists)

##                                                                            ##
################################################################################



