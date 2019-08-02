import numpy as np
from random import randint, random
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
var_omega = 0.0000  # [1/s^2/u] # the variance in omega per u, where u is the time between measurements

NUM_PARTICLES = 600



# probability of excitation at time t for a given value of omega
def prob_excited(t, omega):
    return 0.5 * (1. - (np.exp(- 0.5 * v_0 * t) * np.cos(omega * t)))


# normalize a discrete probability distribution
def normalize(dist):
    return dist / np.sum(dist)


# clip a set of omega values to the range
def clip_omega(omegas):
    return np.clip(omegas, omega_min, omega_max)


def perturb_omega(omega):
    omeg = np.array(omega)
    return clip_omega(omeg + np.random.normal(0., np.sqrt(var_omega), size=omeg.shape))


# randomly sample from a distribution
# values should be in sorted order
# dist is a probability distribution on values
def sample_dist(values, dist, size=None):
    i = 1 + np.random.choice(np.arange(0, len(values)), p=dist, size=size)
    extvals = np.concatenate((values[0:1], values, values[-1:]))
    return np.random.uniform((extvals[i-1] + extvals[i])/2, (extvals[i] + extvals[i+1])/2)


# returns the number of excited states measured
def measure(omega, t):
    return np.random.binomial(1, prob_excited(t, omega))

# make many measurements given a list of omegas, ts
# assume spacing between measurements is large
def many_measure(omega_list, ts):
    return np.array([measure(omega, t) for omega, t in zip(omega_list, ts)])


def get_likelihood(omega, t, m):
    pe = prob_excited(t, omega)
    return (pe * m) + (1 - pe) * (1 - m)


# gives the log-likelihood of a particular omega, given some measurement m
def log_likelihood(omega, t, m):
    return np.log(get_likelihood(omega, t, n))



# RULE: all fn calls should preserve normalization 
class ParticleDist:
    size = NUM_PARTICLES
    search_depth = 32
    max_t = 32. * np.pi
    def normalize(self):
        self.dist = normalize(self.dist)
    def mean(self):
        return np.sum(self.dist * self.omegas)
    def many_update(self, ts, ms):
        for t, m in zip(ts, ms):
            self.wait_u()
            self.update(t, m)
    def sample(self, n):
        ''' Take n samples from this distribution. '''
        return np.random.choice(self.omegas, p=np.abs(self.dist), size=n)
    def tau_n(self, n):
        return np.pi * (2 * n + 1) / abs(self.omega1 - self.omega2)
    def tau_m(self, m):
        return np.pi * (2 * m + 1) / (self.omega1 + self.omega2)
    def pick_t(self):
        ''' choose a good t for the next experiment '''
        if np.random.binomial(1, 0.2): # some chance of just picking t randomly
            return np.random.uniform(0., self.max_t)
        self.omega1, self.omega2 = np.sort(self.sample(16))[np.array([0, -1])]
        stddev_omega = np.sqrt(max(np.sum(self.omegas**2 * self.dist) - np.sum(self.omegas * self.dist)**2, 1e-6))
        while self.omega1 == self.omega2:
            self.omega2 = clip_omega(self.omega1 + np.random.normal(0., 6. * stddev_omega))
        if self.tau_m(0) > self.max_t:
            return self.max_t
        if self.tau_n(0) > self.max_t:
            return self.tau_m(np.floor(
                (self.max_t * (self.omega1 + self.omega2) / np.pi - 1) / 2 ))
        n, m = 0, 0
        best_pair = 0, 0
        best_dist = abs(self.tau_n(n) - self.tau_m(m))
        for i in range(self.search_depth):
            if self.tau_n(n) < self.tau_m(m):
                n += 1
            else:
                m += 1
            if max(self.tau_n(n), self.tau_m(m)) > self.max_t:
                break
            curr_dist = abs(self.tau_n(n) - self.tau_m(m))
            if curr_dist < best_dist:
                best_dist = curr_dist
                best_pair = n, m
        if self.tau_n(best_pair[0]) > self.max_t:
            return self.tau_m(best_pair[1])
        else:
            return (self.tau_n(best_pair[0]) + self.tau_m(best_pair[1])) / 2


class GridDist(ParticleDist):
    def __init__(self, omegas, prior):
        assert len(omegas) == len(prior) == self.size
        self.omegas = np.copy(omegas)
        self.dist = np.copy(prior)
    def wait_u(self):
        ''' given a posterior distribution for omega at time t,
            we find the dist for omega at time t+u '''
        diff = self.omegas[-1] - self.omegas[0]
        fact = (var_omega * np.pi**2) / (2. * diff**2)
        cos_coeffs = dct(self.dist) # switch to fourier space, in terms of cosines to get Neumann BC
        n = np.arange(cos_coeffs.size)
        cos_coeffs *= np.exp( - fact * n**2 ) # heat eq update
        self.dist = idct(cos_coeffs) / (2 * cos_coeffs.size) # switch back to the usual representation
    def update(self, t, m):
        self.dist *= get_likelihood(self.omegas, t, m)
        self.normalize()


class DynamicDist(ParticleDist):
    gini_limit = 50
    n_ess_limit = 0.5
    a = 0.1 # a fudge factor to prevent degeneracy
    b = 1.2 # additional fudge factor for resampling
    def __init__(self, omegas, prior):
        self.omegas = omegas[deterministic_sample(self.size, prior)]
        self.dist = np.ones(self.size) / self.size
        self.target_cov = self.cov() # initialize target covariance to actual covariance
    def wait_u(self):
        self.omegas = perturb_omega(self.omegas)
        self.target_cov += var_omega
    def update(self, t, m):
        old_cov = self.cov()
        self.dist *= get_likelihood(self.omegas, t, m)
        self.normalize()
        new_cov = self.cov()
        self.target_cov *= max(self.a, new_cov / old_cov) # assume target covariance changes by same amount, add fudge factor to prevent degeneracy
        if self.n_ess() < self.n_ess_limit * self.size:
        #if gini(self.dist) > self.gini_limit / self.size: # gini method (may be better? but seems like not)
            self.resample()
    def cov(self):
        return np.cov(self.omegas, ddof=0, aweights=self.dist)
    def n_ess(self): # compute the effective sample size
        return 1. / np.sum(self.dist**2)
    def resample(self):
        # resample
        self.omegas = self.omegas[deterministic_sample(self.size, self.dist)]
        self.dist = np.ones(self.size) / self.size
        # adjust covariance
        cov_new = self.cov()
        if cov_new > self.target_cov:
            self.target_cov = cov_new # we ended up with more variance than expected, that's cool
        else:
            add_var = self.b * (self.target_cov - cov_new)
            epsilon = ( np.random.exponential(scale=np.sqrt(add_var/2), size=self.size) -
                        np.random.exponential(scale=np.sqrt(add_var/2), size=self.size) )
            self.omegas = clip_omega(self.omegas + epsilon)


# http://docs.qinfer.org/en/latest/guide/timedep.html#specifying-custom-time-step-updates
class DiffusivePrecessionModel(SimplePrecessionModel):
    def update_timestep(self, modelparams, expparams):
        steps = np.random.normal(0., np.sqrt(var_omega), 
            size=(modelparams.shape[0], 1, expparams.shape[0]))
        return clip_omega(modelparams[:, :, np.newaxis] + steps)

class PriorSample(Distribution):
    n_rvs = 1
    def __init__(self, values, dist):
        self.values = values
        self.dist = dist
    def sample(self, n=1):
        return np.reshape(sample_dist(self.values, self.dist, size=n), (n, 1))


# simple wrapper class for the qinfer implementation
class QinferDist(ParticleDist):
    a = 1.
    h = 0.005
    def __init__(self, omegas, prior):
        self.qinfer_model = DiffusivePrecessionModel(min_freq=omega_min)
        self.qinfer_prior = PriorSample(omegas, prior)
        self.qinfer_updater = qinfer.SMCUpdater( self.qinfer_model, self.size,
            self.qinfer_prior, resampler=LiuWestResampler(self.a, self.h, debug=False) )
    def many_update(self, ts, ms):
        for t, m in zip(ts, ms):
            self.qinfer_updater.update(np.array([m]), np.array([t]))
    def mean(self):
        return self.qinfer_updater.est_mean()
    def posterior_marginal(self, *args, **kwargs):
        return self.qinfer_updater.posterior_marginal(*args, **kwargs)


###############################################################################
##          Estimators for Omega:                                            ##

# estimates omega at the mean of the posterior dist
def grid_mean(omegas, prior, ts, measurements):
    dist = GridDist(omegas, prior)
    dist.many_update(ts, measurements)
    return dist.mean()

# uses particle method to estimate omega
def dynm_mean(omegas, prior, ts, measurements):
    dist = DynamicDist(omegas, prior)
    dist.many_update(ts, measurements)
    return dist.mean()


def qinfer_mean(omegas, prior, ts, measurements):
    dist = QinferDist(omegas, prior)
    dist.many_update(ts, measurements)
    return dist.mean()

##                                                                           ##
###############################################################################



def sample_omega_list(omegas, prior, length):
    omega0 = sample_dist(omegas, prior)
    omega_list = [omega0]
    for i in range(1, length):
        omega_list.append(perturb_omega(omega_list[-1]))
    return omega_list


# given a prior on omega and a measurement strategy, 
# do runs runs, storing the loss for each
# also returns some useful statistics
def do_runs(omegas, prior, get_strat, estimators, runs=1000):
    run_hist = np.zeros((runs, len(estimators)), dtype=np.float64)
    no_exception_yet = [True] * len(estimators) # keep track of which estimators have had exceptions so far
    for r in range(0, runs):
        ts = get_strat()
        omega_list = sample_omega_list(omegas, prior, len(ts))
        ms = many_measure(omega_list, ts)
        # each estimator sees the same measurements
        for i, estimator in enumerate(estimators):
            if no_exception_yet[i]:
                try:
                    omega_est = estimator(omegas, prior, ts, ms)
                    run_hist[r, i] = (omega_list[-1] - omega_est)**2
                except RuntimeError:
                    no_exception_yet[i] = False
                    run_hist[r:, i] = np.nan
    avg_loss = np.squeeze( np.sum(run_hist, axis=0) / runs )
    var_loss = np.squeeze( np.sum(run_hist**2, axis=0) / runs ) - avg_loss**2
    return run_hist, avg_loss, var_loss


# get_get_strat is fn of x
def do_runs_of_x(xlist, omegas, prior, get_get_strat, estimators, runs=1000):
    run_hists, avg_losses, var_losses = [], [], []
    for x in xlist:
        print('\t...\t  ', x) # show progress
        run_hist, avg_loss, var_loss = do_runs(omegas, prior,
            get_get_strat(x), estimators, runs)
        run_hists.append(run_hist)
        avg_losses.append(avg_loss)
        var_losses.append(var_loss)
    return ( np.array(run_hists).transpose((2, 0, 1)),
        np.array(avg_losses).T, np.array(var_losses).T )


def save_x_trace(plottype, xlist, xlistnm, omegas, prior, get_get_strat, estimators, estimator_names, runs=1000):
    run_hists, avg_losses, var_losses = do_runs_of_x(xlist, omegas, prior, get_get_strat, estimators, runs)
    data = {
        'omega_min': omega_min,
        'omega_max': omega_max,
        'v_0': v_0,
        'var_omega': var_omega,
        'NUM_PARTICLES': NUM_PARTICLES,
        'omegas': omegas,
        'prior': prior,
        xlistnm: xlist,
        'estimator_names': estimator_names,
        'get_get_strat': inspect.getsource(get_get_strat),
        'runs': runs,
        'run_hists': run_hists,
        'avg_losses': avg_losses,
        'avg_loss_vars': var_losses,
        'plottype': plottype,
        'particle_params': get_numeric_class_vars(ParticleDist),
        'grid_params': get_numeric_class_vars(GridDist),
        'dynamic_params': get_numeric_class_vars(DynamicDist),
        'qinfer_params': get_numeric_class_vars(QinferDist),
    }
    save_data(data, get_filepath(data['plottype']))



def main():
    omegas = np.linspace(omega_min, omega_max, NUM_PARTICLES)
    prior = normalize(1. + 0.*omegas)
    
    estimators = [grid_mean, dynm_mean, qinfer_mean]
    estimator_names = ['grid_mean', 'dynm_mean', 'qinfer_mean']
    
    whichthing = 0
    
    if whichthing == 0:
        ts = np.random.uniform(0., 4.*np.pi, 300)
        omega_list_true = sample_omega_list(omegas, prior, len(ts))
        print('true omega:', omega_list_true)
        ms = many_measure(omega_list_true, ts)
        print(ms)
        grid = GridDist(omegas, prior)
        dynm = DynamicDist(omegas, prior)
        qnfr = QinferDist(omegas, prior)
        grid.many_update(ts, ms)
        dynm.many_update(ts, ms)
        qnfr.many_update(ts, ms)
        
        new_ts = [grid.pick_t() for i in range(0, 6)]
        print(new_ts)
        omstars = np.linspace(0.1, 1.9, 10000)
        for t in new_ts:
            plt.plot(omstars, 5 * np.sin(t * omstars) / omegas.size)
        
        pin_plot(dynm.omegas, dynm.dist)
        plt.plot(omegas, prior, label='prior')
        plt.plot(omegas, grid.dist, label='posterior')
        
        qinfer_omegas, qinfer_post = qnfr.posterior_marginal(res=25)
        plt.plot(qinfer_omegas, normalize(qinfer_post), label='qinfer posterior')
        
        for dist, nm in [(grid, 'grid_mean'), (dynm, 'dynm_mean'), (qnfr, 'qinfer_mean')]:
            plt.plot([dist.mean()], prior[0], marker='o', label=nm)
        plt.plot(omega_list_true, np.linspace(0., prior[0], len(ts)),
            marker='*', markersize=10, label='true_omega', color=(0., 0., 0.))
        plt.legend()
        plt.ylim(bottom=0.)
        plt.show()
        
    elif whichthing == 1:
        tlist = np.arange(0.1, 9., 0.3)
        def get_get_strat(t):
            def get_strat():
                return [t] * 30
            return get_strat

        save_x_trace('measure_time', tlist, 'tlist',
            omegas, prior, get_get_strat, estimators, estimator_names, runs=100)

    
    elif whichthing == 2:
        pass
    
    elif whichthing == 3:
        pass
    
    elif whichthing == 4:
        N_list = np.array([1, 2, 3, 6, 10, 20, 30, 60, 100, 200, 300, 600, 1000, 2000, 3000, 6000, 10000, 30000, 100000])
        def get_get_strat(N):
            t_min = 0.
            t_max = 4. * np.pi
            def get_strat():
                return np.random.uniform(t_min, t_max, N)
            return get_strat
        save_x_trace('measurement_performance', N_list, 'N_list',
            omegas, prior, get_get_strat, estimators, estimator_names, runs=2000)
    
    elif whichthing == 5:
        nlist = np.concatenate([np.arange(1, 10, 1), np.arange(10, 20, 2),
            np.arange(20, 100, 10), np.arange(100, 200, 20), np.arange(200, 1000, 100)])
        def get_get_strat(n):
            def get_strat():
                ts = [7.3] * n
                return ts
            return get_strat

        save_x_trace('measure_number', nlist, 'nlist',
            omegas, prior, get_get_strat, estimators, estimator_names)


if __name__ == '__main__':
    main()


