import numpy as np
from random import randint, random
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
from util import save_data, get_filepath, gammaln, get_numeric_class_vars
from plot_util import pin_plot
import inspect
import math
import qinfer
from qinfer import SimplePrecessionModel, Distribution


# constants:
omega_min = 0.1     # [1/s]
omega_max = 1.9     # [1/s]
v_0       = 0.0     # [1/s]     # the noise in omega (essentially a decoherence rate)
var_omega = 0.000   # [1/s^2/u] # the variance in omega per u, where u is the time between measurements

NUM_PARTICLES = 100



# probability of excitation at time t for a given value of omega
def prob_excited(t, omega):
    return 0.5 * (1. - (np.exp(- 0.5 * v_0 * t) * np.cos(omega * t)))


# normalize a discrete probability distribution
def normalize(dist):
    return dist / np.sum(dist, axis=-1)


# clip a set of omega values to the range
def clip_omega(omegas):
    return np.clip(omegas, omega_min, omega_max)


def perturb_omega(omega):
    return clip_omega(omega + np.random.normal(0., np.sqrt(var_omega)))


# randomly sample from a distribution
# values should be evenly spaced and in sorted order
# dist is a probability distribution on values
def sample_dist(values, dist, size=None):
    delta = values[1] - values[0]
    epsilon = np.random.uniform(-delta / 2, delta / 2, size=size)
    x = np.random.choice(values, p=dist, size=size)
    return np.clip(x + epsilon, values[0], values[-1]) # <- this is a little bit hacky


# returns the number of excited states measured
def measure(omega, t, n):
    return np.random.binomial(n, prob_excited(t, omega))

# make many measurements given a list of omegas, ts, ns
# assume spacing between measurements is large
def many_measure(omega_list, ts, ns):
    return np.array([measure(omega, t, n) for omega, t, n in zip(omega_list, ts, ns)])


# gives the log-likelihood of a particular omega, given some measurement m
def log_likelihood(omega, t, n, m):
    pe = prob_excited(t, omega)
    ans = (
        gammaln[1 + n] - gammaln[1 + m] - gammaln[1 + n - m] +  # binomial coefficient
        m * np.log(pe) +             # p^m
        (n - m) * np.log(1. - pe)    # (1-p)^(n-m)
    )
    ans[np.isnan(ans)] = -np.inf # deal with zero values
    return ans

def get_likelihood(omega, t, n, m):
    return np.exp(log_likelihood(omega, t, n, m))


# RULE: all fn calls should preserve normalization 
class ParticleDist:
    size = NUM_PARTICLES
    def normalize(self):
        self.dist = normalize(self.dist)
    def mean(self):
        return np.sum(self.dist * self.omegas)
    def many_update(self, ts, ns, ms):
        for t, n, m in zip(ts, ns, ms):
            self.wait_u()
            self.update(t, n, m)


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
    def update(self, t, n, m):
        self.dist *= get_likelihood(self.omegas, t, n, m)
        self.normalize()


class DynamicDist(ParticleDist):
    prob_mass_limit = 0.15
    a = 1.00  # pg 10, Christopher E Granade et al 2012 New J. Phys. 14 103013
    b = 0.00  # additional fudge factor for resampling
    def __init__(self, omegas, prior):
        self.omegas = np.random.choice(omegas, size=self.size, p=prior)
        self.dist = np.ones(self.size) / self.size
        self.probability_mass = 1. # fraction of probability mass remaining since last resampling
    def wait_u(self):
        self.omegas = perturb_omega(self.omegas)
    def update(self, t, n, m):
        self.dist *= get_likelihood(self.omegas, t, n, m)
        self.probability_mass *= np.sum(self.dist)
        self.normalize()
        if self.probability_mass < self.prob_mass_limit:
            self.resample()
    def cov(self):
        return np.cov(self.omegas, ddof=0, aweights=self.dist)
    def resample(self):
        mu = self.mean()
        sampled_particles = np.random.choice(self.omegas, size=self.size, p=self.dist)
        mu_i = (self.a * sampled_particles) + ((1 - self.a) * mu)
        epsilon = np.sqrt(self.b + self.cov() * (1. - self.a**2)) * np.random.randn(self.size) 
        self.omegas = clip_omega(mu_i + epsilon)
        self.dist = np.ones(self.size) / self.size
        self.probability_mass = 1.
        self.normalize()


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


###############################################################################
##          Estimators for Omega:                                            ##

# estimates omega at the mean of the posterior dist
def grid_mean(omegas, prior, ts, ns, measurements):
    dist = GridDist(omegas, prior)
    dist.many_update(ts, ns, measurements)
    return dist.mean()

# uses particle method to estimate omega
def dynm_mean(omegas, prior, ts, ns, measurements):
    dist = DynamicDist(omegas, prior)
    dist.many_update(ts, ns, measurements)
    return dist.mean()

# benchmark qinfer implementation
qinfer_model = DiffusivePrecessionModel(min_freq=omega_min)
# CAUTION: assumes all ns are equal to 1
# make a qinfer inductor, update it with all measurements
def qinfer_make(omegas, prior, ts, ns, measurements):
    qinfer_prior = PriorSample(omegas, prior)
    qinfer_updater = qinfer.SMCUpdater(qinfer_model, NUM_PARTICLES, qinfer_prior)
    for t, n, m in zip(ts, ns, measurements):
        qinfer_updater.update(np.array([m]), np.array([t]))
    return qinfer_updater
# CAUTION: assumes all ns are equal to 1
def qinfer_mean(omegas, prior, ts, ns, measurements):
    return qinfer_make(omegas, prior, ts, ns, measurements).est_mean()

##                                                                           ##
###############################################################################



def sample_omega_list(omegas, prior, length):
    omega0 = sample_dist(omegas, prior)
    omega_list = [omega0]
    for i in range(1, length):
        omega_list.append(perturb_omega(omega_list[-1]))
    return omega_list


# given a prior on omega and a measurement strategy, compute the average loss using monte-carlo
# loss is the squared difference between estimator and omega's true *final* value
# each estimator is a fn taking (omegas, prior, ts, ns, measurements)
# get_strat is fn that produces a strategy, may make calls to random
def avg_loss(omegas, prior, get_strat, estimators, runs=1000):
    avg = np.zeros(len(estimators), dtype=np.float64)
    avgsq = np.zeros(len(estimators), dtype=np.float64)
    no_exception_yet = [True] * len(estimators) # keep track of which estimators have had exceptions so far
    for r in range(0, runs):
        ts, ns = get_strat()
        omega_list = sample_omega_list(omegas, prior, len(ts))
        ms = many_measure(omega_list, ts, ns)
        # each estimator sees the same measurements
        for i, estimator in enumerate(estimators):
            if no_exception_yet[i]:
                try:
                    omega_est = estimator(omegas, prior, ts, ns, ms)
                    avg[i] += (omega_list[-1] - omega_est)**2
                    avgsq[i] += (omega_list[-1] - omega_est)**4
                except RuntimeError:
                    no_exception_yet[i] = False
                    avg[i] = np.nan
                    avgsq[i] = np.nan
    return avg / runs, ((avgsq / runs) - (avg / runs)**2) / runs

# compute median loss
def med_loss(omegas, prior, get_strat, estimators, runs=1000):
    data = np.zeros((runs, len(estimators)), dtype=np.float64)
    avgsq = np.zeros(len(estimators), dtype=np.float64)
    no_exception_yet = [True] * len(estimators) # keep track of which estimators have had exceptions so far
    for r in range(0, runs):
        ts, ns = get_strat()
        omega_list = sample_omega_list(omegas, prior, len(ts))
        ms = many_measure(omega_list, ts, ns)
        # each estimator sees the same measurements
        for i, estimator in enumerate(estimators):
            if no_exception_yet[i]:
                try:
                    omega_est = estimator(omegas, prior, ts, ns, ms)
                    data[r, i] = (omega_list[-1] - omega_est)**2
                    avgsq[i] += (omega_list[-1] - omega_est)**4
                except RuntimeError:
                    no_exception_yet[i] = False
                    data[r, i] = np.nan
                    avgsq[i] = np.nan
    return np.median(data, axis=0), ((avgsq / runs) - (np.sum(data, axis=0) / runs)**2)

# get_get_strat is fn of x
def avg_loss_of_x(xlist, omegas, prior, get_get_strat, estimators, runs=1000):
    avg_losses = [[] for i in range(0, len(estimators))]
    avg_loss_vars = [[] for i in range(0, len(estimators))]
    for x in xlist:
        print(x)
        avgloss_x, avgloss_var_x = avg_loss(omegas, prior,
            get_get_strat(x), estimators, runs)
        for i in range(0, len(estimators)):
            avg_losses[i].append(avgloss_x[i])
            avg_loss_vars[i].append(avgloss_var_x[i])
    return avg_losses, avg_loss_vars


def save_x_trace(plottype, xlist, xlistnm, omegas, prior, get_get_strat, estimators, estimator_names, runs=1000):
    avg_losses, avg_loss_vars = avg_loss_of_x(xlist, omegas, prior, get_get_strat, estimators, runs)
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
        'avg_losses': avg_losses,
        'avg_loss_vars': avg_loss_vars,
        'plottype': plottype,
        'particle_params': get_numeric_class_vars(ParticleDist),
        'grid_params': get_numeric_class_vars(GridDist),
        'dynamic_params': get_numeric_class_vars(DynamicDist),
    }
    save_data(data, get_filepath(data['plottype']))



def main():
    omegas = np.linspace(omega_min, omega_max, NUM_PARTICLES)
    prior = normalize(1. + 0.*omegas)
    
    estimators = [grid_mean, dynm_mean, qinfer_mean]
    estimator_names = ['grid_mean', 'dynm_mean', 'qinfer_mean']
    
    whichthing = 0
    
    if whichthing == 0:
        ts = np.random.uniform(0., 4.*np.pi, 3)
        ns = [1] * 3
        omega_list_true = sample_omega_list(omegas, prior, len(ts))
        print('true omega:', omega_list_true)
        ms = many_measure(omega_list_true, ts, ns)
        print(ms)
        grid = GridDist(omegas, prior)
        dynm = DynamicDist(omegas, prior)
        grid.many_update(ts, ns, ms)
        dynm.many_update(ts, ns, ms)
        qinfer_updater = qinfer_make(omegas, prior, ts, ns, ms)
        
        pin_plot(dynm.omegas, dynm.dist)
        plt.plot(omegas, prior, label='prior')
        plt.plot(omegas, grid.dist, label='posterior')
        
        qinfer_omegas, qinfer_post = qinfer_updater.posterior_marginal(res=25)
        plt.plot(qinfer_omegas, normalize(qinfer_post), label='qinfer posterior')
        
        for dist, nm in [(grid, 'grid_mean'), (dynm, 'dynm_mean'), (qinfer_updater.est_mean(), 'qinfer_mean')]:
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
                ts = [t] * 30
                ns = [1] * 30
                return ts, ns
            return get_strat

        save_x_trace('measure_time', tlist, 'tlist',
            omegas, prior, get_get_strat, estimators, estimator_names)

    
    elif whichthing == 2:
        pass
    
    elif whichthing == 3:
        pass
    
    elif whichthing == 4:
        N_list = np.array([1, 2, 3, 6, 10, 20, 30, 60, 100, 200, 300, 600, 1000, 2000, 3000, 6000, 10000])
        def get_get_strat(N):
            t_min = 0.
            t_max = 4. * np.pi
            def get_strat():
                ts = np.random.uniform(t_min, t_max, N)
                ns = np.ones(N, dtype=np.int64)
                return ts, ns
            return get_strat
        save_x_trace('measurement_performance', N_list, 'N_list',
            omegas, prior, get_get_strat, estimators, estimator_names, runs=500)
    
    elif whichthing == 5:
        nlist = np.concatenate([np.arange(1, 10, 1), np.arange(10, 20, 2),
            np.arange(20, 100, 10), np.arange(100, 200, 20), np.arange(200, 1000, 100)])
        def get_get_strat(n):
            def get_strat():
                ts = [7.3] * n
                ns = [1] * n
                return ts, ns
            return get_strat

        save_x_trace('measure_number', nlist, 'nlist',
            omegas, prior, get_get_strat, estimators, estimator_names)


if __name__ == '__main__':
    main()


