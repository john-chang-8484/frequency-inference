import numpy as np
from random import randint, random
#from scipy.special import gammaln
import matplotlib.pyplot as plt
from util import save_data, get_filepath, gammaln
from plot_util import pin_plot
import inspect
import math


# constants:
omega_min = 0.1     # [1/s]
omega_max = 1.9     # [1/s]
v_0       = 0.0     # [1/s]   # the noise in omega (essentially a decoherence rate)
var_omega = 0.00001 # [s^2/u] # the variance in omega per u, where u is the time between measurements


# normalize a discrete probability distribution
def normalize(dist):
    return dist / np.sum(dist, axis=-1)


# clip a set of omega values to the range
def clip_omega(omegas):
    return np.clip(omegas, omega_min, omega_max)


# randomly sample from a distribution
# values should be evenly spaced and in sorted order
# dist is a probability distribution on values
def sample_dist(values, dist):
    delta = values[1] - values[0]
    epsilon = np.random.uniform(-delta / 2, delta / 2)
    x = np.random.choice(values, p=dist)
    return np.clip(x + epsilon, values[0], values[-1]) # <- this is a little bit hacky


# given omega, prior, and likelihood arrays, computes the posterior distribution
# (this posterior is on the *discrete* values of omega given)
def get_posterior(prior, likelihood):
    return normalize(prior * likelihood)


# probability of excitation at time t for a given value of omega
def prob_excited(t, omega):
    return 0.5 * (1. - (np.exp(- 0.5 * v_0 * t) * np.cos(omega * t)))


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


# update prior given a measurement at time t, with n hits, result m
def update(omegas, prior, t, n, m):
    return get_posterior(prior, get_likelihood(omegas, t, n, m))


# given a posterior distribution for omega at time t,
# return the prob dist for omega at time t+u
def wait_u(omegas, dist):
    delta_omega = omegas[1] - omegas[0]
    n = math.ceil(0.1 + (var_omega / delta_omega**2))
    for i in range(n):
        dist_new = np.copy(dist)
        # heat eq evolution
        dist_new[1:-1] += (var_omega / (2. * n * delta_omega**2)) * (
            dist[2:] + dist[:-2] - 2.*dist[1:-1] )
        # boundary conditions
        dist_new[0] += (var_omega / (2. * n * delta_omega**2)) * (dist[1] - dist[0])
        dist_new[-1] += (var_omega / (2. * n * delta_omega**2)) * (dist[-2] - dist[-1])
        dist = dist_new
    return dist_new


# get overall posterior for many measurements
def get_overall_posterior(omegas, prior, ts, ns, measurements):
    post = np.copy(prior)
    for t, n, m in zip(ts, ns, measurements):
        post = update(omegas, wait_u(omegas, post), t, n, m)
    return post


# RULE: all fn calls should preserve normalization
class ParticleDist:
    size = 100
    prob_mass_limit = 0.1
    a = 0.9 # pg 10, Christopher E Granade et al 2012 New J. Phys. 14 103013
    b = 2.9  # additional fudge factor for resampling
    def __init__(self, values, dist):
        self.particles = np.random.choice(values, size=self.size, p=dist)
        self.weights = np.ones(self.size) / self.size
        self.probability_mass = 1. # fraction of probability mass remaining since last resampling
    def normalize(self):
        self.weights = normalize(self.weights)
    def wait_u(self):
        self.particles = clip_omega(self.particles + np.random.normal(0., np.sqrt(var_omega)))
    def update(self, t, n, m):
        self.weights *= get_likelihood(self.particles, t, n, m)
        self.probability_mass *= np.sum(self.weights)
        self.normalize()
        if self.probability_mass < self.prob_mass_limit:
            self.resample()
    def mean(self):
        return np.sum(self.weights * self.particles)
    def cov(self):
        return np.cov(self.particles, ddof=0, aweights=self.weights)
    def resample(self):
        mu = self.mean()
        sampled_particles = np.random.choice(self.particles, size=self.size, p=self.weights)
        mu_i = (self.a * sampled_particles) + ((1 - self.a) * mu)
        epsilon = np.sqrt(self.b * self.cov() * (1. - self.a**2)) * np.random.randn(self.size) 
        self.particles = clip_omega(mu_i + epsilon)
        self.weights = np.ones(self.size) / self.size
        self.probability_mass = 1.
        self.normalize()

def get_particle_posterior(omegas, prior, ts, ns, measurements):
    pdist = ParticleDist(omegas, prior)
    for t, n, m in zip(ts, ns, measurements):
        pdist.wait_u()
        pdist.update(t, n, m)
    return pdist

###############################################################################
##          Estimators for Omega:                                            ##

# estimates omega at the mean of the posterior dist
def omega_mmse(omegas, prior, ts, ns, measurements):
    return np.sum(omegas * get_overall_posterior(omegas, prior, ts, ns, measurements))

# uses particle method to estimate omega
def omega_particles_mmse(omegas, prior, ts, ns, measurements):
    return get_particle_posterior(omegas, prior, ts, ns, measurements).mean()

##                                                                           ##
###############################################################################



def sample_omega_list(omegas, prior, length):
    omega0 = sample_dist(omegas, prior)
    omega_list = [omega0]
    for i in range(1, length):
        omega_list.append(clip_omega(omega_list[-1] + np.random.normal(0., np.sqrt(var_omega))))
    return omega_list


# given a prior on omega and a measurement strategy, compute the average loss using monte-carlo
# loss is the squared difference between estimator and omega's true *final* value
# each estimator is a fn taking (omegas, prior, ts, ns, measurements)
# get strat is fn that produces a strategy, may make calls to random
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

# get_get_strat is fn of x
def avg_loss_of_x(xlist, omegas, prior, get_get_strat, estimators, runs=1000):
    avg_losses = [[] for i in range(0, len(estimators))]
    avg_loss_vars = [[] for i in range(0, len(estimators))]
    for x in xlist:
        print(x)
        avgloss, avgloss_var = avg_loss(omegas, prior,
            get_get_strat(x), estimators, runs)
        for i in range(0, len(estimators)):
            avg_losses[i].append(avgloss[i])
            avg_loss_vars[i].append(avgloss_var[i])
    return avg_losses, avg_loss_vars


def save_x_trace(plottype, xlist, xlistnm, omegas, prior, get_get_strat, estimators, estimator_names, runs=1000):
    avg_losses, avg_loss_vars = avg_loss_of_x(xlist, omegas, prior, get_get_strat, estimators, runs)
    data = {
        'omega_min': omega_min,
        'omega_max': omega_max,
        'v_0': v_0,
        'var_omega': var_omega,
        'omegas': omegas,
        'prior': prior,
        xlistnm: xlist,
        'estimator_names': estimator_names,
        'get_get_strat': inspect.getsource(get_get_strat),
        'runs': runs,
        'avg_losses': avg_losses,
        'avg_loss_vars': avg_loss_vars,
        'plottype': plottype,
        'particle_params': {
            key: vars(ParticleDist)[key]
                for key in vars(ParticleDist)
                if type(vars(ParticleDist)[key]) in [int, float]
        }
    }
    save_data(data, get_filepath(data['plottype']))



def main():
    omegas = np.linspace(omega_min, omega_max, 100)
    prior = normalize(1. + 0.*omegas)
    
    estimators = [omega_mmse, omega_particles_mmse]
    estimator_names = ['mmse', 'particles_mmse']
    
    whichthing = 1
    
    if whichthing == 0:
        ts = np.random.uniform(0., 4.*np.pi, 30)
        ns = [1] * 30
        #omegas = np.arange(omega_min, omega_max, 0.005)
        prior = normalize(1. + 0.*omegas)
        omega_list_true = sample_omega_list(omegas, prior, len(ts))
        print('true omega:', omega_list_true)
        ms = many_measure(omega_list_true, ts, ns)
        print(ms)
        posterior = get_overall_posterior(omegas, prior, ts, ns, ms)
        particle_post = get_particle_posterior(omegas, prior, ts, ns, ms)
        pin_plot(particle_post.particles, particle_post.weights)
        plt.plot(omegas, prior, label='prior')
        plt.plot(omegas, posterior, label='posterior')
        for estimator, nm in zip(estimators, estimator_names):
            if nm != 'particles_mmse':
                plt.plot([estimator(omegas, prior, ts, ns, ms)], prior[0], marker='o', label=nm)
        plt.plot(particle_post.mean(), prior[0], marker='o', label='particles_mmse')
        plt.plot(omega_list_true, np.linspace(0., prior[0], len(ts)),
            marker='*', markersize=10, label='true_omega', color=(0., 0., 0.))
        plt.legend()
        plt.ylim(bottom=0.)
        plt.show()
        
    elif whichthing == 1:
        tlist = np.arange(0.1, 16., 4.3)
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
        N_list = np.concatenate([np.arange(1, 10, 1), np.arange(10, 30, 4),
            np.arange(30, 100, 10), np.arange(100, 300, 40),
            np.arange(300, 1000, 100), np.arange(1000, 3000, 400),
            np.arange(3000, 10000, 1000)])
        def get_get_strat(N):
            t_min = 0.
            t_max = 4. * np.pi
            def get_strat():
                ts = np.random.uniform(t_min, t_max, N)
                ns = np.ones(N, dtype=np.int64)
                return ts, ns
            return get_strat
        save_x_trace('measurement_performance', N_list, 'N_list',
            omegas, prior, get_get_strat, estimators, estimator_names, runs=100)
    
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

    elif whichthing == 6: # numerical stability test for heat eq
        omegas = np.arange(omega_min, omega_max, 0.005)
        dists = [normalize(np.random.uniform(0., 1., len(omegas)))]
        for i in range(0, 50):
            dists.append(wait_u(omegas, dists[-1]))
        for dist in dists:
            plt.plot(dist)
        plt.show()


if __name__ == '__main__':
    main()

