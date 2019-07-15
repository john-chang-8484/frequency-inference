import numpy as np
from random import randint, random
from scipy.special import gammaln
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from util import save_data, get_filepath
import inspect


# constants:
omega_min = 0.8     # [1/s]
omega_max = 1.2     # [1/s]
v_0       = 0.3     # [1/s] # the noise in omega (essentially a decoherence rate)
var_omega = 0.001   # [s^2/u] # the variance in omega per u, where u is the time between measurements


# normalize a discrete probability distribution
def normalize(dist):
    return dist / np.sum(dist, axis=-1)


# randomly sample from a distribution
# values is a list of values
# dist is a probability distribution on those values
def sample_dist(values, dist):
    dist = normalize(dist)
    sample = random()
    total = 0.
    for val, dp in zip(values, dist):
        total += dp
        if total >= sample:
            return val
    return values[-1]


# given omega, prior, and likelihood arrays, computes the posterior distribution
# (this posterior is on the *discrete* values of omega given)
def get_posterior(prior, likelihood):
    return normalize(prior * likelihood)


# probability of excitation at time t for a given value of omega
def prob_excited(t, omega):
    return 0.5 * (1. - (np.exp(- v_0 * t) * np.cos(omega * t)))


# returns the number of excited states measured
def measure(omega, t, n):
    return np.random.binomial(n, prob_excited(t, omega))

# make many measurements given a list of omegas, ts, ns
# assume spacing between measurements is large
def many_measure(omega_list, ts, ns):
    return np.array([measure(omega, t, n) for omega, t, n in zip(omega_list, ts, ns)])


# gives the log-likelihood of a particular omega, given some set of measurements
# NOTE: assumes unvarying omega
def log_likelihood(omega, ts, ns, measurements):
    ans = 0.0
    for t, n, m in zip(ts, ns, measurements):
        pe = prob_excited(t, omega)
        ans += (
            gammaln(1 + n) - gammaln(1 + m) - gammaln(1 + n - m) +  # binomial coefficient
            m * np.log(pe) +             # p^m
            (n - m) * np.log(1. - pe)    # (1-p)^(n-m)
        )
    ans[np.isnan(ans)] = -np.inf # deal with zero values
    return ans

# NOTE: assumes unvarying omega
def get_likelihood(omega, ts, ns, measurements):
    return np.exp(log_likelihood(omega, ts, ns, measurements))

###############################################################################
##          Estimators for Omega:                                            ##

# maximum likelihood estimator for omega
# takes a set of measurements at times ts, and numbers ns
# prior is unused
def omega_mle(omegas, prior, ts, ns, measurements):
    log_likelihoods = log_likelihood(omegas, ts, ns, measurements)
    return omegas[np.argmax(log_likelihoods)]

# maximum a posteriori estimator for omega
# takes a set of measurements at times ts, and numbers ns
def omega_map(omegas, prior, ts, ns, measurements):
    post = get_posterior(prior, get_likelihood(omegas, ts, ns, measurements))
    return omegas[np.argmax(post)]

# estimates omega at the mean of the posterior dist
# takes a set of measurements at times ts, and numbers ns
def omega_mmse(omegas, prior, ts, ns, measurements):
    post = get_posterior(prior, get_likelihood(omegas, ts, ns, measurements))
    return np.sum(omegas * post, axis=-1)

# perform a fit based on the probability estimators
# p_est are the estimated probabilities
def omega_fit_unweighted(omegas, prior, ts, ns, measurements):
    p_est = (1. + np.array(measurements)) / (2. + np.array(ns)) # (beta distribution mean)
    omega_est, uncertainty = curve_fit(
        prob_excited, ts, p_est,
        p0=[1.], method='lm'
    )
    return omega_est[0]

# perform a fit based on the probability estimators
# p_est are the estimated probabilities
def omega_fit_weighted(omegas, prior, ts, ns, measurements):
    m = np.array(measurements)
    n = np.array(ns)
    p_est = (1. + m) / (2. + n)
    var_est = (m * (n - m) + n + 1.) / ((2 + n)**2 * (3 + n)) # (beta distribution variance)
    omega_est, uncertainty = curve_fit(
        prob_excited, ts, p_est, sigma=np.sqrt(var_est),
        p0=[1.], method='lm'
    )
    return omega_est[0]

##                                                                           ##
###############################################################################


###############################################################################
##          Estimators for t_theta:                                          ##

# estimate t_theta from maximum likelihood estimator for omega
def t_omega_mle(theta, omegas, prior, ts, ns, measurements):
    omega_est = omega_mle(omegas, prior, ts, ns, measurements)
    return 2 * theta / omega_est

# estimate t_theta from maximum a posteriori estimator for omega
def t_omega_map(theta, omegas, prior, ts, ns, measurements):
    omega_est = omega_map(omegas, prior, ts, ns, measurements)
    return 2 * theta / omega_est

# estimate t_theta from the "mean of posterior" estimator for omega
def t_omega_mmse(theta, omegas, prior, ts, ns, measurements):
    omega_est = omega_mmse(omegas, prior, ts, ns, measurements)
    return 2 * theta / omega_est

# estimate t_theta from the unweighted fit estimator for omega
def t_omega_fit_unweighted(theta, omegas, prior, ts, ns, measurements):
    omega_est = omega_fit_unweighted(omegas, prior, ts, ns, measurements)
    return 2 * theta / omega_est

# estimate t_theta from the weighted fit estimator for omega
def t_omega_fit_weighted(theta, omegas, prior, ts, ns, measurements):
    omega_est = omega_fit_weighted(omegas, prior, ts, ns, measurements)
    return 2 * theta / omega_est

# estimate t_theta by taking the mean of the posterior t_theta distribution
def t_mmse(theta, omegas, prior, ts, ns, measurements):
    post = get_posterior(prior, get_likelihood(omegas, ts, ns, measurements))
    return np.sum((2. * theta / omegas) * post, axis=-1)

##                                                                           ##
###############################################################################

'''
# like avg_loss, but the loss is (np.sin(omega * t_theta_est / 2.) - np.sin(theta))**2
def avg_t_loss_all_omega(theta, omegas, prior, strat, estimators, runs=1000):
    ts, ns = strat
    avg = np.zeros(len(estimators), dtype=np.float64)
    avgsq = np.zeros(len(estimators), dtype=np.float64)
    for r in range(0, runs):
        omega = sample_dist(omegas, prior)
        t_theta = 2. * theta / omega
        ms = many_measure(omega, ts, ns)
        # each estimator sees the same measurements
        for i, estimator in enumerate(estimators):
            t_theta_est = estimator(theta, omegas, prior, ts, ns, ms)
            avg[i] += (np.sin(omega * t_theta_est / 2.) - np.sin(theta))**2
            avgsq[i] += (np.sin(omega * t_theta_est / 2.) - np.sin(theta))**4
    return avg / runs, ((avgsq / runs) - (avg / runs)**2) / runs
'''

def sample_omega_list(omegas, prior, length):
    omega0 = sample_dist(omegas, prior)
    omega_list = [omega0]
    for i in range(1, length):
        omega_list.append(np.clip(
            omega_list[-1] + np.random.normal(0., np.sqrt(var_omega)),
            omega_min, omega_max ))
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
        'plottype': plottype
    }
    save_data(data, get_filepath(data['plottype']))



def main():
    omegas = np.arange(omega_min, omega_max, 0.01)
    prior = normalize(1. + 0.*omegas)
    
    estimators = [omega_mle, omega_map, omega_mmse, omega_fit_unweighted, omega_fit_weighted]
    estimator_names = ['mle', 'map', 'mmse', 'fit_unweighted', 'fit_weighted']
    
    t_estimators = [t_omega_mle, t_omega_map, t_omega_mmse, t_omega_fit_unweighted, t_omega_fit_weighted, t_mmse]
    t_estimator_names = ['omega_mle', 'omega_map', 'omega_mmse', 'omega_fit_unweighted', 'omega_fit_weighted', 'mmse']
    
    whichthing = 1
    
    if whichthing == 0:
        ts = [7.9]
        ns = [100]
        omegas = np.arange(omega_min, omega_max, 0.001)
        prior = normalize(1. + 0.*omegas)
        omega_true = sample_dist(omegas, prior)
        print('true omega:', omega_true)
        ms = many_measure(omega_true, ts, ns)
        print(ms)
        likelihood = normalize(get_likelihood(omegas, ts, ns, ms))
        posterior = get_posterior(prior, likelihood)
        for estimator, nm in zip(estimators, estimator_names):
            plt.plot([estimator(omegas, prior, ts, ns, ms)], prior[0], marker='o', label=nm)
        plt.plot(omegas, prior, label='prior')
        plt.plot(omegas, likelihood, label='likelihood')
        plt.plot(omegas, posterior, label='posterior')
        plt.plot(omega_true, prior[0], marker='*', markersize=10, label='true_omega', color=(0., 0., 0.))
        plt.legend()
        plt.ylim(bottom=0.)
        plt.show()
        
    elif whichthing == 1:
        tlist = np.arange(0.1, 28., 0.1)
        def get_get_strat(t):
            def get_strat():
                ts = [t] * 30
                ns = [1] * 30
                return ts, ns
            return get_strat

        save_x_trace('measure_time', tlist, 'tlist',
            omegas, prior, get_get_strat, estimators, estimator_names)

    
    elif whichthing == 2:
        nshots_list = np.array([1, 2, 4, 5, 10, 20, 25, 50, 100], dtype=np.int64)
        def get_get_strat(nshots):
            N = 100
            t_min = 0.
            t_max = 4. * np.pi
            def get_strat():
                ts = np.random.uniform(t_min, t_max, nshots)
                ns = (N // nshots) * np.ones(nshots, dtype=np.int64)
                return ts, ns
            return get_strat

        save_x_trace('shot_number', nshots_list, 'nshots_list',
            omegas, prior, get_get_strat, estimators, estimator_names)
    
    elif whichthing == 3:
        theta_list = np.linspace(0., 4. * np.pi, 100)
        avg_losses = [[] for i in range(0, len(t_estimators))]
        avg_loss_vars = [[] for i in range(0, len(t_estimators))]
        for theta in theta_list:
            print(theta)
            avgloss, avgloss_var = avg_t_loss_all_omega(theta, omegas, prior,
                (ts, ns), t_estimators, 1000)
            for i in range(0, len(t_estimators)):
                avg_losses[i].append(avgloss[i])
                avg_loss_vars[i].append(avgloss_var[i])
        data = {
            'omega_min': omega_min,
            'omega_max': omega_max,
            'v_0': v_0,
            'var_omega': var_omega,
            'theta_list': theta_list,
            'omegas': omegas,
            'prior': prior,
            'ts': ts,
            'ns': ns,
            't_estimator_names': t_estimator_names,
            'avg_losses': avg_losses,
            'avg_loss_vars': avg_loss_vars,
            'plottype': 't_theta_loss'
        }
        save_data(data, get_filepath(data['plottype']))
    
    elif whichthing == 4:
        N_list = np.arange(1, 100, 1, dtype=np.int64)
        def get_get_strat(N):
            t_min = 0.
            t_max = 4. * np.pi
            def get_strat():
                ts = np.random.uniform(t_min, t_max, N)
                ns = np.ones(N, dtype=np.int64)
                return ts, ns
            return get_strat
        save_x_trace('measurement_performance', N_list, 'N_list',
            omegas, prior, get_get_strat, estimators, estimator_names)
    
    elif whichthing == 5:
        nlist = np.arange(15, 40, 1, dtype=np.int64)
        def get_get_strat(n):
            def get_strat():
                ts = [7.3]
                ns = [n]
                return ts, ns
            return get_strat

        save_x_trace('measure_number', nlist, 'nlist',
            omegas, prior, get_get_strat, estimators, estimator_names)



if __name__ == '__main__':
    main()

