import numpy as np
from random import randint, random
from scipy.special import gammaln
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from util import save_data, get_filepath


# constants:
omega_min = 0.8     # [1/s]
omega_max = 1.2     # [1/s]


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
    return np.sin(omega * t * 0.5)**2


# returns the number of excited states measured
def measure(omega, t, n):
    return np.random.binomial(n, prob_excited(t, omega))

# make many measurements given a list of ts, ns
def many_measure(omega, ts, ns):
    return np.array([measure(omega, t, n) for t, n in zip(ts, ns)])


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
def likelihood(omega, ts, ns, measurements):
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
    post = get_posterior(prior, likelihood(omegas, ts, ns, measurements))
    return omegas[np.argmax(post)]

# estimates omega at the mean of the posterior dist
# takes a set of measurements at times ts, and numbers ns
def omega_mmse(omegas, prior, ts, ns, measurements):
    post = get_posterior(prior, likelihood(omegas, ts, ns, measurements))
    return np.sum(omegas * post, axis=-1)

# perform a fit based on the probability estimators
# p_est are the estimated probabilities
def omega_fit_unweighted(omegas, prior, ts, ns, measurements):
    p_est = (1. + np.array(measurements)) / (2. + np.array(ns)) # (beta distribution mean)
    inloop = True
    errcount = 0
    while inloop:
        try:
            omega_est, uncertainty = curve_fit(
                prob_excited, ts, p_est,
                p0=[1.], method='lm'
            )
            inloop = False
        except RuntimeError:
            errcount += 1
            p_est += 0.001 * random() # try again with slightly different values
            print('\t', errcount, '!')
    return omega_est[0]

# perform a fit based on the probability estimators
# p_est are the estimated probabilities
def omega_fit_weighted(omegas, prior, ts, ns, measurements):
    m = np.array(measurements)
    n = np.array(ns)
    p_est = (1. + m) / (2. + n)
    var_est = (m * (n - m) + n + 1.) / ((2 + n)**2 * (3 + n)) # (beta distribution variance)
    inloop = True
    errcount = 0
    while inloop:
        try:
            omega_est, uncertainty = curve_fit(
                prob_excited, ts, p_est, sigma=np.sqrt(var_est),
                p0=[1.], method='lm'
            )
            inloop = False
        except RuntimeError:
            errcount += 1
            p_est += 0.001 * random() # try again with slightly different values
            print('\t', errcount, '!')
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
    post = get_posterior(prior, likelihood(omegas, ts, ns, measurements))
    return np.sum((2. * theta / omegas) * post, axis=-1)

##                                                                           ##
###############################################################################


# given a prior on omega and a measurement strategy, compute the average loss using monte-carlo
# loss is the squared difference between estimator and true
# each estimator is a fn taking (omegas, prior, ts, ns, measurements)
def avg_loss_all_omega(omegas, prior, strat, estimators, runs=1000):
    ts, ns = strat
    avg = np.zeros(len(estimators), dtype=np.float64)
    avgsq = np.zeros(len(estimators), dtype=np.float64)
    for r in range(0, runs):
        omega = sample_dist(omegas, prior)
        ms = many_measure(omega, ts, ns)
        # each estimator sees the same measurements
        for i, estimator in enumerate(estimators):
            omega_est = estimator(omegas, prior, ts, ns, ms)
            avg[i] += (omega - omega_est)**2
            avgsq[i] += (omega - omega_est)**4
    return avg / runs, ((avgsq / runs) - (avg / runs)**2) / runs

# like avg_loss_all_omega, but the loss is (np.sin(omega * t_theta_est / 2.) - np.sin(theta))**2
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


# NOTE: assumes unvarying omega
def main():
    ts = [1.0, 4.5]
    ns = [20, 20]
    omegas = np.arange(omega_min, omega_max, 0.01)
    prior = normalize(1. + 0.*omegas)
    
    estimators = [omega_mle, omega_map, omega_mmse, omega_fit_unweighted, omega_fit_weighted]
    estimator_names = ['mle', 'map', 'mmse', 'fit_unweighted', 'fit_weighted']
    
    t_estimators = [t_omega_mle, t_omega_map, t_omega_mmse, t_omega_fit_unweighted, t_omega_fit_weighted, t_mmse]
    t_estimator_names = ['omega_mle', 'omega_map', 'omega_mmse', 'omega_fit_unweighted', 'omega_fit_weighted', 'mmse']
    
    whichthing = 3
    
    if whichthing == 0:
        pass
        '''
        omega_true = sample_dist(omegas, prior)
        print('true omega:', omega_true)
        ms = many_measure(omega_true, ts, ns)
        print(ms)
        omega_mle = omega_mle(omegas, None, ts, ns, ms)
        omega_map = omega_map(omegas, prior, ts, ns, ms)
        omega_mean = mean(omegas, prior, ts, ns, ms)
        plt.plot([omega_mle], [0.0], color=(0., 1., 0.), marker='o')
        plt.plot(omegas, normalize(likelihood(omegas, ts, ns, ms)), color=(0., 1., 0.))
        plt.plot([omega_map], [0.0], color=(0., 0., 1.), marker='o')
        plt.plot([omega_mean], [0.0], color=(0.5, 0.0, 1.), marker='o')
        plt.plot(omegas, get_posterior(prior, likelihood(omegas, ts, ns, ms)), color=(0., 0., 1.))
        plt.plot(omegas, prior, color=(1., 0., 0.))
        plt.plot([omega_true], [0.0], color=(1., 0., 0.), marker='o')
        plt.ylim(bottom=0.)
        plt.show()
        '''
    elif whichthing == 1:
        t_change_idx = ts.index(None)
        tlist = np.arange(0.1, 25., 0.1)
        
        avg_losses = [[] for i in range(0, len(estimators))]
        avg_loss_vars = [[] for i in range(0, len(estimators))]
        for t in tlist:
            print(t)
            ts[t_change_idx] = t
            avgloss, avgloss_var = avg_loss_all_omega(omegas, prior,
                (ts, ns), estimators, 1000)
            for i in range(0, len(estimators)):
                avg_losses[i].append(avgloss[i])
                avg_loss_vars[i].append(avgloss_var[i])
        
        ts[t_change_idx] = None
        data = {
            'omega_min': omega_min,
            'omega_max': omega_max,
            'ts': ts,
            'ns': ns,
            'omegas': omegas,
            'prior': prior,
            'tlist': tlist,
            'estimator_names': estimator_names,
            'avg_losses': avg_losses,
            'avg_loss_vars': avg_loss_vars,
            'plottype': 'measure_time'
        }
        save_data(data, get_filepath(data['plottype']))
    
    elif whichthing == 2:
        N = 100
        nshots_list = np.arange(1, 25, 1, dtype=np.int64)
        t_min = 0.
        t_max = 20. * np.pi
        avg_losses = [[] for i in range(0, len(estimators))]
        avg_loss_vars = [[] for i in range(0, len(estimators))]
        for nshots in nshots_list:
            print(nshots, nshots * (N // nshots))
            ts = np.linspace(t_max, t_min, nshots, endpoint=False)
            ns = (N // nshots) * np.ones(nshots, dtype=np.int64)
            avgloss, avgloss_var = avg_loss_all_omega(omegas, prior,
                (ts, ns), estimators, 1000)
            for i in range(0, len(estimators)):
                avg_losses[i].append(avgloss[i])
                avg_loss_vars[i].append(avgloss_var[i])
        
        data = {
            'omega_min': omega_min,
            'omega_max': omega_max,
            't_min': t_min,
            't_max': t_max,
            'N': N,
            'nshots_list': nshots_list,
            'omegas': omegas,
            'prior': prior,
            'estimator_names': estimator_names,
            'avg_losses': avg_losses,
            'avg_loss_vars': avg_loss_vars,
            'plottype': 'shot_number'
        }
        save_data(data, get_filepath(data['plottype']))
    
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
            'theta_list': theta_list,
            'omegas': omegas,
            'prior': prior,
            't_estimator_names': t_estimator_names,
            'avg_losses': avg_losses,
            'avg_loss_vars': avg_loss_vars,
            'plottype': 't_theta_loss'
        }
        save_data(data, get_filepath(data['plottype']))
        


if __name__ == '__main__':
    main()

