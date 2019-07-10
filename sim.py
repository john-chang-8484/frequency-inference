import numpy as np
from random import randint, random
from scipy.special import gammaln
from scipy.optimize import curve_fit
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
def posterior(prior, likelihood):
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
        ans += (
            gammaln(1 + n) - gammaln(1 + m) - gammaln(1 + n - m) +  # binomial coefficient
            m * np.log(prob_excited(t, omega)) +             # p^m
            (n - m) * np.log(1. - prob_excited(t, omega))    # (1-p)^(n-m)
        )
    return ans

# NOTE: assumes unvarying omega
def likelihood(omega, ts, ns, measurements):
    return np.exp(log_likelihood(omega, ts, ns, measurements))

###############################################################################
##          Estimators:                                                      ##

# maximum likelihood estimator for omega
# takes a set of measurements at times ts, and numbers ns
# prior is unused
def max_likelihood(omegas, prior, ts, ns, measurements):
    log_likelihoods = log_likelihood(omegas, ts, ns, measurements)
    return omegas[np.argmax(log_likelihoods)]

# maximum a posteriori estimator for omega
# takes a set of measurements at times ts, and numbers ns
def max_ap(omegas, prior, ts, ns, measurements):
    post = posterior(prior, likelihood(omegas, ts, ns, measurements))
    return omegas[np.argmax(post)]

# mean estimator for omega
# takes a set of measurements at times ts, and numbers ns
def mean(omegas, prior, ts, ns, measurements):
    post = posterior(prior, likelihood(omegas, ts, ns, measurements))
    return np.sum(omegas * post, axis=-1)

# perform a fit based on the probability estimators
# p_est are the estimated probabilities
# (see wikipedia on the mean of the beta distribution)
def fit_unweighted(omegas, prior, ts, ns, measurements):
    p_est = (1. + np.array(measurements)) / (2. + np.array(ns))
    inloop = True
    errcount = 0
    while inloop:
        try:
            omega_est, uncertainty = curve_fit(
                prob_excited, ts, p_est,
                p0=[1.], bounds=([omega_min], [omega_max]), method='trf'
            )
            inloop = False
        except RuntimeError:
            errcount += 1
            p_est += 0.0003 * (random() - 0.5) # try again with slightly different values
            print(errcount)
    return omega_est[0]

##                                                                           ##
###############################################################################


# given a prior on omega and a measurement strategy, compute the average loss using monte-carlo
# loss is the squared difference between estimator and true
# each estimator is a fn taking (omegas, prior, ts, ns, measurements)
# runs is the number of monte-carlo runs to do
# returns an array corresponding to the average loss for each estimator, and a variance of the loss
# NOTE: assumes unvarying omega
def avg_loss_all_omega(omegas, prior, strat, estimators, runs=1000):
    ts, ns = strat
    avg = np.zeros(len(estimators), dtype=np.float64)
    avgsq = np.zeros(len(estimators), dtype=np.float64)
    for r in range(0, runs):
        omega = sample_dist(omegas, prior)
        ms = many_measure(omega, ts, ns)
        for i, estimator in enumerate(estimators):
            omega_est = estimator(omegas, prior, ts, ns, ms)
            avg[i] += (omega - omega_est)**2
            avgsq[i] += (omega - omega_est)**4
    return avg / runs, ((avgsq / runs) - (avg / runs)**2) / runs
    


# NOTE: assumes unvarying omega
def main():
    ts = [None]
    ns = [20]
    omegas = np.arange(omega_min, omega_max, 0.01)
    prior = normalize(1. + 0.*omegas)
    
    whichthing = 1
    
    if whichthing == 0:
        omega_true = sample_dist(omegas, prior)
        print('true omega:', omega_true)
        ms = many_measure(omega_true, ts, ns)
        print(ms)
        omega_mle = max_likelihood(omegas, None, ts, ns, ms)
        omega_map = max_ap(omegas, prior, ts, ns, ms)
        omega_mean = mean(omegas, prior, ts, ns, ms)
        plt.plot([omega_mle], [0.0], color=(0., 1., 0.), marker='o')
        plt.plot(omegas, normalize(likelihood(omegas, ts, ns, ms)), color=(0., 1., 0.))
        plt.plot([omega_map], [0.0], color=(0., 0., 1.), marker='o')
        plt.plot([omega_mean], [0.0], color=(0.5, 0.0, 1.), marker='o')
        plt.plot(omegas, posterior(prior, likelihood(omegas, ts, ns, ms)), color=(0., 0., 1.))
        plt.plot(omegas, prior, color=(1., 0., 0.))
        plt.plot([omega_true], [0.0], color=(1., 0., 0.), marker='o')
        plt.ylim(bottom=0.)
        plt.show()
    
    elif whichthing == 1:
        t_change_idx = ts.index(None)
        tlist = np.arange(0.1, 25., 0.1)
        estimators = [max_likelihood, max_ap, mean, fit_unweighted]
        estimator_names = ['mle', 'map', 'mmse', 'fit1']
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



if __name__ == '__main__':
    main()

