import numpy as np
from random import randint, random
from matplotlib import pyplot as plt
from scipy.special import gammaln
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
def prob_excited(omega, t):
    return np.sin(omega * t * 0.5)**2


# returns the number of excited states measured
def measure(omega, t, n):
    return np.random.binomial(n, prob_excited(omega, t))

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
            m * np.log(prob_excited(omega, t)) +             # p^m
            (n - m) * np.log(1. - prob_excited(omega, t))    # (1-p)^(n-m)
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

##                                                                           ##
###############################################################################


# given a prior on omega0 and a measurement strategy, compute the average loss using monte-carlo
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
    ts = [7.8, 21., 0.]
    ns = [33, 33, 33]
    omegas = np.arange(omega_min, omega_max, 0.001)
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
        mle, mpe, mmse = [], [], []
        mle_var, mpe_var, mmse_var = [], [], []
        t2list = np.arange(0.1, 38., 0.3)
        for t2 in t2list:
            print(t2)
            ts[2] = t2
            avl, avl_var = avg_loss_all_omega(omegas, prior, (ts, ns), [max_likelihood, max_ap, mean], 1000)
            mle.append(avl[0]); mpe.append(avl[1]); mmse.append(avl[2])
            mle_var.append(avl_var[0]); mpe_var.append(avl_var[1]); mmse_var.append(avl_var[2])
        
        data = {
            'omega_min': omega_min,
            'omega_max': omega_max,
            'ts': ts,
            'ns': ns,
            'omegas': omegas,
            'prior': prior,
            'tlist': t2list,
            'mle': mle,
            'mpe': mpe,
            'mmse': mmse,
            'mle_var': mle_var,
            'mpe_var': mpe_var,
            'mmse_var': mmse_var,
            'plottype': 'measure_time'
        }
        save_data(data, get_filepath(data['plottype']))



if __name__ == '__main__':
    main()

