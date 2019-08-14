from estimators import *
from sys import argv

omega_min = 100000.
omega_max = 200000.


class ExperimentalEstimator(Estimator):
    def __init__(self, dist, exp_ts, exp_ms):
        self.dist = dist
        self.exp_ts = exp_ts
        self.exp_ms = exp_ms
    def many_measure(self, whichts=None):
        if whichts is None:
            whichts = np.arange(len(self.exp_ts))
        for i in whichts:
            self.dist.wait_u()
            self.dist.update(self.exp_ts[i], self.exp_ms[i])
            if False:
                plt.plot(self.dist.omegas, self.dist.dist)
                plt.plot(self.dist.omegas, normalize(likelihood(self.dist.omegas, self.exp_ts[i], self.exp_ms[i])))
                plt.show()


def main():
    omegas = np.linspace(omega_min, omega_max, 300)
    omega_prior = normalize(1. + 0.*omegas) # uniform prior
    #omega_prior = normalize(np.exp(-160.*(omegas-1)**2)) # normal prior
    log_v1s = np.linspace(-20., -8., 20)
    v1s = np.exp(log_v1s)
    v1_prior = normalize(1. + 0.*v1s)
    prior = np.outer(omega_prior, v1_prior)

    fnames = argv[1:]
    data = np.load(fnames[0])
    exp_ts = data[0]
    exp_ms = data[1]
    
    times, counts, totals = [exp_ts[0]], [0], [0.]
    for t, m in zip(exp_ts, exp_ms):
        if t == times[-1]:
            counts[-1] += 1
            totals[-1] += m
        else:
            times.append(t)
            counts.append(1)
            totals.append(m)
    fractions = np.array(totals) / np.array(counts)
    
    est = ExperimentalEstimator(GridDist1D(omegas, omega_prior, 0.), exp_ts, exp_ms)
    est.many_measure(np.random.randint(0, len(exp_ts)-1, size=len(exp_ts))) # randomize measurements to prevent lots of low t measurements sending weights to 0
    plt.plot(est.dist.omegas, est.dist.dist) ; plt.show()
    mean_omega = est.mean_omega()
    print(mean_omega)
    plt.plot(times, fractions)
    ts = np.linspace(exp_ts[0], exp_ts[-1], 200)
    plt.plot(ts, likelihood(mean_omega, ts, 1))
    plt.show()
    


if __name__ == '__main__':
    main()

