from estimators import *
from sys import argv

omega_min = 100000.
omega_max = 200000.


class ExperimentalEstimator(Estimator):
    def __init__(self, dist, exp_ts, exp_ms, exp_t_us):
        self.dist = dist
        self.exp_ts = exp_ts
        self.exp_ms = exp_ms
        self.exp_t_us = exp_t_us
    def many_measure(self, whichts=None, t_u_list=None):
        """ whichts determines the subset of ts which we actually do inference
            with. should be a list of integers in increasing order """
        if whichts is None:
            whichts = np.arange(len(self.exp_ts))
        for i in whichts:
            if i > 0:
                self.dist.wait_u(self.exp_t_us[i] - self.exp_t_us[i-1])
            self.dist.update(self.exp_ts[i], self.exp_ms[i])
            if True and i > 2044 and i % 1 == 0:
                print(i)
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
    
    exp_t_us = [0.] # here, u = 1s
    curr_t = exp_ts[0]
    for i in range(1, len(exp_ts)):
        if exp_ts[i] == curr_t:
            exp_t_us.append(exp_t_us[-1] + 0.03) # 30 ms between shots
        else:
            curr_t = exp_ts[i]
            exp_t_us.append(exp_t_us[-1] + 1.) # 1s delay to send data
    plt.plot(exp_t_us); plt.show()
    
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
    
    est = ExperimentalEstimator(GridDist1D(omegas, omega_prior, 100.), exp_ts, exp_ms, exp_t_us)
    est.many_measure(np.arange(len(exp_ts)), exp_t_us) # randomize measurements to prevent lots of low t measurements sending weights to 0
    plt.plot(est.dist.omegas, est.dist.dist) ; plt.show()
    mean_omega = est.mean_omega()
    print(mean_omega)
    plt.plot(times, fractions)
    ts = np.linspace(exp_ts[0], exp_ts[-1], 200)
    plt.plot(ts, likelihood(mean_omega, ts, 1))
    plt.show()
    


if __name__ == '__main__':
    main()

