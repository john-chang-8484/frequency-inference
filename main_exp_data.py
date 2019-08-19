from estimators import *
from sys import argv
from scipy.optimize import curve_fit

omega_min = 100000.
omega_max = 200000.


def chunkify_data(exp_ts, exp_ms, exp_t_us, tseq):
    lookup_table = {t: i for i, t in enumerate(tseq)}
    def comp(a, b):
        return lookup_table[a] < lookup_table[b]
    chunked_ts   = [[exp_ts[0]]]
    chunked_ms   = [[exp_ms[0]]]
    chunked_t_us = [[exp_t_us[0]]]
    for i in range(1, len(exp_ts)):
        if not comp(chunked_ts[-1][-1], exp_ts[i]):
            chunked_ts.append([])
            chunked_ms.append([])
            chunked_t_us.append([])
        chunked_ts[-1].append(exp_ts[i])
        chunked_ms[-1].append(exp_ms[i])
        chunked_t_us[-1].append(exp_t_us[i])
    return chunked_ts, chunked_ms, chunked_t_us


class ExperimentalEstimator(Estimator):
    def __init__(self, dist, exp_ts, exp_ms, exp_t_us):
        self.dist = dist
        self.exp_ts = exp_ts
        self.exp_ms = exp_ms
        self.exp_t_us = exp_t_us
    def many_measure(self, whichts=None, t_u_list=None):
        """ whichts determines the subset of ts we actually do inference
            with. should be a list of integers in increasing order
            this fn ignores the t_u_list param """
        if whichts is None:
            whichts = np.arange(len(self.exp_ts))
        for i in whichts:
            if i > 0:
                self.dist.wait_u(self.exp_t_us[i] - self.exp_t_us[i-1])
            self.dist.update(self.exp_ts[i], self.exp_ms[i])


class FittingExperimentalEstimator(ExperimentalEstimator):
    def __init__(self, exp_ts, exp_ms, exp_t_us):
        self.exp_ts = exp_ts
        self.exp_ms = exp_ms
        self.exp_t_us = exp_t_us
    def many_measure(self, whichts=None):
        if whichts is None:
            whichts = np.arange(len(self.exp_ts))
        counts = {}
        for i in whichts:
            t, m = self.exp_ts[i], self.exp_ms[i]
            if t in counts:
                counts[t][0] += 1
                counts[t][1] += m
            else:
                counts[t] = [1, m]
        self.unique_ts = np.array([t for t in counts])
        self.est_probs = np.array([counts[t][1] / counts[t][0] for t in self.unique_ts])
    def mean_omega(self):
        prob_m1 = (lambda t, omega: likelihood(omega, t, 1))
        omega_est, uncertainty = curve_fit(
            prob_m1, self.unique_ts, self.est_probs,
            p0=[mu_omega], method='lm'
        )
        return omega_est


def ChunkChooser(OptimizingChooser):
    def __init__(self, n_omegas, n_ts):
        self.n_omegas, self.n_ts = n_omegas, n_ts
        self.parent_est = None # the parent estimator of this chooser
    def get_potential_ts(self):
        return np.random.choice(self.parent_est.exp_ts[self.parent_est.idx_c],
            size=self.n_ts, replace=False)


class ChunkedExperimentalEstimator(Estimator):
    def __init__(self, dist, chooser, exp_ts, exp_ms, exp_t_us):
        """ the time series data should be pre-chunked
            i.e. exp_ts = [chunk, chunk, chunk, chunk...] 
            where each chunk is a list of ts in the prespecified permutation
            same for exp_ms, exp_t_us
        """
        self.dist = dist
        self.chooser = chooser
        chooser.parent_est = self
        self.exp_ts = exp_ts ; self.exp_ms = exp_ms ; self.exp_t_us = exp_t_us
        self.idx_c = None
    def many_measure(self, start_chunk, end_chunk, measurements_per_chunk):
        self.idx_c = start_chunk
        while self.idx_c <= end_chunk: # loop through all available chunks
            if self.idx_c > 0:
                # we pretend all measurements within a chunk are simultaneous
                delta_t_u = self.exp_t_us[idx_c][0] - self.exp_t_us[idx_c - 1][0]
                self.dist.wait_u(delta_t_u)
            for i in range(measurements_per_chunk):
                t = self.chooser.get_t(self.dist)
                idx_t = self.exp_ts[self.idx_c].index(t)
                m = self.exp_ms[idx_c][idx_t]
                self.dist.update(t, m)
            self.idx_c += 1
        self.idx_c = None


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
    
    est = ExperimentalEstimator(GridDist1D(omegas, omega_prior, 1.e-30), exp_ts, exp_ms, exp_t_us)
    fit_est = FittingExperimentalEstimator(exp_ts, exp_ms, exp_t_us)
    whichts = np.arange(len(exp_ts))#; np.random.shuffle(whichts)
    est.many_measure(whichts)#np.arange(len(exp_ts))[::2], exp_t_us)
    fit_est.many_measure(whichts)
    #plt.plot(est.dist.omegas, est.dist.dist) ; plt.show()
    mean_omega = est.mean_omega()
    fit_omega = fit_est.mean_omega()
    print(mean_omega, fit_omega, (mean_omega - fit_omega)**2)
    plt.plot(times, fractions)
    ts = np.linspace(exp_ts[0], exp_ts[-1], 200)
    plt.plot(ts, likelihood(mean_omega, ts, 1))
    plt.plot(ts, likelihood(fit_omega, ts, 1))
    plt.show()
    


if __name__ == '__main__':
    main()

