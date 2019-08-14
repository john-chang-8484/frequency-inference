import numpy as np
from estimators import *
from scipy.optimize import curve_fit
from util import Bunch, idprn
import matplotlib.pyplot as plt


class FittingChooser(TimeChooser):
    name = 'fitting'
    def __init__(self, ts, ns):
        self.ts = ts
        self.ns = ns
        self.i = 0
        self.j = 0
    def get_t(self, dist):
        ans = self.ts[self.i]
        self.j += 1
        if self.j == self.ns[self.i]:
            self.i += 1
            self.j = 0
        return ans
        


class FittingEstimator(Estimator):
    def __init__(self, ts, ns):
        self.chooser = FittingChooser(ts, ns)
        self.dist = Bunch({'name': 'none'})
        self.m1s =    np.array([0 for t in ts]) # number of shots where we got m = 1
        self.counts = np.array([0 for t in ts]) # total number of shots
    def mean_omega(self): # note: really this is just an *estimation* of omega, not the expected value
        p_est = self.m1s / self.counts
        prob_m1 = (lambda t, omega: likelihood(omega, t, 1))
        omega_est, uncertainty = curve_fit(
            prob_excited, self.chooser.ts, p_est,
            p0=[140000], method='lm'
        )
        if False:
            plt.plot(self.chooser.ts, p_est)
            print(omega_est)
            ts_smooth = np.linspace(0, self.chooser.ts[-1], 250)
            plt.plot(ts_smooth, prob_m1(ts_smooth, 140000))
            plt.plot(ts_smooth, prob_m1(ts_smooth, omega_est))
            plt.show()
        return omega_est
    def mean_log_v1(self):
        return np.nan # this type of estimator does not care about v1
    def many_measure(self, omega_list):
        length = len(omega_list)
        t_hist = []
        for i in range(length):
            t_idx = self.chooser.i
            t = self.chooser.get_t(self.dist)
            t_hist.append(t)
            m = measure(omega_list[i], t)
            if m == 1:
                self.m1s[t_idx] += 1
            self.counts[t_idx] += 1
        return t_hist


def main():
    omegas = np.linspace(omega_min, omega_max, 100)
    #omega_prior = normalize(1. + 0.*omegas) # uniform prior
    omega_prior = normalize(np.exp(-1e-7 * (omegas-140000)**2)) # normal prior
    ts = np.array([0.00008, 0.0001, 0.00013, 0.0002])
    
    fit_shots = [12, 20, 32, 60, 100, 200, 300, 600, 1000, 2000, 3000, 6000, 10000, 20000, 30000, 60000, 100000, 300000]
    
    def get_v1(x, r):
        return np.exp(0.)
    def get_omega_list(x, r, v1):
        return sample_omega_list(omegas, omega_prior, v1, x)
    def get_estimator(x, r, v1):
        return FittingEstimator(ts, np.ones(4, dtype=np.int64)*x//4)
    
    sim = Simulator(get_v1, get_omega_list, get_estimator)
    data = sim.x_trace(200, fit_shots, 'fit_shots')
    data['omegas'], data['omega_prior'] = omegas, omega_prior
    data['v1s'], data['v1_prior'] = None, None
    data['ts'] = ts
    save_data(data, get_filepath(data['plottype']))


if __name__ == '__main__':
    main()


