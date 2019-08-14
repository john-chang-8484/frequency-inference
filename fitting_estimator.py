import numpy as np
from estimators import *
from scipy.optimize import curve_fit
from util import Bunch, idprn
import matplotlib.pyplot as plt


# constants:
true_omega_mean = 140000


class FittingChooser(TimeChooser):
    """ cycle through a specific sequence of ts """
    name = 'fitting'
    def __init__(self, ts):
        self.ts = ts
        self.i = 0
    def get_t(self, dist):
        ans = self.ts[self.i]
        self.i = (self.i + 1) % len(self.ts)
        return ans
        


class FittingEstimator(Estimator):
    name = 'fitting'
    def __init__(self, ts):
        self.chooser = FittingChooser(ts)
        self.dist = Bunch({'name': 'none'})
        self.m1s =    np.array([0 for t in ts]) # number of shots where we got m = 1
        self.counts = np.array([0 for t in ts]) # total number of shots
    def mean_omega(self): # note: really this is just an *estimation* of omega, not the expected value
        p_est = self.m1s / self.counts
        prob_m1 = (lambda t, omega: likelihood(omega, t, 1))
        omega_est, uncertainty = curve_fit(
            prob_excited, self.chooser.ts, p_est,
            p0=[true_omega_mean], method='lm'
        )
        '''plt.plot(self.chooser.ts, p_est)
        print(omega_est)
        ts_smooth = np.linspace(0, self.chooser.ts[-1], 250)
        plt.plot(ts_smooth, prob_m1(ts_smooth, true_omega_mean))
        plt.plot(ts_smooth, prob_m1(ts_smooth, omega_est))
        plt.show()'''
        return omega_est[0]
    def mean_log_v1(self):
        return np.nan # this type of estimator does not care about v1
    def many_measure(self, omega_list):
        length = len(omega_list)
        t_hist = []
        for j in range(length):
            t_idx = self.chooser.i
            t = self.chooser.get_t(self.dist)
            t_hist.append(t)
            m = measure(omega_list[j], t)
            if m == 1:
                self.m1s[t_idx] += 1
            self.counts[t_idx] += 1
        return t_hist


class ChunkFittingEstimator(FittingEstimator):
    name = 'chunk_fitting'
    def __init__(self, ts, chunksize):
        self.chooser = FittingChooser(ts)
        self.dist = Bunch({'name': 'none'})
        self.m1s =    np.array([0 for t in ts]) # number of shots where we got m = 1
        self.counts = np.array([0 for t in ts]) # total number of shots
        self.chunksize = chunksize
        self.omega_hist = []
    def mean_log_v1(self):
        self.chunk_if_needed()
        if len(self.omega_hist) < 3:
            return np.nan
        else:
            est_omegas = np.array(self.omega_hist)
            return np.log(np.cov(np.array(est_omegas[0:-1] - est_omegas[1:])) / self.chunksize)
    def mean_omega(self):
        if sum(self.counts) < self.chunksize / 2 and len(self.omega_hist) > 0:
            return self.omega_hist[-1]
        else:
            return super().mean_omega()
    def chunk_if_needed(self):
        """ upon reaching the chunk point, we
            reset our estimation of omega, and save old value
        """
        if sum(self.counts) >= self.chunksize:
            self.omega_hist.append(self.mean_omega())
            self.m1s = np.zeros(len(self.m1s))
            self.counts = np.zeros(len(self.m1s))
    def many_measure(self, omega_list):
        length = len(omega_list)
        t_hist = []
        for j in range(length):
            self.chunk_if_needed()
            t_idx = self.chooser.i
            t = self.chooser.get_t(self.dist)
            t_hist.append(t)
            m = measure(omega_list[j], t)
            if m == 1:
                self.m1s[t_idx] += 1
            self.counts[t_idx] += 1
        return t_hist


def main():
    omegas = np.linspace(omega_min, omega_max, 100)
    #omega_prior = normalize(1. + 0.*omegas) # uniform prior
    omega_prior = normalize(np.exp(-1e-7 * (omegas-true_omega_mean)**2)) # normal prior
    ts = np.array([0.00008, 0.0001, 0.00013, 0.0002])
    
    fit_shots = [3000, 4000, 6000, 8000, 10000, 20000, 30000, 60000, 100000, 300000]
    
    def get_v1(x, r):
        return np.exp(10.)
    def get_omega_list(x, r, v1):
        return sample_omega_list(omegas, omega_prior, v1, x)
    def get_estimator(x, r, v1):
        return ChunkFittingEstimator(ts, 1000)
    
    sim = Simulator(get_v1, get_omega_list, get_estimator)
    data = sim.x_trace(200, fit_shots, 'fit_shots')
    data['omegas'], data['omega_prior'] = omegas, omega_prior
    data['v1s'], data['v1_prior'] = None, None
    data['ts'] = ts
    save_data(data, get_filepath(data['plottype']))


if __name__ == '__main__':
    main()


