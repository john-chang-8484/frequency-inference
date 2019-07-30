import numpy as np
from scipy.fftpack import dct, idct
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sim import *



def perturb_omega(omega, v1):
    return clip_omega(omega + np.random.normal(0., np.sqrt(v1)))


def sample_omega_list(omegas, prior, v1, length):
    omega0 = sample_dist(omegas, prior)
    omega_list = [omega0]
    for i in range(1, length):
        omega_list.append(perturb_omega(omega_list[-1], v1))
    return omega_list


# RULE: all fn calls should preserve normalization 
class ParticleDist:
    def normalize(self):
        self.dist = normalize(self.dist)
    def mean_omega(self):
        return np.sum(self.dist * self.omegas)
    def mean_log_v1(self):
        return np.sum(self.dist * np.log(self.v1s))
    def many_update(self, ts, ms):
        for t, m in zip(ts, ms):
            self.wait_u()
            self.update(t, m)
    def get_name(self):
        return self.name


class GridDist(ParticleDist):
    name = 'grid_dist'
    def __init__(self, omegas, v1s, prior):
        assert omegas.shape + v1s.shape == prior.shape
        self.shape = prior.shape
        self.omegas = np.copy(omegas).reshape((omegas.size, 1))
        self.v1s = np.copy(v1s).reshape((1, v1s.size))
        self.dist = np.copy(prior)
    def wait_u(self):
        ''' given a posterior distribution for omega at time t,
            we find the dist for omega at time t+u '''
        diff = self.omegas[-1] - self.omegas[0]
        fact = ((self.v1s * np.pi**2) / (2. * diff**2))
        cos_coeffs = dct(self.dist, axis=0) # switch to fourier space, in terms of cosines to get Neumann BC
        n = np.outer(np.arange(self.shape[0]), np.ones(self.shape[1]))
        cos_coeffs *= np.exp( - fact * n**2 ) # heat eq update
        self.dist = idct(cos_coeffs, axis=0) / (2 * self.shape[0]) # switch back to the usual representation
    def update(self, t, m):
        self.dist *= get_likelihood(self.omegas, t, m)
        self.normalize()


# TODO: this class is unfinished
class KalmanSwarm(ParticleDist):
    def __init__(self, omegas, v1s, prior):
        assert omegas.shape + v1s.shape == prior.shape
        self.shape = prior.shape
        self.omegas = np.repeat(np.copy(omegas).reshape((omegas.size, 1)), v1s.size, axis=1).flatten()
        self.v1s = np.repeat(np.copy(v1s).reshape((1, v1s.size)), omegas.size, axis=0).flatten()
        self.dist = np.copy(prior).flatten()
        self.omega_vars = np.ones_like(self.dist) * ((omegas[-1] - omegas[0]) / omegas.size)**2
        self.v1_vars = np.ones_like(self.dist) * ((v1s[-1] - v1s[0]) / omegas.size)*s*2
    def wait_u(self):
        drift = np.random.normal(0., np.sqrt(self.v1s))
        self.omegas += drift
        self.omega_vars += self.v1s
    def update(self, t, m):
        pass


def get_measurements(v1s, v1_prior, omegas, omega_prior, get_ts, get_v1):
    ts = get_ts()
    
    v1_true = get_v1(v1s, v1_prior)
    omega_list_true = sample_omega_list(omegas, omega_prior, v1_true, len(ts))
    ms = many_measure(omega_list_true, ts)
    
    return v1_true, omega_list_true, ts, ms


def do_run(v1s, v1_prior, omegas, omega_prior, get_ts, get_v1, mk_est):
    estimator = mk_est(omegas, v1s, np.outer(omega_prior, v1_prior))
    v1_true, omega_list_true, ts, ms = get_measurements(v1s, v1_prior, omegas,
        omega_prior, get_ts, get_v1)
    estimator.many_update(ts, ms)
    loss_omega = (omega_list_true[-1] - estimator.mean_omega())**2 / v1_true    # normalize to get rid of scaling issues
    loss_v1 = (np.log(v1_true) - estimator.mean_log_v1())**2                    # log to get rid of scaling issues
    return loss_omega, loss_v1


def do_runs(v1s, v1_prior, omegas, omega_prior, get_ts, get_v1, mk_est, n_runs):
    loss_omegas, loss_v1s = np.zeros(n_runs), np.zeros(n_runs)
    for r in range(0, n_runs):
        loss_omegas[r], loss_v1s[r] = do_run(v1s, v1_prior, omegas, omega_prior,
            get_ts, get_v1, mk_est)
    return loss_omegas, loss_v1s


def x_trace(v1s, v1_prior, omegas, omega_prior, get_get_ts, get_get_v1, est_class,
n_runs, x_list, x_list_nm):
    loss_omegas = np.zeros((len(x_list), n_runs))
    loss_v1s    = np.zeros((len(x_list), n_runs))
    for i, x in enumerate(x_list):
        print(i, '\t', x)
        loss_omegas[i], loss_v1s[i] = do_runs(v1s, v1_prior, omegas,
            omega_prior, get_get_ts(x), get_get_v1(x), est_class, n_runs)
    data = {
        'omega_min': omega_min,
        'omega_max': omega_max,
        'v_0': v_0,
        'omegas': omegas,
        'omega_prior': omega_prior,
        'x_list_nm': x_list_nm,
        'x_list': x_list,
        'estimator_name': est_class.name,
        'get_get_strat': inspect.getsource(get_get_ts),
        'get_get_v1': inspect.getsource(get_get_v1),
        'loss_omegas': loss_omegas,
        'loss_v1s': loss_v1s,
        'plottype': 'est_var_omega_%s' % x_list_nm,
        'estimator_params': get_numeric_class_vars(est_class),
    }
    save_data(data, get_filepath(data['plottype']))
    ####
    '''avg_loss_omegas = np.mean(loss_omegas, axis=1)
    avg_loss_v1s = np.mean(loss_v1s, axis=1)
    plt.plot(x_list, avg_loss_omegas)
    plt.plot(x_list, avg_loss_v1s)
    plt.yscale('log')
    plt.xscale('log')
    plt.show()'''


def main():
    log_v1s = np.linspace(-12., -3., 63)
    v1s = np.exp(log_v1s)
    v1_prior = normalize(1. + 0.*v1s)
    omegas = np.linspace(omega_min, omega_max, 80)
    omega_prior = normalize(1. + 0.*omegas)
    
    whichthing = 0
    
    if whichthing == 1:
        def get_get_ts(nothing):
            def get_ts():
                return np.random.uniform(0., 4.*np.pi, 60)
            return get_ts
        def get_get_v1(x):
            def get_v1(v1s, v1_prior):
                return x
            return get_v1
        x_trace(v1s, v1_prior, omegas, omega_prior, get_get_ts, get_get_v1, GridDist, 100, v1s, 'v1_true')
    
    
    if whichthing == 0:
        def get_ts():
            return np.random.uniform(0., 4.*np.pi, 100)
        def get_v1(v1s, prior):
            return sample_dist(v1s, v1_prior)
        
        v1_true, omega_list_true, ts, ms = get_measurements(v1s, v1_prior, omegas,
            omega_prior, get_ts, get_v1)
        
        grid = GridDist(omegas, v1s, np.outer(omega_prior, v1_prior))
        grid.many_update(ts, ms)
        
        print(grid.dist[grid.dist<-0.001])
        print(grid.mean_omega(), omega_list_true[-1])
        print(np.exp(grid.mean_log_v1()), v1_true)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(log_v1s, omegas)
        ax.plot_surface(X, Y, grid.dist, cmap=plt.get_cmap('copper'))
        plt.show()




if __name__ == '__main__':
    main()

