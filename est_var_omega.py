import numpy as np
from scipy.fftpack import dct, idct
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sim import *

from qinfer import FiniteOutcomeModel



def perturb_omega(omega, v1):
    return clip_omega(omega + np.random.normal(0., np.sqrt(v1)))


def sample_omega_list(omegas, prior, v1, length):
    omega0 = sample_dist(omegas, prior)
    omega_list = [omega0]
    for i in range(1, length):
        omega_list.append(perturb_omega(omega_list[-1], v1))
    return omega_list


# RULE: all fn calls should preserve normalization 
class ParticleDist2D(ParticleDist):
    def normalize(self):
        self.dist = normalize(self.dist)
    def mean_omega(self):
        return np.sum(self.dist * self.omegas)
    def mean_log_v1(self):
        return np.sum(self.dist * np.log(self.v1s))
    def many_update(self, ts, ms):
        for t, m in zip(ts, ms):
            #print(t, m)
            self.wait_u()
            self.update(t, m)
    def get_name(self):
        return self.name
    def sample(self, n):
        ''' Take n samples of omega from this distribution. '''
        return np.random.choice(self.omegas.flatten(), 
            p=normalize(np.abs(np.sum(self.dist, axis=1))), size=n)


class GridDist2D(ParticleDist2D):
    name = 'grid_dist'
    size = None
    def __init__(self, omegas, v1s, prior):
        assert omegas.shape + v1s.shape == prior.shape
        self.size = prior.size
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


class DynamicDist2D(ParticleDist2D):
    name = 'dynamic_dist'
    size = None
    def __init__(self, omegas, v1s, prior):
        assert omegas.shape + v1s.shape == prior.shape
        new_omegas = np.outer(omegas, np.ones(v1s.size)).flatten()
        new_v1s = np.outer(np.ones(omegas.size), v1s).flatten()
        
        chosen_indices = deterministic_sample(self.size, prior.flatten())
        self.omegas = new_omegas[chosen_indices]
        self.v1s = new_v1s[chosen_indices]
        self.dist = np.ones(self.size) / self.size
        self.target_cov = self.cov() # initialize target covariance to actual covariance
    def cov(self):
        return np.cov(np.stack([self.omegas, self.v1s]), ddof=0, aweights=self.dist)
    # TODO: nontrivial adaptation from other implementation here


# http://docs.qinfer.org/en/latest/guide/timedep.html#specifying-custom-time-step-updates
# http://docs.qinfer.org/en/latest/guide/models.html
class DiffusivePrecessionModel2D(FiniteOutcomeModel):
    @property
    def n_modelparams(self):
        return 2
    @property
    def is_n_outcomes_constant(self):
        return True
    def n_outcomes(self, expparams):
        return 2
    def are_models_valid(self, modelparams):
        return np.logical_and(
            modelparams[:,0] >= omega_min, 
            modelparams[:,0] <= omega_max,
            modelparams[:,1] >= 0. )
    @property
    def expparams_dtype(self):
        return [('ts', 'float', 1)]
    def likelihood(self, outcomes, modelparams, expparams):
        super(DiffusivePrecessionModel2D, self).likelihood(outcomes, modelparams, expparams)
        return get_likelihood(modelparams[:,0], expparams, outcomes).reshape(1, modelparams.shape[0], 1)
    def update_timestep(self, modelparams, expparams):
        assert expparams.shape[0] == 1
        modelparams_new = np.copy(modelparams)
        modelparams_new[:,1] = np.clip(modelparams[:,1], 0., np.inf)
        steps = np.random.normal(0., np.sqrt(modelparams_new[:,1]), 
            size=modelparams.shape[0])
        modelparams_new[:,0] = clip_omega(modelparams[:,0] + steps)
        return modelparams_new.reshape(modelparams.shape + (1,))

class PriorSample2D(Distribution):
    n_rvs = 2
    def __init__(self, omegas, v1s, dist):
        self.omegas, self.v1s = np.meshgrid(omegas, v1s)
        self.values = np.stack([self.omegas, self.v1s], axis=-1)
        self.dist = dist
    def sample(self, n=1):
        sz = self.values.size // 2
        cat0 = np.concatenate((self.values[0:1,:], self.values, self.values[-1:,:]), axis=0)
        cat1 = np.concatenate((cat0[:,0:1], cat0, cat0[:,-1:]), axis=1)
        choice = np.random.choice(sz, p=self.dist.flatten(), size=n)
        ind0, ind1 = np.unravel_index(choice, self.dist.shape)
        upper = (cat1[ind1 + 1, ind0 + 1] + cat1[ind1 + 2, ind0 + 2]) / 2
        lower = (cat1[ind1 + 1, ind0 + 1] + cat1[ind1, ind0]) / 2
        return np.random.uniform(lower, upper)

# simple wrapper class for the qinfer implementation
class QinferDist2D(ParticleDist2D):
    a = 1.
    h = 0.005
    size = ...
    def __init__(self, omegas, v1s, prior):
        self.size = prior.size
        self.qinfer_model = DiffusivePrecessionModel2D()
        self.qinfer_prior = PriorSample2D(omegas, v1s, prior)
        self.qinfer_updater = qinfer.SMCUpdater( self.qinfer_model, self.size,
            self.qinfer_prior, resampler=LiuWestResampler(self.a, self.h, debug=False) )
    def many_update(self, ts, ms):
        for t, m in zip(ts, ms):
            self.qinfer_updater.update(np.array([m]), np.array([t]))
    def mean(self):
        return self.qinfer_updater.est_mean()
    def posterior_marginal(self, *args, **kwargs):
        return self.qinfer_updater.posterior_marginal(*args, **kwargs)
    def sample(self, n):
        ''' Take n samples of omega from this distribution. '''
        return self.qinfer_updater.particle_locations[np.random.choice(
            self.qinfer_updater.particle_locations.shape[0],
            p=self.qinfer_updater.particle_weights, size=n ), 0]
    def stddev_omega(self):
        return 1e-6


##                                                                            ##
################################################################################
##                                                                            ##


def do_run(v1s, v1_prior, omegas, omega_prior, get_ts, get_v1, mk_est):
    estimator = mk_est(omegas, v1s, np.outer(omega_prior, v1_prior))
    ts, length = get_ts(estimator)
    
    v1_true = get_v1(v1s, v1_prior)
    omega_list_true = sample_omega_list(omegas, omega_prior, v1_true, length)
    
    for omega, t in zip(omega_list_true, ts):
        estimator.many_update([t], [np.random.binomial(1, prob_excited(t, omega))])
    
    return estimator, v1_true, omega_list_true


def losses(estimator, v1_true, omega_list_true):
    loss_omega = (omega_list_true[-1] - estimator.mean_omega())**2
    loss_v1 = (np.log(v1_true) - estimator.mean_log_v1())**2                    # log to get rid of scaling issues
    return loss_omega, loss_v1


def do_runs(v1s, v1_prior, omegas, omega_prior, get_ts, get_v1, mk_est, n_runs):
    loss_omegas, loss_v1s = np.zeros(n_runs), np.zeros(n_runs)
    for r in range(0, n_runs):
        loss_omegas[r], loss_v1s[r] = losses( *do_run(v1s, v1_prior, omegas,
            omega_prior, get_ts, get_v1, mk_est) )
    return loss_omegas, loss_v1s


def x_trace(v1s, v1_prior, omegas, omega_prior, get_get_ts, get_get_v1,
est_class, n_runs, x_list, x_list_nm):
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
        'get_get_ts': inspect.getsource(get_get_ts),
        'get_get_v1': inspect.getsource(get_get_v1),
        'loss_omegas': loss_omegas,
        'loss_v1s': loss_v1s,
        'plottype': 'est_var_omega_%s' % x_list_nm,
        'estimator_params': get_numeric_class_vars(est_class),
    }
    save_data(data, get_filepath(data['plottype']))


def main():
    log_v1s = np.linspace(-12., -8., 20)
    v1s = np.exp(log_v1s)
    #v1s = np.array([0.])
    v1_prior = normalize(1. + 0.*v1s)
    omegas = np.linspace(omega_min, omega_max, 2000)
    omega_prior = normalize(1. + 0.*omegas)
    
    whichthing = 2
    
    if whichthing == 1:
        def get_get_ts(x):
            def get_ts(est):
                l = 200
                return np.random.uniform(0., 4.*np.pi, l), l
            return get_ts
        def get_get_v1(x):
            def get_v1(v1s, prior):
                return x
            return get_v1
        x_trace(v1s, v1_prior, omegas, omega_prior, get_get_ts, get_get_v1, GridDist2D, 500, [1e-6, 2e-6, 3e-6, 6e-6, 1e-5, 2e-5, 3e-5, 6e-5, 1e-4, 2e-4, 3e-4, 6e-4, 0.001], 'v1_true')
    
    if whichthing == 2:
        # np.random.uniform(0., ParticleDist.max_t, l), l
        def get_get_ts(x):
            def get_ts(est):
                l = x
                return (est.pick_t() for i in range(l)), l
            return get_ts
        def get_get_v1(x):
            def get_v1(v1s, prior):
                return 0.
            return get_v1
        x_trace(v1s, v1_prior, omegas, omega_prior, get_get_ts, get_get_v1, GridDist2D, 500, [3, 6, 10, 20, 30, 60, 100], 'n_measurements')

    
    if whichthing == 0:
        tlist = []
        def get_ts(est):
            l = 200
            return (np.random.uniform(0., 4.*np.pi) for i in range(l)), l
        def get_ts(est):
            l = 200
            return ((lambda x: (x, tlist.append(x))[0])(est.pick_t()) for i in range(l)), l
        def get_v1(v1s, prior):
            return sample_dist(v1s, v1_prior)
        
        if True:
            grid, v1_true, omega_list_true = do_run(v1s, v1_prior, omegas, omega_prior, get_ts, get_v1, GridDist2D)
            
            print(grid.dist[grid.dist<-0.001])
            print(grid.mean_omega(), omega_list_true[-1])
            print(np.exp(grid.mean_log_v1()), v1_true)
            
            if False:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                X, Y = np.meshgrid(log_v1s, omegas)
                ax.plot_surface(X, Y, grid.dist, cmap=plt.get_cmap('inferno'))
            elif True:
                plt.imshow(grid.dist, cmap=plt.get_cmap('inferno'),
                    interpolation='nearest', aspect='auto',
                    extent=[np.log(grid.v1s)[0, 0], np.log(grid.v1s)[0, -1],
                            grid.omegas[0, 0], grid.omegas[-1, 0]] )
            else:
                plt.plot(omegas, grid.dist.flatten()) # works when v_1 has dimension 1 only
            plt.show()
            plt.plot(tlist)
            plt.show()
        else:
            qinfer, v1_true, omega_list_true = do_run(v1s, v1_prior, omegas, omega_prior, get_ts, get_v1, QinferDist2D)
            print(omega_list_true[-1], v1_true)
            print(qinfer.mean())
            qinf_omegas, qinf_post = qinfer.posterior_marginal(idx_param=0, res=20)
            plt.plot(qinf_omegas, qinf_post)
            plt.show()
            




if __name__ == '__main__':
    main()

