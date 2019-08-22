from estimators import *

"""
For a given true value of v1, this program investigates how changing the nominal
value of v1 that the estimators are given affects estimator accuracy.
"""


def main():
    omegas = np.linspace(omega_min, omega_max, 600)
    omega_prior = normalize(1. + 0.*omegas)
    log_v1s = np.linspace(0., 20., 20)
    v1s = np.exp(log_v1s)
    v1_prior = normalize(1. + 0.*v1s)
    prior = np.outer(omega_prior, v1_prior)
    
    v1_true = np.exp(10)
    
    def get_v1(x, r):
        return v1_true # true v1 value is always the same
    def get_omega_list(x, r, v1, t_u_list=None):
        random_seed(x, r)
        if t_u_list is None:
            t_u_list = np.arange(2000)
        ans = sample_omega_list(omegas, omega_prior, v1, t_u_list)
        random_reseed()
        return ans
    def get_estimator0(x, r, v1):
        return Estimator(GridDist1D(omegas, omega_prior, x), RandomChooser())
    def get_estimator1(x, r, v1):
        return Estimator(DynamicDist1D(omegas, omega_prior, x, omegas.size), RandomChooser())
    def get_estimator2(x, r, v1):
        return Estimator(QinferDist1D(omegas, omega_prior, x, omegas.size), RandomChooser())
    
    for get_est in [get_estimator0, get_estimator1, get_estimator2]:
        sim = Simulator(get_v1, get_omega_list, get_est)
        data = sim.x_trace(200, v1s, 'nominal_v1')
        data['omegas'], data['omega_prior'] = omegas, omega_prior
        data['v1s'], data['v1_prior'] = v1s, v1_prior
        data['v1_true'] = v1_true
        data['mu_omega'] = np.sum(omega_prior * omegas)
        save_data(data, get_filepath(data['plottype']))


if __name__ == '__main__':
    main()


