from estimators import *


class ConstantChooser(TimeChooser):
    name = 'constant'
    def __init__(self, t):
        self.t = t
    def get_t(self, dist):
        return self.t


def main():
    omegas = np.linspace(omega_min, omega_max, 100)
    #omega_prior = normalize(1. + 0.*omegas) # uniform prior
    omega_prior = normalize(np.exp(-9e-7 * (omegas-140000)**2)) # normal prior
    log_v1s = np.linspace(-10., 15., 1)
    v1s = np.exp(log_v1s)
    v1_prior = normalize(1. + 0.*v1s)
    prior = np.outer(omega_prior, v1_prior)
    
    t_ms_list = np.linspace(0., 3*t_max, 200)
    
    def get_v1(x, r):
        return np.exp(-10)
    def get_omega_list(x, r, v1):
        random_seed(x, r)
        ans = sample_omega_list(omegas, omega_prior, v1, 1000)
        random_reseed()
        return ans
    
    def get_estimator0(x, r, v1):
        return Estimator(GridDist1D(omegas, omega_prior, np.exp(-10)), ConstantChooser(x))
    def get_estimator1(x, r, v1):
        return Estimator(QinferDist2D(omegas, v1s, prior, prior.size), ConstantChooser(x))
    def get_estimator2(x, r, v1):
        return Estimator(GridDist2D(omegas, v1s, prior), ConstantChooser(x))
    
    for get_est in [get_estimator0, get_estimator1, get_estimator2]:
        sim = Simulator(get_v1, get_omega_list, get_est)
        data = sim.x_trace(200, t_ms_list, 't_ms')
        data['omegas'], data['omega_prior'] = omegas, omega_prior
        data['v1s'], data['v1_prior'] = v1s, v1_prior
        save_data(data, get_filepath(data['plottype']))


if __name__ == '__main__':
    main()


