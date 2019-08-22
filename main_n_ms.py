from estimators import *



def main():
    omegas = np.linspace(omega_min, omega_max, 600)
    #omega_prior = normalize(1. + 0.*omegas) # uniform prior
    omega_prior = normalize(np.exp(-1e-8 * (omegas-mu_omega)**2)) # normal prior
    log_v1s = np.linspace(0., 15., 20)
    v1s = np.exp(log_v1s)
    v1_prior = normalize(1. + 0.*v1s)
    prior = np.outer(omega_prior, v1_prior)
    
    n_ms_list = [1, 2, 3, 6, 10, 20, 30, 60, 100, 200, 300, 600, 1000, 2000, 3000, 6000, 10000, 20000, 30000, 60000]
    
    def get_v1(x, r):
        return np.exp(5.)
    def get_omega_list(x, r, v1, t_u_list=None):
        random_seed(x, r)
        if t_u_list is None:
            t_u_list = np.arange(x)
        ans = sample_omega_list(omegas, omega_prior, v1, t_u_list)
        random_reseed()
        return ans
    
    def get_estimator0(x, r, v1):
        return Estimator(DynamicDist2D(omegas, v1s, prior, prior.size), OptimizingChooser(10, 10))
    def get_estimator1(x, r, v1):
        return Estimator(QinferDist2D(omegas, v1s, prior, prior.size), OptimizingChooser(10, 10))
    def get_estimator2(x, r, v1):
        return Estimator(GridDist2D(omegas, v1s, prior), OptimizingChooser(10, 10))
    def get_estimator3(x, r, v1):
        return Estimator(DynamicDist2D(omegas, v1s, prior, prior.size), RandomChooser())
    def get_estimator4(x, r, v1):
        return Estimator(QinferDist2D(omegas, v1s, prior, prior.size), RandomChooser())
    def get_estimator5(x, r, v1):
        return Estimator(GridDist2D(omegas, v1s, prior), RandomChooser())
    
    # 1d distributions:
    def get_estimator6(x, r, v1):
        return Estimator(GridDist1D(omegas, omega_prior, np.exp(5.)), RandomChooser())
    def get_estimator7(x, r, v1):
        return Estimator(DynamicDist1D(omegas, omega_prior, np.exp(5.), omega_prior.size), RandomChooser())
    def get_estimator8(x, r, v1):
        return Estimator(QinferDist1D(omegas, omega_prior, np.exp(5.), omega_prior.size), RandomChooser())
    
    for get_est in [get_estimator6, get_estimator7, get_estimator8]:#[get_estimator0, get_estimator1, get_estimator2, get_estimator3, get_estimator4, get_estimator5]:
        sim = Simulator(get_v1, get_omega_list, get_est)
        data = sim.x_trace(200, n_ms_list, 'n_ms')
        data['omegas'], data['omega_prior'] = omegas, omega_prior
        data['v1s'], data['v1_prior'] = v1s, v1_prior
        save_data(data, get_filepath(data['plottype']))


if __name__ == '__main__':
    main()


