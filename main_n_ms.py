from estimators import *



def main():
    omegas = np.linspace(omega_min, omega_max, 600)
    omega_prior = normalize(1. + 0.*omegas)
    log_v1s = np.linspace(-12., -8., 20)
    v1s = np.exp(log_v1s)
    v1_prior = normalize(1. + 0.*v1s)
    prior = np.outer(omega_prior, v1_prior)
    
    n_ms_list = [1, 2, 3, 6, 10, 20, 30, 60, 100, 200, 300, 600, 1000, 2000]
    
    def get_v1(x, r):
        return 0.00001
    def get_omega_list(x, r, v1):
        random_seed(x, r)
        return sample_omega_list(omegas, omega_prior, v1, x)
    """
    def get_estimator0(x, r, v1):
        return Estimator(QinferDist1D(omegas, prior, v1, len(prior)), RandomChooser())
    def get_estimator1(x, r, v1):
        return Estimator(GridDist1D(omegas, prior, v1), RandomChooser())
    def get_estimator2(x, r, v1):
        return Estimator(DynamicDist1D(omegas, prior, v1, len(prior)), RandomChooser())
    """
    
    def get_estimator0(x, r, v1):
        return Estimator(QinferDist2D(omegas, v1s, prior, prior.size), RandomChooser())
    def get_estimator1(x, r, v1):
        return Estimator(GridDist2D(omegas, v1s, prior), RandomChooser())
    
    for get_est in [get_estimator0, get_estimator1]:
        sim = Simulator(get_v1, get_omega_list, get_est)
        data = sim.x_trace(500, n_ms_list, 'n_ms')
        data['omegas'], data['prior'] = omegas, prior
        save_data(data, get_filepath(data['plottype']))


if __name__ == '__main__':
    main()


