from estimators import *



def main():
    omegas = np.linspace(omega_min, omega_max, 600)
    omega_prior = normalize(1. + 0.*omegas)
    log_v1s = np.linspace(-12., -8., 20)
    v1s = np.exp(log_v1s)
    v1_prior = normalize(1. + 0.*v1s)
    prior = np.outer(omega_prior, v1_prior)
    
    def get_v1(x, r):
        return x
    def get_omega_list(x, r, v1, t_u_list=None):
        random_seed(x, r)
        if t_u_list is None:
            t_u_list = np.arange(500)
        ans = sample_omega_list(omegas, omega_prior, v1, t_u_list)
        random_reseed()
        return ans
    def get_estimator0(x, r, v1):
        return Estimator(QinferDist2D(omegas, v1s, prior, prior.size), RandomChooser())
    def get_estimator1(x, r, v1):
        return Estimator(GridDist2D(omegas, v1s, prior), RandomChooser())
    
    for get_est in [get_estimator0, get_estimator1]:
        sim = Simulator(get_v1, get_omega_list, get_est)
        data = sim.x_trace(500, v1s, 'v1_true')
        data['omegas'], data['omega_prior'] = omegas, omega_prior
        data['v1s'], data['v1_prior'] = v1s, v1_prior
        save_data(data, get_filepath(data['plottype']))


if __name__ == '__main__':
    main()


