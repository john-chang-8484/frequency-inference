from estimators import *

def main():
    omegas = np.linspace(omega_min, omega_max, 600)
    prior = normalize(1. + 0.*omegas)
    def get_v1(x, r):
        return 0.00001
    def get_omega_list(x, r, v1):
        random_seed(x, r)
        return sample_omega_list(omegas, prior, v1, x)
    
    m_ms_list = [1, 2, 3, 6, 10]
    
    # qinfer estimator
    def get_estimator(x, r, v1):
        return Estimator(QinferDist1D(omegas, prior, v1, len(prior)), RandomChooser())
    sim = Simulator(get_v1, get_omega_list, get_estimator)
    data = sim.x_trace(500, m_ms_list, 'n_ms')
    data['omegas'], data['prior'] = omegas, prior
    save_data(data, get_filepath(data['plottype']))
    
    # grid estimator
    def get_estimator(x, r, v1):
        return Estimator(GridDist1D(omegas, prior, v1), RandomChooser())
    sim = Simulator(get_v1, get_omega_list, get_estimator)
    data = sim.x_trace(500, m_ms_list, 'n_ms')
    data['omegas'], data['prior'] = omegas, prior
    save_data(data, get_filepath(data['plottype']))
    
    # dynamic estimator
    def get_estimator(x, r, v1):
        return Estimator(DynamicDist1D(omegas, prior, v1, len(prior)), RandomChooser())
    sim = Simulator(get_v1, get_omega_list, get_estimator)
    data = sim.x_trace(500, m_ms_list, 'n_ms')
    data['omegas'], data['prior'] = omegas, prior
    save_data(data, get_filepath(data['plottype']))


if __name__ == '__main__':
    main()


