from estimators import *
import matplotlib.pyplot as plt



def main():
    omegas = np.linspace(omega_min, omega_max, 300)
    #omega_prior = normalize(1. + 0.*omegas) # uniform prior
    omega_prior = normalize(np.exp(-160.*(omegas-1)**2)) # normal prior
    log_v1s = np.linspace(-20., -8., 20)
    v1s = np.exp(log_v1s)
    v1_prior = normalize(1. + 0.*v1s)
    prior = np.outer(omega_prior, v1_prior)
    num_particles = prior.size
    
    v1 = 0.00000001 # [1/s^2/u] (u is the time between measurements)
    omega_list = sample_omega_list(omegas, omega_prior, v1, 10000)
    grid = Estimator(GridDist2D(omegas, v1s, prior), OptimizingChooser(10, 10))
    dynm = Estimator(DynamicDist2D(omegas, v1s, prior, num_particles), OptimizingChooser(10, 10))
    qinfer = Estimator(QinferDist2D(omegas, v1s, prior, num_particles), OptimizingChooser(10, 10))
    
    th_grid = grid.many_measure(omega_list)
    th_dynm = dynm.many_measure(omega_list)
    th_qinfer = qinfer.many_measure(omega_list)
    if False:
        plt.plot(th_grid); plt.show()
        plt.plot(th_dynm); plt.show()
        plt.plot(th_qinfer); plt.show()
    
    print(grid.dist.mean_omega(), dynm.dist.mean_omega(), qinfer.dist.mean_omega())
    print('true: ', omega_list[-1])
    
    fig = plt.figure()
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122, sharey=ax1)
    
    
    ax2.imshow(grid.dist.dist, cmap=plt.get_cmap('inferno'),
        interpolation='nearest', aspect='auto',
        extent=[np.log(grid.dist.v1s)[0, 0], np.log(grid.dist.v1s)[0, -1],
                grid.dist.omegas[-1, 0], grid.dist.omegas[1, 0]] )
    ax2.plot([np.log(v1)], [omega_list[-1]], marker='o')
    #ax2.scatter(dynm.dist.vals[1], dynm.dist.vals[0], marker='o', color='g')
    q_v1s, q_omegas, q_dist = qinfer.dist.qinfer_updater.posterior_mesh(1, 0, res1=100, res2=100, smoothing=0.02)
    #ax2.contour(np.log(q_v1s), q_omegas, q_dist, color='r')
    #ax2.scatter(np.log(qinfer.dist.qinfer_updater.particle_locations[:, 1]),
    #    qinfer.dist.qinfer_updater.particle_locations[:, 0], color='r')
    
    ax1.plot(np.arange(len(omega_list)), omega_list)
    ax1.plot(0.5 * len(omega_list) * omega_prior / np.max(omega_prior), omegas)
    ax1.set_ylabel('$\Omega$')
    ax1.set_xlabel('measurement number')
    ax2.set_xlabel('v_1')
    plt.show()

main()
