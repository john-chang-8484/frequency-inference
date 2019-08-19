from estimators import *
import matplotlib.pyplot as plt
from sys import argv


def main():
    np.random.seed(int(argv[1]))
    omegas = np.linspace(omega_min, omega_max, 100)
    #omega_prior = normalize(1. + 0.*omegas) # uniform prior
    omega_prior = normalize(np.exp(-1e-8 * (omegas-mu_omega)**2)) # normal prior
    log_v1s = np.linspace(0., 20., 20)
    v1s = np.exp(log_v1s)
    v1_prior = normalize(1. + 0.*v1s)
    prior = np.outer(omega_prior, v1_prior)
    num_particles = prior.size
    
    v1 = np.exp(5)#v1s[0] # [1/s^2/u] (u is the time between measurements)
    t_u_list = np.arange(0, 2000)
    omega_list = sample_omega_list(omegas, omega_prior, v1, t_u_list)
    grid = Estimator(GridDist2D(omegas, v1s, prior), OptimizingChooser(10, 10))
    dynm = Estimator(DynamicDist2D(omegas, v1s, prior, num_particles), OptimizingChooser(10, 10))
    qinfer = Estimator(QinferDist2D(omegas, v1s, prior, num_particles), OptimizingChooser(10, 10))
    
    th_grid = grid.many_measure(omega_list, t_u_list)
    th_dynm = dynm.many_measure(omega_list, t_u_list)
    th_qinfer = qinfer.many_measure(omega_list, t_u_list)
    if False:
        plt.plot(th_grid); plt.show()
        plt.plot(th_dynm); plt.show()
        #plt.plot(th_qinfer); plt.show()
    
    print(grid.dist.mean_omega(), dynm.dist.mean_omega())#, qinfer.dist.mean_omega())
    print('true: ', omega_list[-1])
    print('loss: ', (grid.dist.mean_omega() - omega_list[-1])**2)
    
    fig = plt.figure()
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122, sharey=ax1)
    
    hlfd_omega = (grid.dist.omegas[1, 0] - grid.dist.omegas[0, 0]) / 2
    hlfd_logv1 = (np.log(grid.dist.v1s[0, 1]) - np.log(grid.dist.v1s[0, 0])) / 2
    ax2.imshow(grid.dist.dist, cmap=plt.get_cmap('Blues'),
        interpolation='nearest', aspect='auto',
        extent=[np.log(grid.dist.v1s)[0, 0] - hlfd_logv1, np.log(grid.dist.v1s)[0, -1] + hlfd_logv1,
                grid.dist.omegas[-1, 0] + hlfd_omega, grid.dist.omegas[0, 0] - hlfd_omega] )
    ax2.plot([np.log(v1)], [omega_list[-1]], marker='o', color='black')
    ax2.plot([grid.dist.mean_log_v1()], [grid.dist.mean_omega()], marker='o', color='tab:blue', mew=1, mec='black')
    ax2.scatter(dynm.dist.vals[1], dynm.dist.vals[0], marker='o', color='tab:orange', s=1)
    ax2.plot([dynm.dist.mean_log_v1()], [dynm.dist.mean_omega()], marker='o', color='tab:orange', mew=1, mec='black')
    ax2.scatter(np.log(qinfer.dist.qinfer_updater.particle_locations[:, 1]),
        qinfer.dist.qinfer_updater.particle_locations[:, 0], color='tab:green', s=1)
    ax2.plot([qinfer.dist.mean_log_v1()], [qinfer.dist.mean_omega()], marker='o', color='tab:green', mew=1, mec='black')
    
    ax1.plot(t_u_list, omega_list)
    ax1.plot(0.5 * t_u_list[-1] * omega_prior / np.max(omega_prior), omegas)
    ax1.set_ylabel('$\Omega$')
    ax1.set_xlabel('t_us')
    ax2.set_xlabel('v_1')
    plt.show()

main()
