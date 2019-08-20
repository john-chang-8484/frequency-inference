from estimators import *
import matplotlib.pyplot as plt
from sys import argv
from plot_util import pin_plot


def main():
    np.random.seed(int(argv[1]))
    omegas = np.linspace(omega_min, omega_max, 200)
    omega_prior = normalize(np.exp(-1e-8 * (omegas-mu_omega)**2)) # normal prior
    v1 = np.exp(10) # [1/s^2/u] (u is the time between measurements)
    num_particles = omega_prior.size
    
    t_u_list = np.arange(0, 200)
    omega_list = sample_omega_list(omegas, omega_prior, v1, t_u_list)
    grid = Estimator(GridDist1D(omegas, omega_prior, v1), OptimizingChooser(10, 10))
    dynm = Estimator(DynamicDist1D(omegas, omega_prior, v1, num_particles), OptimizingChooser(10, 10))
    qinfer = Estimator(QinferDist1D(omegas, omega_prior, v1, num_particles), OptimizingChooser(10, 10))
    
    th_grid = grid.many_measure(omega_list, t_u_list)
    th_dynm = dynm.many_measure(omega_list, t_u_list)
    th_qinfer = qinfer.many_measure(omega_list, t_u_list)
    
    print(grid.dist.mean_omega(), dynm.dist.mean_omega(), qinfer.dist.mean_omega())
    print('true: ', omega_list[-1])
    print('loss: ', (grid.dist.mean_omega() - omega_list[-1])**2)
    
    fig = plt.figure()
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212, sharex=ax1)
    
    ax2_top = np.max(grid.dist.dist) * 1.3
    
    ax1.plot(omega_list, t_u_list, label='true history of $\Omega$')
    ax1.plot(omegas, 0.5 * t_u_list[-1] * omega_prior / np.max(omega_prior), label='omega prior')
    
    pin_plot(ax2, omegas, grid.dist.dist, label='grid', color='tab:blue')
    pin_plot(ax2, [grid.mean_omega()], [ax2_top], linestyle='--', color='tab:blue')
    
    pin_plot( ax2,
        dynm.dist.omegas,
        dynm.dist.dist,
        label = 'dynamic', color='tab:orange' )
    pin_plot(ax2, [dynm.mean_omega()], [ax2_top], linestyle='--', color='tab:orange')
    
    pin_plot( ax2,
        qinfer.dist.qinfer_updater.particle_locations.flatten(),
        qinfer.dist.qinfer_updater.particle_weights,
        label = 'qinfer', color='tab:green' )
    pin_plot(ax2, [qinfer.mean_omega()], [ax2_top], linestyle='--', color='tab:green')
    
    pin_plot(ax1, [omega_list[-1]], [t_u_list[-1]], label='true final value of $\Omega$', color='black')
    pin_plot(ax2, [omega_list[-1]], [ax2_top], label='true final value of $\Omega$', color='black')
    
    ax1.set_ylabel('$t  [u]$')
    ax2.set_ylabel('posterior distribution')
    ax2.set_xlabel('$\Omega$')
    
    ax1.legend()
    ax2.legend()
    plt.show()

main()
