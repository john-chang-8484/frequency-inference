from sim import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plot_util import pin_plot_3d


dim3 = False


def main():
    omegas = np.linspace(omega_min, omega_max, NUM_PARTICLES)
    prior = normalize(1. + 0.*omegas)
    
    ts = np.random.uniform(0., 4.*np.pi, 1000)
    ns = [1] * 1000
    
    omega_list_true = 1. + 0.5*np.sign(np.linspace(-10., 15., len(ts)))#sample_omega_list(omegas, prior, len(ts))
    ms = many_measure(omega_list_true, ts, ns)
    
    dynm = DynamicDist(omegas, prior)
    grid = GridDist(omegas, prior)
    
    particle_lists = [np.copy(dynm.omegas)]
    weight_lists = [np.copy(dynm.dist)]
    posts = [np.copy(grid.dist)]
    
    grid_means = []
    dynm_means = []
    
    for t, n, m in zip(ts, ns, ms):
        grid.wait_u()
        grid.update(t, n, m)
        dynm.wait_u()
        dynm.update(t, n, m)
        
        particle_lists.append(np.copy(dynm.omegas))
        weight_lists.append(np.copy(dynm.dist))
        posts.append(np.copy(grid.dist))
        
        grid_means.append(grid.mean())
        dynm_means.append(dynm.mean())
    
    us = np.arange(0, len(posts))
    
    particle_us = us.reshape(len(posts), 1).repeat(ParticleDist.size, 1)
    particle_lists = np.stack(particle_lists, axis=0)
    weight_lists = np.stack(weight_lists, axis=0)
    
    x_grid, y_grid = np.meshgrid(us, omegas)
    posts = np.stack(posts, axis=-1)

    if dim3:
        # plotting:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        pin_plot_3d(ax, particle_us.flatten(), particle_lists.flatten(), weight_lists.flatten(), alpha=0.55, color='tab:orange')
        ax.plot_wireframe(x_grid, y_grid, posts, alpha=0.4, color='tab:blue')
        
        ax.plot(us[1:], omega_list_true, color='g')
        ax.plot(us[1:], dynm_means, color='tab:orange')
        ax.plot(us[1:], grid_means, color='tab:blue')

        plt.show()
    else:
        plt.plot(us[1:], omega_list_true, color='g', label='true omega')
        plt.plot(us[1:], dynm_means, color='tab:orange', label='dynamic_estimator')
        plt.plot(us[1:], grid_means, color='tab:blue', label='grid_estimator')

        plt.legend()
        plt.xlabel('time (in number of measurements)')
        plt.ylabel('omega')

        plt.show()


if __name__ == '__main__':
    main()


