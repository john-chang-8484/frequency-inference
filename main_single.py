from estimators import *
import matplotlib.pyplot as plt



def main():
    omegas = np.linspace(omega_min, omega_max, 600)
    #omega_prior = normalize(1. + 0.*omegas) # uniform prior
    omega_prior = normalize(np.exp(-160.*(omegas-1)**2)) # normal prior
    log_v1s = np.linspace(-12., -8., 20)
    v1s = np.exp(log_v1s)
    v1_prior = normalize(1. + 0.*v1s)
    prior = np.outer(omega_prior, v1_prior)
    
    v1 = 0.00005 # [1/s^2/u] (u is the time between measurements)
    omega_list = sample_omega_list(omegas, omega_prior, v1, 500)
    estimator = Estimator(GridDist2D(omegas, v1s, prior), RandomChooser())
    estimator.many_measure(omega_list)
    
    print(estimator.dist.mean_omega(), omega_list[-1])
    
    fig = plt.figure()
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122, sharey=ax1)
    
    
    ax2.imshow(estimator.dist.dist, cmap=plt.get_cmap('inferno'),
        interpolation='nearest', aspect='auto',
        extent=[np.log(estimator.dist.v1s)[0, 0], np.log(estimator.dist.v1s)[0, -1],
                estimator.dist.omegas[-1, 0], estimator.dist.omegas[1, 0]] )
    ax2.plot([np.log(v1)], [omega_list[-1]], marker='o')
    ax1.plot(np.arange(len(omega_list)), omega_list)
    ax1.plot(0.5 * len(omega_list) * omega_prior / np.max(omega_prior), omegas)
    ax1.set_ylabel('$\Omega$')
    ax1.set_xlabel('measurement number')
    ax2.set_xlabel('v_1')
    plt.show()

main()
