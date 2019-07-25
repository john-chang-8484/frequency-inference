import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sys import argv
from util import load_data, Bunch



def plot(b):
    n_estimators = b.run_hists.shape[0]
    log_losses = np.linspace(-7., 1., 50)
    losses = np.exp(np.log(10) * log_losses)
    colourmap = plt.get_cmap('jet')
    colours = [colourmap(k) for k in np.linspace(0., 1., len(b.N_list))]
    for i, run_hist, nm in zip(range(n_estimators), b.run_hists, b.estimator_names):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        print('now plotting:', nm)
        for j, (runs, N) in enumerate(zip(run_hist, b.N_list)):
            histogram, _ = np.histogram(runs, bins=losses)
            ax.bar(log_losses[:-1], histogram, zs=np.log(N)/np.log(10), zdir='y', color=colours[j])
        plt.show()


def main():
    for filename in argv[1:]:
        plot(Bunch(load_data(filename)))


if __name__ == '__main__':
    main()


