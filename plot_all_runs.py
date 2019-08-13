import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sys import argv
from util import load_data, Bunch



def plot(b):
    colourmap = plt.get_cmap('jet')
    colours = [colourmap(k) for k in np.linspace(0., 1., len(b.x_list))]
    things = b.loss_omegas
    bins = np.linspace(np.log(np.min(things)), np.log(np.max(things)), 30)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for j in range(0, len(b.x_list), 4):
        histogram, _ = np.histogram(np.log(things[j]), bins=bins)
        ax.plot(bins[:-1], histogram, zs=b.x_list[j], zdir='y', color=colours[j])
        ax.plot(np.log(np.mean(things, axis=1)), 0*b.x_list, zs=b.x_list, zdir='y')
    plt.show()
    

def main():
    for filename in argv[1:]:
        plot(Bunch(load_data(filename)))


if __name__ == '__main__':
    main()


