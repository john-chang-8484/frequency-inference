import numpy as np
import matplotlib.pyplot as plt

def pin_plot(pts, wts):
    pts = np.array(pts)
    wts = np.array(wts)
    x1 = pts
    x2 = pts
    x3 = np.zeros(pts.size) * np.nan
    y1 = wts
    y2 = 0. * wts
    y3 = x3
    x = np.stack((x1, x2, x3), axis=-1).flatten()
    y = np.stack((y1, y2, y3), axis=-1).flatten()
    plt.plot(x, y)


def pin_plot_3d(ax, ptsx, ptsy, wts, **kwargs):
    ptsx = np.array(ptsx)
    ptsy = np.array(ptsy)
    wts = np.array(wts)
    x1 = ptsx
    x2 = ptsx
    x3 = np.zeros(ptsx.size) * np.nan
    y1 = ptsy
    y2 = ptsy
    y3 = np.zeros(ptsx.size) * np.nan
    z1 = wts
    z2 = 0. * wts
    z3 = x3
    x = np.stack((x1, x2, x3), axis=-1).flatten()
    y = np.stack((y1, y2, y3), axis=-1).flatten()
    z = np.stack((z1, z2, z3), axis=-1).flatten()
    ax.plot(x, y, zs=z, **kwargs)

