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
