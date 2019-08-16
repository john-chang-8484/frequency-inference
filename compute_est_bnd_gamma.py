import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from estimators import normalize, likelihood, t_max


omega = np.linspace(-10, 10, 1000).reshape(1000, 1, 1)
normal_dist = normalize(np.exp(-0.5 * omega**2))

t = np.linspace(0.001, t_max - 0.001, 200).reshape(1, 200, 1)
off = np.linspace(0, 5.5, 150).reshape(1, 1, 150)

dist_0 = normal_dist * likelihood(omega + off, t, 0)
dist_0 /= np.sum(dist_0, axis=0)
var0 = np.sum(omega**2 * dist_0, axis=0) - np.sum(omega * dist_0, axis=0)**2

dist_1 = normal_dist * likelihood(omega + off, t, 1)
dist_1 /= np.sum(dist_1, axis=0)
var1 = np.sum(omega**2 * dist_1, axis=0) - np.sum(omega * dist_1, axis=0)**2

p_g = np.sum(normal_dist * likelihood(omega + off, t, 0), axis=0)
weighted_var = (var0 * p_g) + (var1 * (1-p_g))
print('EST_BND_GAMMA, min possible variance: ', min(np.min(var0), np.min(var1)))
print('EST_BND_GAMMA, min outcome weighted variance: ', np.min(weighted_var))
#print('EST_BND_GAMMA, random time weighted variance: ', np.min(np.mean(weighted_var, axis=1))) # note: there is no formal reason to take this particular average
off_n, t_n = np.meshgrid(off.flatten(), t.flatten())

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(t_n, off_n, var1)
ax.set_xlabel('t')
ax.set_ylabel('offset')

plt.show()


