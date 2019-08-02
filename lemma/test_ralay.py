import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib import cm
from matplotlib import pyplot as plt


np.set_printoptions(precision=6, suppress=True, threshold=np.nan)

n = 3
M = 2 * np.random.rand(n, n) - 1
Lam = np.random.rand(n)
P = np.einsum("i,ij,jk->ik", Lam, M.T, M) + np.einsum("ij,jk,k->ik", M.T, M, Lam)
print(P)
eig = sorted(np.linalg.eig(P)[0])
print(eig)

theta, phi = np.meshgrid(np.linspace(-np.pi, np.pi, 96), np.linspace(-np.pi / 2, np.pi / 2, 64))
v = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])
rho = np.einsum("ixy,ij,jxy->xy", v, P, v)

X = np.cos(theta) * np.cos(phi)
Y = np.sin(theta) * np.cos(phi)
Z = np.sin(phi)


fig = plt.figure()
ax = fig.gca(projection='3d')
if eig[0] < 0:
    levels = sorted(list(eig) + [0])
    levels = list(np.linspace(levels[0], levels[1], 5, endpoint=False)) + \
             list(np.linspace(levels[1], levels[2], 5, endpoint=False)) + \
             list(np.linspace(levels[2], levels[3], 10, endpoint=False))
else:
    levels = list(np.linspace(eig[0], eig[1], 5, endpoint=False)) + \
             list(np.linspace(eig[1], eig[2], 10, endpoint=False))
CS = ax.contour(theta, phi, rho, levels=levels)
allsegs = CS.allsegs
tcolors = CS.tcolors

plt.clf()
ax = fig.gca(projection='3d')

# show contour lines
for level, lines, colors in zip(levels, allsegs, tcolors):
    for line in lines:
        theta = line[:, 0]
        phi = line[:, 1]
        ax.plot(np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi), c=colors[0])

        if level == 0:
            ax.plot(np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi), c='r')

# show Lam line
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.05, linewidth=0, antialiased=False)
Lam_line = np.einsum("i, j->ij", np.array(1 / Lam / np.linalg.norm(1 / Lam, ord=2)), [-1, 1])
# ax.plot(Lam_line[0], Lam_line[1], Lam_line[2])
# ax.scatter(Lam_line[0], Lam_line[1], Lam_line[2])

ax.set_title(r"$\rho(v) = \frac{v^TPv}{v^Tv}, P=MM^T\Lambda + \Lambda MM^T$")

plt.show()


# n = 16
# rho_list = []
# for i in range(100):
#     M = 1 * np.random.rand(n, n) - 2
#
#     v = 2 * np.random.rand(n) - 1
#     v = v / np.linalg.norm(v, ord=2)
#
#     tv = np.random.rand(n) * v
#     tv = tv / np.linalg.norm(tv, ord=2)
#
#     rho = np.einsum("i,ij,jk,k->", v, M, M.T, tv)
#     rho_list.append(rho)
#
# print(np.mean(np.array(rho_list) > 0))
