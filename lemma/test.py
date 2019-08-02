import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt


np.set_printoptions(precision=6, suppress=True, threshold=np.nan)


df = pd.DataFrame(columns=['n', 'vtv', "vMTMTv"])

n = 5
vMTMTv_list = []
for i in range(20):
    M = 2 * np.random.rand(n, n) - 1

    temp_list = []
    for j in range(1000):
        v = 2 * np.random.rand(n) - 1
        v = v / np.linalg.norm(v, ord=2)

        tv = np.random.rand(n) * v
        tv = 0.5 * tv / np.linalg.norm(tv, ord=2)

        vtv = np.einsum("i,i", v, tv)

        temp_list.append(np.einsum("i,ik,kj,j", v, M.T, M, tv))
    print(temp_list)
    print(np.mean(np.array(temp_list) > 0))
    vMTMTv_list.append(np.mean(temp_list))

print(vMTMTv_list)

tab = []
for i in range(1000):
    M = 1 * np.random.rand(n, n) - 2
    Lam = np.random.rand(n)
    P = np.einsum("i,ij,jk->ik", Lam, M.T, M) + np.einsum("ij,jk,k->ik", M.T, M, Lam)
    tab.append(np.linalg.eig(P)[0])
tab = np.array(tab)
print(tab)
print(np.mean(tab, axis=0))


M = 2 * np.random.rand(n, n) - 1
Lam = np.random.rand(n)
P = np.einsum("i,ij,jk->ik", Lam, M.T, M) + np.einsum("ij,jk,k->ik", M.T, M, Lam)
u, s, vh = np.linalg.svd(P)
print(u @ u.T)
print(vh @ vh.T)
print(u - vh.T)
es = list(vh)
print(es[0] / np.einsum("ij,j->i", P, es[0]))

fig = plt.figure()
ax = fig.gca(projection='3d')

# plt.show()

P = M.T@M
u, s, vh = np.linalg.svd(P)
print(u - vh.T)