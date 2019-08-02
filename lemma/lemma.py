import numpy as np
import pandas as pd


df = pd.DataFrame(columns=['n', "lam", "vtu", "f_lam"])

cursor = 0
for n, lam in zip([2, 5, 10, 25], [1, 2, 3, 4]):
    for i in range(10):
        alpha = 2 * np.random.rand(n) - 1
        alpha = alpha / np.linalg.norm(alpha, ord=2)

        gamma = 2 * np.random.rand(n) - 1
        gamma = gamma / np.linalg.norm(gamma, ord=2)

        vtu = np.einsum("i,i", alpha, gamma)

        f_lams = []
        for j in range(10000):
            B = 2 * np.random.rand(n, n) - 1
            B = B / np.linalg.norm(B, ord=2) * lam

            f_lams.append(np.einsum("i,ik,kj,j", alpha, B.T, B, gamma))
        f_lam = np.mean(f_lams)

        df.loc[cursor] = [n, lam, vtu, f_lam]
        cursor += 1
df = df.sort_values(['n', "lam", "vtu"])
df.to_csv("./logs/csv/lemma.csv", index=False)
