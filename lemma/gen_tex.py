import pandas as pd


df = pd.read_csv("./logs/csv/lemma.csv")

for n in [2, 5, 10, 25]:
    with open("./logs/dat/lemma_level_{}.dat".format(n), "w") as f:
        f.write("vtu f_lam \n")

        for line in df[df['n'] == n].values:
            f.write("{} {} \n".format(str(line[2]), str(line[3])))


with open("../../fig-Lemma.tex", "w") as f:
    f.write(r"\begin{figure}[H]" + "\n")
    f.write("\t" + r"\centering" + "\n")
    f.write("\t" + r"\pgfplotsset{compat=1.5}" + "\n")
    f.write("\n")
    f.write("\t" + r"\begin{tikzpicture}%[xscale=0.7,yscale=0.7]" + "\n")
    f.write("\t" + r"\begin{axis}[" + "\n")
    f.write("\t" + r"xlabel={$v^Tu$}," + "\n")
    f.write("\t" + r"ylabel={$\tilde{f}_{\lambda}(v, u)$}," + "\n")
    f.write("\t" + r"grid=major," + "\n")
    f.write("\t" + r"legend pos=south east," + "\n")
    f.write("\t" + r"legend entries={n=2$\lambda$=1,n=5$\lambda$=2,n=10$\lambda$=3,n=25$\lambda$=4}," + "\n")
    f.write("\t" + r"]" + "\n")
    f.write("\t" + r"\addplot table {./virtual_grad/lemma/logs/dat/lemma_level_2.dat};" + "\n")
    f.write("\t" + r"\addplot table {./virtual_grad/lemma/logs/dat/lemma_level_5.dat};" + "\n")
    f.write("\t" + r"\addplot table {./virtual_grad/lemma/logs/dat/lemma_level_10.dat};" + "\n")
    f.write("\t" + r"\addplot table {./virtual_grad/lemma/logs/dat/lemma_level_25.dat};" + "\n")
    f.write("\t" + r"\end{axis}" + "\n")
    f.write("\t" + r"\end{tikzpicture}" + "\n")
    f.write("\t" + r"\caption{$\cdots$}" + "\n")
    f.write("\t" + r"\label{fig:Lemma}" + "\n")
    f.write(r"\end{figure}")
