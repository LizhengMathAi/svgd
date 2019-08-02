import pandas as pd

for file_name in ["so", "gd", "adagrad"]:
    df = pd.read_csv("./logs/csv/math_" + file_name + ".csv").values[:, [1, 2]]

    with open("./logs/dat/" + file_name + ".dat", "w") as f:
        f.write("iter loss \n")
        for line in df:
            f.write("{} {} \n".format(str(line[0]), str(line[1])))


with open("../../fig-Math-Question.tex", "w") as f:
    f.write(r"\begin{figure}[H]" + "\n")
    f.write("\t" + r"\centering" + "\n")
    f.write("\t" + r"\pgfplotsset{compat=1.5}" + "\n")
    f.write("\n")
    f.write("\t" + r"\begin{tikzpicture}%[xscale=0.7,yscale=0.7]" + "\n")
    f.write("\t" + r"\begin{axis}[" + "\n")
    f.write("\t" + r"xlabel={iter.}," + "\n")
    f.write("\t" + r"ylabel={test loss}," + "\n")
    f.write("\t" + r"grid=major," + "\n")
    f.write("\t" + r"legend pos=north east," + "\n")
    f.write("\t" + r"legend entries={GD,AdaGrad,VGD}," + "\n")
    f.write("\t" + r"]" + "\n")
    f.write("\t" + r"\addplot table {./virtual_grad/math_question/logs/dat/gd.dat};" + "\n")
    f.write("\t" + r"\addplot table {./virtual_grad/math_question/logs/dat/adagrad.dat};" + "\n")
    f.write("\t" + r"\addplot table {./virtual_grad/math_question/logs/dat/so.dat};" + "\n")
    f.write("\t" + r"\end{axis}" + "\n")
    f.write("\t" + r"\end{tikzpicture}" + "\n")
    f.write("\t" + r"\caption{$\cdots$}" + "\n")
    f.write("\t" + r"\label{fig:Math-Question}" + "\n")
    f.write(r"\end{figure}")