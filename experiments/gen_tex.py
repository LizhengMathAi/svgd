import pandas as pd

for file_name in ["so", "sgd", "rmsprop", "adam"]:
    # ----------------------- mlp -----------------------
    df = pd.read_csv("./logs/csv/mlp_" + file_name + ".csv").values[:, [2, 3, 4]]

    with open("./logs/dat/mlp_" + file_name + "_loss.dat", "w") as f:
        f.write("iter loss \n")
        for line in df:
            f.write("{} {} \n".format(str(line[0]/1000), str(line[1])))

    with open("./logs/dat/mlp_" + file_name + "_error.dat", "w") as f:
        f.write("iter error \n")
        for line in df:
            f.write("{} {} \n".format(str(line[0]/1000), str((1 - line[2]) * 100)))

    # ----------------------- vgg -----------------------
    df = pd.read_csv("./logs/csv/vgg_" + file_name + ".csv").values[:, [0, 2, 3]]

    with open("./logs/dat/vgg_" + file_name + "_loss.dat", "w") as f:
        f.write("iter loss \n")
        for line in df:
            f.write("{} {} \n".format(str(line[0]/1000), str(line[1])))

    with open("./logs/dat/vgg_" + file_name + "_error.dat", "w") as f:
        f.write("iter error \n")
        for line in df:
            f.write("{} {} \n".format(str(line[0]/1000), str(100 - line[2])))

    # ----------------------- resnet-20 -----------------------
    df = pd.read_csv("./logs/csv/resnet_" + file_name + ".csv").values[:, [0, 2, 3]]

    with open("./logs/dat/resnet_" + file_name + "_loss.dat", "w") as f:
        f.write("iter loss \n")
        for line in df:
            f.write("{} {} \n".format(str(line[0]/10000), str(line[1])))

    with open("./logs/dat/resnet_" + file_name + "_error.dat", "w") as f:
        f.write("iter error \n")
        for line in df:
            f.write("{} {} \n".format(str(line[0]/10000), str(100 - line[2])))

