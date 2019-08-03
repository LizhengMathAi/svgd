# svgd

## Introduction
Codes for paper "[SVGD: A VIRTUAL GRADIENTS DESCENT METHOD FOR STOCHASTIC OPTIMIZATION](https://arxiv.org/abs/1907.04021)".

The second encapsulation form(Graph theory algorithm, no need to compile) of [SVGD](https://arxiv.org/abs/1907.04021) will appear in the repository for my next paper as a special case.

## Compile & Experiments
Generate the library `ops.so`.
```bash
# Environment Required:
# ubuntu 16.04, cuda-9.0, python 3.5, tensorflow-gpu 1.8.0
svgd/ops/src$ bash builid.sh  # compiling ops.cc & ops.cu.cc
svgd/ops$ python so_test.py  # test custom ops and kernels
```

Proof of Lemma 4.1 in paper. (Fig. 7)
```bash
svgd/lemma$ python lemma.py  # generate data in svgd/lemma/logs/csv/lemma.csv
```

Generate training logs in `svgd/experiments/logs/csv` (Replace `./experiments/src/ops.so` by `./ops/src/ops.so` if necessary.)
```bash
# Experiments of multi-layer neural network in paper. (Fig. 9)
svgd/experiments$ python mlp_sgd.py
svgd/experiments$ python mlp_rmsprop.py
svgd/experiments$ python mlp_adam.py
svgd/experiments$ python mlp_so.py

# Experiments of convolutional neural network in paper. (Fig. 11)
svgd/experiments$ python vgg_sgd.py
svgd/experiments$ python vgg_rmsprop.py
svgd/experiments$ python vgg_adam.py
svgd/experiments$ python vgg_so.py

# Experiments of deep neural network in paper. (Fig. 12) 
svgd/experiments$ python resnet_sgd.py
svgd/experiments$ python resnet_rmsprop.py
svgd/experiments$ python resnet_adam.py
svgd/experiments$ python resnet_so.py
```