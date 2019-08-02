#!/usr/bin/env bash
set -e

echo -e "\033[34mactivate tensorflow environment\033[0m"
source ~/Documents/pyenv/tensorflow/bin/activate 

# --------------------------
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
CUDA_PATH=/usr/local/cuda/

# --------------------------
echo -e "\033[34mcompiling ops.cu.cc\033[0m"
nvcc -std=c++11 -c -o ops.cu.o ops.cu.cc \
	-I /usr/local \
	${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr

echo -e "\033[34mcompiling ops.cc\033[0m"
g++ -std=c++11 -shared -o ops.so ops.cc ops.cu.o \
	${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L $CUDA_PATH/lib64

echo -e "\033[34mremove ops.cu.o\033[0m"
rm ops.cu.o

# --------------------------
echo -e "\033[34mBuild done!\033[0m"

# --------------------------
echo -e "\033[34mstart to test ops.so\033[0m"
cd ../

for device in "cpu" "gpu"
do
	for dtype in "float" "double"
	do
		echo -e "\033[34mpython so_test.py --device $device --dtype $dtype\033[0m"
		python so_test.py --device $device --dtype $dtype
	done
done

# --------------------------
echo -e "\033[34mTest done!\033[0m"
