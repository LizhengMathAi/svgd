import tensorflow as tf
from tensorflow.python.framework import ops


OP_MODULE = tf.load_op_library('./src/ops.so')


# ---------------------------------------- for Variable ----------------------------------------
def identity(input_tensor):
    dim = input_tensor.get_shape().as_list().__len__()
    return OP_MODULE.def_identity(input_tensor, dim=dim)


@ops.RegisterGradient('DefIdentity')
def gradients(op, grad):
    return grad * tf.rsqrt(tf.square(grad) + 1e-6)


# ---------------------------------------- for Tensor ----------------------------------------
def grad_transform(grad):
    return grad * tf.rsqrt(tf.square(grad) + 1e-6)


def matmul(mat1, mat2):
    return OP_MODULE.def_mat_mul(mat1, mat2, transpose=False)


@ops.RegisterGradient('DefMatMul')
def gradients(op, grad):
    mat1_tensor, mat2_tensor = op.inputs

    transpose_mat1_tensor = tf.transpose(mat1_tensor, perm=[1, 0])
    transpose_mat2_tensor = tf.transpose(mat2_tensor, perm=[1, 0])
    grad_mat1 = tf.matmul(grad, transpose_mat2_tensor)
    grad_mat2 = tf.matmul(transpose_mat1_tensor, grad_transform(grad))
    return grad_mat1, grad_mat2

