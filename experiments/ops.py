import tensorflow as tf
from tensorflow.python.framework import ops


OP_MODULE = tf.load_op_library('./src/ops.so')

SCALE = 0.1


# ---------------------------------------- for Variable ----------------------------------------
def identity(input_tensor):
    dim = input_tensor.get_shape().as_list().__len__()
    return OP_MODULE.def_identity(input_tensor, dim=dim)


@ops.RegisterGradient('DefIdentity')
def gradients(op, grad):
    r = tf.Variable(initial_value=tf.zeros_like(grad), trainable=False)
    r_ = 0.9 * r + 0.1 * tf.square(grad)

    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, r.assign(r_))
    return grad * tf.rsqrt(r + 1e-6)


# ---------------------------------------- for Tensor ----------------------------------------
def grad_transform(grad):
    shape = grad.get_shape().as_list()[1:]
    r = tf.Variable(initial_value=tf.zeros([1] + shape), trainable=False, dtype=grad.dtype)
    r_ = 0.9 * r + tf.reduce_mean(0.1 * tf.square(grad), axis=0, keepdims=True)

    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, r.assign(r_))
    return grad * tf.rsqrt(r + 1e-6)


def matmul(mat1, mat2):
    return OP_MODULE.def_mat_mul(mat1, mat2, transpose=False)


@ops.RegisterGradient('DefMatMul')
def gradients(op, grad):
    mat1_tensor, mat2_tensor = op.inputs

    transpose_mat1_tensor = tf.transpose(mat1_tensor, perm=[1, 0])
    transpose_mat2_tensor = tf.transpose(mat2_tensor, perm=[1, 0])
    grad_mat1 = tf.matmul(grad, transpose_mat2_tensor)
    grad_mat2 = tf.matmul(transpose_mat1_tensor, grad_transform(grad))

    return grad_mat1, SCALE * grad_mat2


def conv2d(input_tensor, kernel, strides, padding):
    return OP_MODULE.def_conv2d(input_tensor, kernel, strides=strides, padding=padding)


@ops.RegisterGradient('DefConv2d')
def gradients(op, grad):
    input_tensor, kernel = op.inputs

    grad_input_tensor = OP_MODULE.def_conv2d_grad_input(
        grad, input_tensor, kernel, strides=op.get_attr("strides"), padding=op.get_attr("padding"))
    grad_kernel = OP_MODULE.def_conv2d_grad_kernel(
        grad_transform(grad), input_tensor, kernel, strides=op.get_attr("strides"), padding=op.get_attr("padding"))

    return grad_input_tensor, SCALE * grad_kernel


# def bias_add(input_tensor, bias):
#     assert input_tensor.get_shape().as_list().__len__() in [2, 4]
#     return OP_MODULE.def_bias_add(input_tensor, bias, dim=input_tensor.get_shape().as_list().__len__())
#
#
# @ops.RegisterGradient('DefBiasAdd')
# def gradients(op, grad_tensor):
#     dim = op.get_attr("dim")
#
#     if dim == 4:
#         return grad_tensor, tf.einsum("nhwc->c", grad_tensor)
#     else:
#         return grad_tensor, tf.einsum("nc->c", grad_tensor)


def max_pool(input_tensor, ksize, strides, padding):
    return OP_MODULE.def_max_pool(input_tensor, ksize=ksize, strides=strides, padding=padding)


@ops.RegisterGradient('DefMaxPool')
def gradients(op, grad_output_tensor):
    input_tensor = op.inputs[0]
    output_tensor = op.outputs[0]

    ksize = op.get_attr("ksize")
    strides = op.get_attr("strides")
    padding = op.get_attr("padding")
    return OP_MODULE.def_max_pool_grad(
        grad_output_tensor, input_tensor, output_tensor, ksize=ksize, strides=strides, padding=padding)

