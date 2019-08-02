import tensorflow as tf
from tensorflow.python.framework import ops


OP_MODULE = tf.load_op_library('./src/ops.so')

def identity(input_tensor):
    dim = input_tensor.get_shape().as_list().__len__()
    return OP_MODULE.def_identity(input_tensor, dim=dim)


@ops.RegisterGradient('DefIdentity')
def gradients(op, grad):
    return grad


def reduce_argmax(vec, share_memory_size: int = None):
    return OP_MODULE.def_reduce_argmax(vec, share_memory_size=share_memory_size)


def reduce_sum(vec, share_memory_size: int = None):
    if share_memory_size is None:
        return OP_MODULE.def_reduce_sum(vec)
    else:
        return OP_MODULE.def_reduce_sum(vec, share_memory_size=share_memory_size)


@ops.RegisterGradient('DefReduceSum')
def gradients(op, grad_recv):
    vec = op.inputs[0]

    return grad_recv * tf.ones(vec.get_shape().as_list(), dtype=vec.dtype)


def reduce_inner_product(vec1, vec2, share_memory_size: int = None):
    if share_memory_size is None:
        return OP_MODULE.def_reduce_inner_product(vec1, vec2)
    else:
        return OP_MODULE.def_reduce_inner_product(vec1, vec2, share_memory_size=share_memory_size)


@ops.RegisterGradient('DefReduceInnerProduct')
def gradients(op, grad_recv):
    vec1 = op.inputs[0]
    vec2 = op.inputs[1]

    return grad_recv * vec2, grad_recv * vec1


def reduce_double_inner_product(vec1, vec2, vec3, share_memory_size: int = None):
    if share_memory_size is None:
        return OP_MODULE.def_reduce_double_inner_product(vec1, vec2, vec3)
    else:
        return OP_MODULE.def_reduce_double_inner_product(vec1, vec2, vec3, share_memory_size=share_memory_size)


@ops.RegisterGradient('DefReduceDoubleInnerProduct')
def gradients(op, grad_recv):
    vec1 = op.inputs[0]
    vec2 = op.inputs[1]
    vec3 = op.inputs[2]

    recv = op.outputs[0]
    recv23 = tf.reduce_sum(vec2 * vec3)

    return [
        grad_recv * vec3 / recv23,
        -grad_recv * recv * vec3 / recv23,
        grad_recv * (vec1 - vec2 * recv) / recv23
    ]


def add_n(vecs):
    vecs = vecs if isinstance(vecs, list) else [vecs]
    return OP_MODULE.def_add_n(vecs)


@ops.RegisterGradient('DefAddN')
def gradients(op, grad_rhs):
    vecs = list(op.inputs)

    return [grad_rhs for _ in range(vecs.__len__())]


def mat_mul(mat1, mat2, transpose: bool = False):
    return OP_MODULE.def_mat_mul(mat1, mat2, transpose=transpose)


@ops.RegisterGradient('DefMatMul')
def gradients(op, grad_recv):
    mat1 = op.inputs[0]
    mat2 = op.inputs[1]

    transpose = op.get_attr("transpose")

    if transpose:
        transpose_grad_recv = tf.transpose(grad_recv, perm=[1, 0])

        return mat_mul(grad_recv, mat2), mat_mul(transpose_grad_recv, mat1)
    else:
        transpose_mat1 = tf.transpose(mat1, perm=[1, 0])
        transpose_mat2 = tf.transpose(mat2, perm=[1, 0])

        return mat_mul(grad_recv, transpose_mat2), mat_mul(transpose_mat1, grad_recv)


def kronecker_product(mat1, mat2):
    return OP_MODULE.def_kronecker_product(mat1, mat2)


@ops.RegisterGradient('DefKroneckerProduct')
def gradients(op, grad_recv):
    mat1 = op.inputs[0]
    mat2 = op.inputs[1]

    m, n = mat1.get_shape().as_list()
    p, q = mat2.get_shape().as_list()
    grad_recv_tensor = tf.reshape(grad_recv, [m, p, n, q])

    return tf.einsum("iujv,uv->ij", grad_recv_tensor, mat2), tf.einsum("iujv,ij->uv", grad_recv_tensor, mat1)


def plu(m, one_hot=False):
    """L*U = pi -> (A*B)[indices]"""
    return OP_MODULE.def_plu(m, one_hot=one_hot)


@ops.RegisterGradient('DefPlu')
def gradients(op, grad_pi, grad_l, grad_u):
    m = op.inputs[0]

    pi = op.outputs[0]
    l = op.outputs[1]
    u = op.outputs[2]

    one_hot = op.get_attr("one_hot")
    if one_hot:
        pi = tf.cast(pi, tf.float32)
        pi = tf.cast(tf.argmax(pi, axis=1), pi.dtype)

    return OP_MODULE.def_plu_grad(pi, l, u, grad_l, grad_u)


def plu_solve(m, rhs):
    return OP_MODULE.def_plu_solve(m, rhs)


@ops.RegisterGradient('DefPluSolve')
def gradients(op, grad_solve):
    m = op.inputs[0]
    rhs = op.inputs[1]

    solve = op.outputs[0]

    transpose_m = tf.transpose(m, perm=[1, 0])

    grad_rhs = plu_solve(transpose_m, grad_solve)
    grad_m = -tf.einsum("i,j->ij", grad_rhs, solve)

    return grad_m, grad_rhs


def conv2d(input_tensor, kernel, strides, padding):
    return OP_MODULE.def_conv2d(input_tensor, kernel, strides=strides, padding=padding)


@ops.RegisterGradient('DefConv2d')
def gradients(op, grad_tensor):
    input_tensor = op.inputs[0]
    kernel = op.inputs[1]

    strides = op.get_attr("strides")
    padding = op.get_attr("padding")

    # grad_input_tensor = OP_MODULE.def_conv2d_grad_input(grad_tensor, input_tensor, kernel, strides=strides, padding=padding)
    # grad_kernel = OP_MODULE.def_conv2d_grad_kernel(grad_tensor, input_tensor, kernel, strides=strides, padding=padding)
    # return grad_input_tensor, grad_kernel
    return OP_MODULE.def_conv2d_grad(grad_tensor, input_tensor, kernel, strides=strides, padding=padding)

def bias_add(input_tensor, bias):
    assert input_tensor.get_shape().as_list().__len__() in [2, 4]
    return OP_MODULE.def_bias_add(input_tensor, bias, dim=input_tensor.get_shape().as_list().__len__())


@ops.RegisterGradient('DefBiasAdd')
def gradients(op, grad_tensor):
    dim = op.get_attr("dim")

    if dim == 4:
        return grad_tensor, tf.einsum("nhwc->c", grad_tensor)
    else:
        return grad_tensor, tf.einsum("nc->c", grad_tensor)


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


def batch_norm(input_tensor, mean, variance, offset, scale, variance_epsilon):
    dim = input_tensor.get_shape().as_list().__len__()
    return OP_MODULE.def_batch_norm(
        input_tensor, mean, variance, offset, scale, variance_epsilon=variance_epsilon, dim=dim)


@ops.RegisterGradient('DefBatchNorm')
def gradients(op, grad_output_tensor):
    input_tensor = op.inputs[0]
    mean = op.inputs[1]
    variance = op.inputs[2]
    scale = op.inputs[4]

    variance_epsilon = op.get_attr("variance_epsilon")
    dim = op.get_attr("dim")
    
    return OP_MODULE.def_grad_batch_norm(
        grad_output_tensor, input_tensor, mean, variance, scale, variance_epsilon=variance_epsilon, dim=dim)
