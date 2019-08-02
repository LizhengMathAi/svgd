import numpy as np
import tensorflow as tf
import ops


tf.set_random_seed(0)
np.random.seed(0)
np.set_printoptions(precision=5, linewidth=120, suppress=True)


tf.app.flags.DEFINE_string("device", "gpu", "compute device.")
tf.app.flags.DEFINE_string("dtype", "double", "data type for io tensors.")
FLAGS = tf.app.flags.FLAGS

if FLAGS.dtype == "float":
    dtype = tf.float32
elif FLAGS.dtype == "double":
    dtype = tf.float64
else:
    raise ValueError


def identity_test():
    with tf.device('/' + FLAGS.device + ":0"):
        input_tensor = tf.Variable(initial_value=tf.truncated_normal([128, 32, 32, 64], mean=0.1, dtype=dtype))
        param = tf.constant(np.random.rand(128, 32, 32, 64), dtype=dtype)

        output_tensor = ops.identity(input_tensor)
        recv = tf.reduce_sum(output_tensor * param)
        grads = tf.gradients(recv, [input_tensor])

        output_tensor_ = tf.identity(input_tensor)
        recv_ = tf.reduce_sum(output_tensor_ * param)
        grads_ = tf.gradients(recv_, [input_tensor])

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        assert np.max(np.abs(sess.run(output_tensor - output_tensor_))) < 1e-5
        for g, g_ in zip(grads, grads_):
            np.max(np.abs(sess.run(g - g_))) < 1e-5


def argmax_test():
    with tf.device('/' + FLAGS.device + ":0"):
        vec = tf.Variable(initial_value=tf.truncated_normal(shape=[1000], dtype=dtype), name="vec")
        k = ops.reduce_argmax(vec, share_memory_size=128)
        k_ = tf.argmax(vec)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        assert sess.run(k) == sess.run(k_)


def reduce_sum_test():
    with tf.device('/' + FLAGS.device + ":0"):
        vec = tf.Variable(initial_value=tf.truncated_normal(shape=[1000], dtype=dtype), name="vec")

        recv = ops.reduce_sum(vec, share_memory_size=128)
        recv_ = tf.reduce_sum(vec)
        gradients = tf.gradients(tf.square(recv), [vec])
        gradients_ = tf.gradients(tf.square(recv_), [vec])

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        assert abs(sess.run(recv - recv_)) < 1e-4
        for g, g_ in zip(gradients, gradients_):
            assert np.mean(np.abs(sess.run(g - g_))) < 1e-4


def reduce_inner_product_test():
    with tf.device('/' + FLAGS.device + ":0"):
        vec1 = tf.Variable(initial_value=tf.truncated_normal(shape=[1000], dtype=dtype), name="vec1")
        vec2 = tf.Variable(initial_value=tf.truncated_normal(shape=[1000], dtype=dtype), name="vec2")

        recv = ops.reduce_inner_product(vec1, vec2, share_memory_size=128)
        recv_ = tf.reduce_sum(vec1 * vec2)
        gradients = tf.gradients(tf.square(recv), [vec1, vec2])
        gradients_ = tf.gradients(tf.square(recv_), [vec1, vec2])

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        assert abs(sess.run(recv - recv_)) < 1e-4
        for g, g_ in zip(gradients, gradients_):
            assert np.mean(np.abs(sess.run(g - g_))) < 1e-4


def reduce_double_inner_product_test():
    with tf.device('/' + FLAGS.device + ":0"):
        vec1 = tf.Variable(initial_value=tf.truncated_normal(shape=[1000], dtype=dtype), name="vec1")
        vec2 = tf.Variable(initial_value=tf.truncated_normal(shape=[1000], mean=0.1, dtype=dtype), name="vec2")
        vec3 = tf.Variable(initial_value=tf.truncated_normal(shape=[1000], mean=0.2, dtype=dtype), name="vec3")

        recv = ops.reduce_double_inner_product(vec1, vec2, vec3, share_memory_size=128)
        recv_ = tf.reduce_sum(vec1 * vec3) / tf.reduce_sum(vec2 * vec3)
        gradients = tf.gradients(tf.square(recv), [vec1, vec2, vec3])
        gradients_ = tf.gradients(tf.square(recv_), [vec1, vec2, vec3])

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        assert abs(sess.run(recv - recv_)) < 1e-4
        for g, g_ in zip(gradients, gradients_):
            assert np.mean(np.abs(sess.run(g - g_))) < 1e-4


def add_n_test():
    with tf.device('/' + FLAGS.device + ":0"):
        vecs = [tf.Variable(initial_value=tf.truncated_normal(shape=[1000], dtype=dtype)) for _ in range(2)]

        rhs = ops.add_n(vecs)
        rhs_ = tf.add_n(vecs)
        gradients = tf.gradients(tf.reduce_sum(tf.square(rhs)), vecs)
        gradients_ = tf.gradients(tf.reduce_sum(tf.square(rhs_)), vecs)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        assert np.mean(np.abs(sess.run(rhs - rhs_))) < 1e-4
        for g, g_ in zip(gradients, gradients_):
            assert np.mean(np.abs(sess.run(g - g_))) < 1e-4


def mat_mul_test():
    with tf.device('/' + FLAGS.device + ":0"):
        mat1 = tf.Variable(initial_value=tf.truncated_normal(shape=[7, 10], mean=0.1, dtype=dtype), name="mat1")
        mat2 = tf.Variable(initial_value=tf.truncated_normal(shape=[10, 3], mean=0.1, dtype=dtype), name="mat2")
        mat3 = tf.Variable(initial_value=tf.truncated_normal(shape=[5, 3], mean=0.1, dtype=dtype), name="mat3")

        rhs = ops.mat_mul(ops.mat_mul(mat1, mat2), mat3, transpose=True)
        rhs_ = tf.matmul(tf.matmul(mat1, mat2), mat3, transpose_b=True)
        gradients = tf.gradients(tf.reduce_sum(tf.square(rhs)), [mat1, mat2, mat3])
        gradients_ = tf.gradients(tf.reduce_sum(tf.square(rhs_)), [mat1, mat2, mat3])

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        assert np.max(np.abs(sess.run(rhs - rhs_))) < 1e-5
        for g, g_ in zip(gradients, gradients_):
            assert np.max(np.abs(sess.run(g - g_))) < 2e-5


def kronecker_product_test():
    with tf.device('/' + FLAGS.device + ":0"):
        mat1 = tf.Variable(initial_value=tf.truncated_normal(shape=[3, 4], dtype=dtype), name="mat1")
        mat2 = tf.Variable(initial_value=tf.truncated_normal(shape=[5, 2], dtype=dtype), name="mat2")

        recv = ops.kronecker_product(mat1, mat2)
        recv_ = tf.reshape(tf.einsum("ij,uv->iujv", mat1, mat2), recv.get_shape().as_list())
        gradients = tf.gradients(tf.reduce_sum(tf.square(recv)), [mat1, mat2])
        gradients_ = tf.gradients(tf.reduce_sum(tf.square(recv_)), [mat1, mat2])

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        assert np.mean(np.abs(sess.run(recv - recv_))) < 1e-4
        for g, g_ in zip(gradients, gradients_):
            assert np.mean(np.abs(sess.run(g - g_))) < 1e-2


def plu_test():
    with tf.device('/' + FLAGS.device + ":0"):
        n = 10
        m = tf.Variable(initial_value=tf.truncated_normal([n, n], dtype=dtype))
        pi, l, u = ops.plu(m)

        recv = tf.gather(m, pi) - tf.matmul(l, u)
        loss = tf.reduce_mean(l+u)
        gradients = tf.gradients(loss, [m])

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        assert np.mean(np.abs(sess.run(recv))) < 1e-4


def plu_solve_test():
    with tf.device('/' + FLAGS.device + ":0"):
        n = 8
        m = tf.Variable(initial_value=tf.truncated_normal([n, n], dtype=dtype))
        rhs = tf.Variable(initial_value=tf.truncated_normal([n], dtype=dtype))

        recv = ops.plu_solve(m, rhs)
        gradients = tf.gradients(tf.reduce_mean(recv), [m, rhs])

    with tf.device("/cpu:0"):
        m_ = tf.Variable(initial_value=m.initial_value)
        rhs_ = tf.Variable(initial_value=rhs.initial_value)

        recv_ = tf.reshape(tf.linalg.solve(m_, tf.reshape(rhs_, [n, 1])), [n])
        gradients_ = tf.gradients(tf.reduce_mean(recv_), [m_, rhs_])

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        assert np.mean(np.abs(sess.run(recv - recv_))) < 1e-5
        for g, g_ in zip(gradients, gradients_):
            assert np.mean(np.abs(sess.run(g - g_))) < 1e-5


def conv2d_test():
    with tf.device('/' + FLAGS.device + ":0"):
        input_tensor = tf.Variable(initial_value=tf.truncated_normal([8, 13, 13, 8], mean=0.1, dtype=dtype))
        kernel_1 = tf.Variable(initial_value=tf.truncated_normal([5, 5, 8, 16], mean=0.1, dtype=dtype))
        kernel_2 = tf.Variable(initial_value=tf.truncated_normal([3, 3, 16, 32], mean=0.1, dtype=dtype))
        param = tf.constant(np.random.rand(8, 2, 2, 32), dtype=dtype)

        output_tensor = ops.conv2d(input_tensor, kernel_1, strides=[1, 3, 3, 1], padding="SAME")
        output_tensor = ops.conv2d(output_tensor, kernel_2, strides=[1, 2, 2, 1], padding="VALID")
        recv = tf.reduce_mean(output_tensor * param)
        gradients = tf.gradients(recv, [input_tensor, kernel_1, kernel_2])

        output_tensor_ = tf.nn.conv2d(input_tensor, kernel_1, strides=[1, 3, 3, 1], padding="SAME")
        output_tensor_ = tf.nn.conv2d(output_tensor_, kernel_2, strides=[1, 2, 2, 1], padding="VALID")
        recv_ = tf.reduce_mean(output_tensor_ * param)
        gradients_ = tf.gradients(recv_, [input_tensor, kernel_1, kernel_2])

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        assert np.max(np.abs(sess.run(output_tensor - output_tensor_))) < (1e-2 if FLAGS.dtype == "float" else 1e-10)
        for g, g_ in zip(gradients, gradients_):
            assert np.max(np.abs(sess.run(g - g_))) < (1e-2 if FLAGS.dtype == "float" else 1e-10)


def bias_add_test():
    with tf.device('/' + FLAGS.device + ":0"):
        input_tensor = tf.Variable(initial_value=tf.truncated_normal([2, 11, 11, 3], mean=0.1, dtype=dtype))
        bias_1 = tf.Variable(initial_value=tf.truncated_normal([3], mean=0.1, dtype=dtype))
        bias_2 = tf.Variable(initial_value=tf.truncated_normal([11], mean=0.1, dtype=dtype))
        param = tf.constant(np.random.rand(66, 11), dtype=dtype)

        output_tensor = ops.bias_add(tf.reshape(ops.bias_add(input_tensor, bias_1), [-1, 11]), bias_2)
        recv = tf.reduce_mean(output_tensor * param)
        grads = tf.gradients(recv, [input_tensor, bias_1, bias_2])

        output_tensor_ = tf.reshape(input_tensor + bias_1, [-1, 11]) + bias_2
        recv_ = tf.reduce_mean(output_tensor_ * param)
        grads_ = tf.gradients(recv_, [input_tensor, bias_1, bias_2])

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        assert np.max(np.abs(sess.run(output_tensor - output_tensor_))) < 1e-5
        for g, g_ in zip(grads, grads_):
            assert np.max(np.abs(sess.run(g - g_))) < 1e-5


def max_pool_test():
    with tf.device('/' + FLAGS.device + ":0"):
        input_tensor = tf.Variable(initial_value=tf.truncated_normal([128, 11, 11, 64], mean=0.1, dtype=dtype))
        param = tf.constant(np.random.rand(128, 2, 2, 64), dtype=dtype)

        output_tensor = ops.max_pool(input_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        output_tensor = ops.max_pool(output_tensor, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding="VALID")
        recv = tf.reduce_mean(output_tensor * param)
        grads = tf.gradients(recv, [input_tensor])

    with tf.device("/cpu:0"):
        output_tensor_ = tf.nn.max_pool(input_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        output_tensor_ = tf.nn.max_pool(output_tensor_, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding="VALID")
        recv_ = tf.reduce_mean(output_tensor_ * param)
        grads_ = tf.gradients(recv_, [input_tensor])

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        assert np.max(np.abs(sess.run(output_tensor) - sess.run(output_tensor_))) < 1e-5
        for g, g_ in zip(grads, grads_):
            assert np.max(np.abs(sess.run(g) - sess.run(g_))) < 1e-5


def batch_norm_test():
    with tf.device('/' + FLAGS.device + ":0"):
        input_tensor = tf.Variable(initial_value=tf.truncated_normal([4, 3, 3, 2], mean=0.1, dtype=dtype))
        mean = tf.Variable(initial_value=tf.truncated_normal([2], mean=0.1, dtype=dtype))
        variance = tf.Variable(initial_value=tf.square(tf.truncated_normal([2], mean=0.1, dtype=dtype)))
        offset = tf.Variable(initial_value=tf.truncated_normal([2], mean=0.1, dtype=dtype))
        scale = tf.Variable(initial_value=tf.truncated_normal([2], mean=0.1, dtype=dtype))
        param_1 = tf.constant(np.random.rand(4, 3, 3, 2), dtype=dtype)
        param_2 = tf.constant(np.random.rand(4 * 3 * 3, 2), dtype=dtype)

        output_tensor = ops.batch_norm(input_tensor, mean, variance, offset, scale, 1e-6)
        output_tensor = tf.reshape(output_tensor * param_1, [-1, 2])
        output_tensor = ops.batch_norm(output_tensor, mean, variance, offset, scale, 1e-6)
        recv = tf.reduce_sum(output_tensor * param_2)
        grads = tf.gradients(recv, [input_tensor, mean, variance, offset, scale])

        output_tensor_ = tf.nn.batch_normalization(input_tensor, mean, variance, offset, scale, 1e-6)
        output_tensor_ = tf.reshape(output_tensor_ * param_1, [-1, 2])
        output_tensor_ = ops.batch_norm(output_tensor_, mean, variance, offset, scale, 1e-6)
        recv_ = tf.reduce_sum(output_tensor_ * param_2)
        grads_ = tf.gradients(recv_, [input_tensor, mean, variance, offset, scale])

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            assert np.max(np.abs(sess.run(output_tensor) - sess.run(output_tensor_))) < 1e-5
            for g, g_ in zip(grads, grads_):
                assert np.max(np.abs(sess.run(g) - sess.run(g_))) < 2e-4


def main(argv):
    identity_test()
    print("\033[34midentity test passed!\033[0m")
    argmax_test()
    print("\033[34margmax test passed!\033[0m")
    reduce_sum_test()
    print("\033[34mreduce_sum test passed!\033[0m")
    reduce_inner_product_test()
    print("\033[34mreduce_inner_product test passed!\033[0m")
    reduce_double_inner_product_test()
    print("\033[34mreduce_double_inner_product test passed!\033[0m")
    add_n_test()
    print("\033[34madd_n test passed!\033[0m")
    mat_mul_test()
    print("\033[34mmat_mul test passed!\033[0m")
    kronecker_product_test()
    print("\033[34mkronecker_product test passed!\033[0m")
    plu_test()
    print("\033[34mplu test passed!\033[0m")
    plu_solve_test()
    print("\033[34mplu_solve test passed!\033[0m")
    conv2d_test()
    print("\033[34mconv2d test passed!\033[0m")
    bias_add_test()
    print("\033[34mbias_add test passed!\033[0m")
    max_pool_test()
    print("\033[34mmax_pool test passed!\033[0m")
    batch_norm_test()
    print("\033[34mbatch_norm test passed!\033[0m")


if __name__ == '__main__':
    tf.app.run(main)
