import numpy as np
import tensorflow as tf

# import ops


def bn_layer(input_tensor):
    size = input_tensor.get_shape().as_list()[-1]

    mean, variance = tf.nn.moments(input_tensor, axes=[0, 1, 2])
    beta = tf.Variable(initial_value=tf.zeros(size, dtype=tf.float32), name="beta")
    gamma = tf.Variable(initial_value=tf.ones(size, dtype=tf.float32), name="gamma")

    return tf.nn.batch_normalization(input_tensor, mean, variance, beta, gamma, 0.001)


def conv_layer(input_tensor, filter_shape, stride):
    """
    A helper function to conv, batch normalize and relu the input tensor sequentially
    :param input_tensor: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, in_channels, out_channels]
    :param stride: stride size for conv
    :return: 4D tensor. Y = conv2d(ReLU(batch_normalize(X))) or Y = ReLU(batch_normalize(conv2d(X)))
    """
    assert input_tensor.get_shape().as_list()[-1] == filter_shape[2]

    kernel_init = tf.truncated_normal(filter_shape, stddev=np.sqrt(2 / (filter_shape[2] + filter_shape[3])))
    kernel = tf.Variable(initial_value=kernel_init, dtype=tf.float32, name="kernel")
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.reduce_sum(tf.abs(kernel)))

    conv2d_tensor = tf.nn.conv2d(input_tensor, kernel, strides=[1, stride, stride, 1], padding='SAME')

    bias = tf.Variable(initial_value=tf.zeros((filter_shape[3]), dtype=tf.float32), name="bias")
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.reduce_sum(tf.abs(bias)))
    add_tensor = conv2d_tensor + bias

    output_tensor = tf.nn.relu(add_tensor)
    return output_tensor


def vgg_block(input_tensor, num, output_channels, scope='vgg_block'):
    input_channels = input_tensor.get_shape().as_list()[-1]

    with tf.name_scope(scope):
        for i in range(num - 1):
            input_tensor = conv_layer(input_tensor, filter_shape=[3, 3, input_channels, input_channels], stride=1)
        input_tensor = conv_layer(input_tensor, filter_shape=[3, 3, input_channels, output_channels], stride=1)
        output_tensor = bn_layer(input_tensor)
    return output_tensor


def fc_layer(input_tensor, out_channels):
    """
    Helper function to do batch normalization of full connection layer
    :param input_tensor: 2D tensor
    :param out_channels: int
    :return: the 2D tensor after being normalized
    """
    weights_shape = [input_tensor.get_shape().as_list()[-1], out_channels]

    weights_init = tf.truncated_normal(weights_shape, stddev=np.sqrt(2 / (weights_shape[0] + weights_shape[1])))
    weights = tf.Variable(initial_value=weights_init, dtype=tf.float32, name="weights")
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.reduce_sum(tf.abs(weights)))

    mul_tensor = tf.matmul(input_tensor, weights)

    bias = tf.Variable(initial_value=tf.zeros((weights_shape[1]), dtype=tf.float32), name="bias")
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.reduce_sum(tf.abs(bias)))
    add_tensor = mul_tensor + bias

    return add_tensor


def inference(input_tensor_batch, n):
    """
    The main function that defines the ResNet. total layers = 1 + 2n + 2n + 2n +1 = 6n + 2
    :param input_tensor_batch: 4D tensor
    :param n: num_residual_blocks
    :return: last layer in the network. Not softmax-ed
    """
    tensor = vgg_block(input_tensor_batch, 2, 16)
    tensor = tf.nn.max_pool(tensor, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')

    tensor = vgg_block(tensor, 2, 32)
    tensor = tf.nn.max_pool(tensor, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')

    tensor = vgg_block(tensor, 1, 128)
    tensor = tf.reduce_mean(tensor, [1, 2])

    with tf.name_scope('fc'):
        logits = fc_layer(tensor, 10)
    return logits


def gen_loss(labels, logits, reg_rate):
    with tf.name_scope("loss"):
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        loss_reg = loss + reg_rate * tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    return loss, loss_reg


def gen_accuracy(labels, logits):
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


def gen_train_op(loss, lr):
    # vs = tf.trainable_variables()
    # gs = tf.gradients(loss, vs)
    # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    #     return tf.group(*[v.assign_sub(lr * g) for v, g in zip(vs, gs)])
    return tf.train.GradientDescentOptimizer(lr).minimize(loss)


tf.app.flags.DEFINE_float("p1", 2, "p1.")
tf.app.flags.DEFINE_float("p2", 0.5, "p2.")
tf.app.flags.DEFINE_float("p3", 0.005, "p3.")
tf.app.flags.DEFINE_float("r", 1e-5, "r.")
# tf.app.flags.DEFINE_float("s", 0.1, "s.")
FLAGS = tf.app.flags.FLAGS


def main(argv):
    from datetime import datetime
    import pandas as pd
    import cifar10

    images = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name="images")
    labels = tf.placeholder(tf.int64, shape=[None], name="labels")

    reg_rate = tf.placeholder(tf.float32, shape=(), name="reg_rate")
    lr = tf.placeholder(tf.float32, shape=(), name="lr")

    logits = inference(images, 3)

    count = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print("count: ", count)

    loss, loss_reg = gen_loss(labels, logits, reg_rate)
    accuracy = gen_accuracy(labels, logits)
    train_op = gen_train_op(loss_reg, lr)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        data = cifar10.Cifar10()
        test_images, test_labels = data.test_batch()
        lr_val = FLAGS.p1

        df = pd.DataFrame(columns=["datetime", "loss", "accuracy"])
        for i in range(35000):
            if i % 1000 == 0:
                loss_list, accuracy_list = [], []
                for j in range(10):
                    batch_test_images = test_images[j * 1000:(j + 1) * 1000]
                    batch_test_labels = test_labels[j * 1000:(j + 1) * 1000]
                    l, a = sess.run([loss, accuracy], feed_dict={
                        images: batch_test_images, labels: batch_test_labels})
                    loss_list.append(l)
                    accuracy_list.append(a)
                test_loss = np.mean(loss_list)
                test_accuracy = 100 * np.mean(accuracy_list)
                print("{}\tstep:{}\tloss:{:.4f}\taccuracy:{:.2f}%".format(datetime.now(), i, test_loss, test_accuracy))
                df.loc[i] = [str(datetime.now()), str(test_loss), str(test_accuracy)]

            # if i == 32000:
            #     lr_val = FLAGS.p2
            # elif i == 48000:
            #     lr_val = FLAGS.p3

            if i == 12000:
                lr_val = FLAGS.p2
            elif i == 24000:
                lr_val = FLAGS.p3

            batch_images, batch_labels = data.next_batch(128)
            sess.run(train_op, feed_dict={
                images: batch_images, labels: batch_labels, reg_rate: FLAGS.r, lr: lr_val})

        df.to_csv("./logs/csv/vgg_sgd_({})_({})_({})_({}).csv".format(FLAGS.p1, FLAGS.p2, FLAGS.p3, FLAGS.r))


if __name__ == '__main__':
    tf.app.run(main)
