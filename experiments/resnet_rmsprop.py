import numpy as np
import tensorflow as tf


def bn_layer(input_tensor):
    size = input_tensor.get_shape().as_list()[-1]

    mean, variance = tf.nn.moments(input_tensor, axes=[0, 1, 2])
    beta = tf.Variable(initial_value=tf.zeros(size, dtype=tf.float32), name="beta")
    gamma = tf.Variable(initial_value=tf.ones(size, dtype=tf.float32), name="gamma")

    return tf.nn.batch_normalization(input_tensor, mean, variance, beta, gamma, 0.001)


def conv_layer(input_tensor, filter_shape, stride, transpose=False):
    """
    A helper function to conv, batch normalize and relu the input tensor sequentially
    :param input_tensor: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, in_channels, out_channels]
    :param stride: stride size for conv
    :param transpose: bool
    :return: 4D tensor. Y = conv2d(ReLU(batch_normalize(X))) or Y = ReLU(batch_normalize(conv2d(X)))
    """
    if transpose:
        assert input_tensor.get_shape().as_list()[-1] == filter_shape[2]

        bn_tensor = bn_layer(input_tensor)

        relu_tensor = tf.nn.relu(bn_tensor)

        kernel_init = tf.truncated_normal(filter_shape, stddev=np.sqrt(2 / (filter_shape[2] + filter_shape[3])))
        kernel = tf.Variable(initial_value=kernel_init, dtype=tf.float32, name="kernel")
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.reduce_sum(tf.abs(kernel)))

        output_tensor = tf.nn.conv2d(relu_tensor, kernel, strides=[1, stride, stride, 1], padding="SAME")
    else:
        assert input_tensor.get_shape().as_list()[-1] == filter_shape[2]

        kernel_init = tf.truncated_normal(filter_shape, stddev=np.sqrt(2 / (filter_shape[2] + filter_shape[3])))
        kernel = tf.Variable(initial_value=kernel_init, dtype=tf.float32, name="kernel")
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.reduce_sum(tf.abs(kernel)))

        conv2d_tensor = tf.nn.conv2d(input_tensor, kernel, strides=[1, stride, stride, 1], padding='SAME')

        bn_tensor = bn_layer(conv2d_tensor)

        output_tensor = tf.nn.relu(bn_tensor)
    return output_tensor


def residual_block(input_tensor, output_channel, first_block=False):
    """
    Defines a residual block in ResNet
    :param input_tensor: 4D tensor
    :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
    :param first_block: if this is the first residual block of the whole network
    :return: 4D tensor.
    """
    input_channel = input_tensor.get_shape().as_list()[-1]

    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')

    with tf.name_scope('conv_1'):
        if first_block:
            kernel_init = tf.truncated_normal(
                [3, 3, input_channel, output_channel], stddev=np.sqrt(2 / (input_channel + output_channel)))
            kernel = tf.Variable(initial_value=kernel_init, dtype=tf.float32, name="kernel")
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.reduce_sum(tf.abs(kernel)))
            conv1 = tf.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding="SAME")
        else:
            conv1 = conv_layer(input_tensor, [3, 3, input_channel, output_channel], stride, transpose=True)

    with tf.name_scope('conv_2'):
        conv2 = conv_layer(conv1, [3, 3, output_channel, output_channel], 1, transpose=True)

    if increase_dim is True:
        pooled_tensor = tf.nn.avg_pool(
            input_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        padded_tensor = tf.pad(
            pooled_tensor, [[0, 0], [0, 0], [0, 0], [input_channel // 2, input_channel // 2]])
    else:
        padded_tensor = input_tensor

    output_tensor = conv2 + padded_tensor
    return output_tensor


def flat(input_tensor):
    with tf.name_scope("flat"):
        bn_tensor = bn_layer(input_tensor)

        relu_tensor = tf.nn.relu(bn_tensor)

        output_tensor = tf.reduce_mean(relu_tensor, [1, 2])
        assert output_tensor.get_shape().as_list()[-1:] == [64]
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
    with tf.name_scope('conv0'):
        tensor = conv_layer(input_tensor_batch, [3, 3, 3, 16], 1, transpose=False)

    for i in range(n):
        with tf.name_scope('conv1_%d' % i):
            if i == 0:
                tensor = residual_block(tensor, 16, first_block=True)
            else:
                tensor = residual_block(tensor, 16)

    for i in range(n):
        with tf.name_scope('conv2_%d' % i):
            tensor = residual_block(tensor, 32)

    for i in range(n):
        with tf.name_scope('conv3_%d' % i):
            tensor = residual_block(tensor, 64)
        assert tensor.get_shape().as_list()[1:] == [8, 8, 64]

    tensor = flat(tensor)

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
    return tf.train.RMSPropOptimizer(lr).minimize(loss)


tf.app.flags.DEFINE_float("p1", 0.001, "p1.")
tf.app.flags.DEFINE_float("p2", 0.0001, "p2.")
tf.app.flags.DEFINE_float("p3", 0.0001, "p3.")
FLAGS = tf.app.flags.FLAGS


def main():
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
        for i in range(50000):
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

            if i == 32000:
                lr_val = FLAGS.p2
            elif i == 42000:
                lr_val = FLAGS.p3

            batch_images, batch_labels = data.next_batch(128)
            sess.run(train_op, feed_dict={
                images: batch_images, labels: batch_labels, reg_rate: 1e-4, lr: lr_val})

        df.to_csv("./logs/csv/resnet_rmsprop_({})_({})_({}).csv".format(FLAGS.p1, FLAGS.p2, FLAGS.p3))


main()
