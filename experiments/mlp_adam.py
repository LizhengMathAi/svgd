import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


BATCH_SIZE = 32


def bn_layer(input_tensor):
    size = input_tensor.get_shape().as_list()[-1]

    mean, variance = tf.nn.moments(input_tensor, axes=[0])
    beta = tf.Variable(initial_value=tf.zeros(size, dtype=tf.float32), name="beta")
    gamma = tf.Variable(initial_value=tf.ones(size, dtype=tf.float32), name="gamma")

    return tf.nn.batch_normalization(input_tensor, mean, variance, beta, gamma, 0.001)


def fc_layer(input_tensor, out_channels):
    weights_shape = [input_tensor.get_shape().as_list()[-1], out_channels]

    weights_init = tf.truncated_normal(weights_shape, stddev=np.sqrt(2 / (weights_shape[0] + weights_shape[1])))
    weights = tf.Variable(initial_value=weights_init, dtype=tf.float32, name="weights")
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.reduce_sum(tf.abs(weights)))

    mul_tensor = tf.matmul(input_tensor, weights)

    bias = tf.Variable(initial_value=tf.zeros((weights_shape[1]), dtype=tf.float32), name="bias")
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.reduce_sum(tf.abs(bias)))
    return mul_tensor + bias


def inference(input_tensor_batch):
    with tf.name_scope('hidden1'):
        hidden1 = tf.nn.relu(bn_layer(fc_layer(input_tensor_batch, 256)))

    with tf.name_scope('hidden2'):
        hidden2 = tf.nn.relu(bn_layer(fc_layer(hidden1, 128)))

    with tf.name_scope('hidden3'):
        hidden3 = tf.nn.relu(bn_layer(fc_layer(hidden2, 64)))

    with tf.name_scope('hidden4'):
        hidden4 = tf.nn.relu(bn_layer(fc_layer(hidden3, 32)))

    with tf.name_scope('softmax_linear'):
        logits = fc_layer(hidden4, 10)
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
    return tf.train.AdamOptimizer(lr).minimize(loss)


# ===================================================================
tf.app.flags.DEFINE_float("p1", 0.001, "p1.")
tf.app.flags.DEFINE_float("p2", 0.00005, "p2.")
tf.app.flags.DEFINE_float("p3", 0.00005, "p3.")
tf.app.flags.DEFINE_float("r", 5e-5, "r.")
FLAGS = tf.app.flags.FLAGS


def main(argv):
    from datetime import datetime
    import pandas as pd

    images = tf.placeholder(tf.float32, shape=(None, 28 * 28), name="images")
    labels = tf.placeholder(tf.int64, shape=(None), name="labels")

    reg_rate = tf.placeholder(tf.float32, shape=(), name="reg_rate")
    lr = tf.placeholder(tf.float32, shape=(), name="lr")

    logits = inference(images)

    count = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print("count: ", count)

    loss, loss_reg = gen_loss(labels, logits, reg_rate)
    accuracy = gen_accuracy(labels, logits)
    train_op = gen_train_op(loss_reg, lr)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        mnist = input_data.read_data_sets("./mnist", fake_data=False)
        lr_val = FLAGS.p1

        df = pd.DataFrame(columns=["datetime", "step", "loss", "accuracy"])
        for i in range(6000):
            if i % 100 == 0:
                test_loss, test_accuracy = sess.run(
                    [loss, accuracy], feed_dict={images: mnist.test.images, labels: mnist.test.labels})
                print("{}\tstep:{}\tloss:{:.4f}\taccuracy:{:.4f}%".format(datetime.now(), i, test_loss, test_accuracy))

                df.loc[i] = [str(datetime.now()), str(i), str(test_loss), str(test_accuracy)]

            if i == 1600:
                lr_val = FLAGS.p2
            elif i == 3600:
                lr_val = FLAGS.p3

            batch_images, batch_labels = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={images: batch_images, labels: batch_labels, reg_rate: FLAGS.r, lr: lr_val})

        # df.to_csv("./logs/csv/mlp_adam_({})_({})_({})_({}).csv".format(FLAGS.p1, FLAGS.p2, FLAGS.p3, FLAGS.r))
        df.to_csv("./logs/csv/mlp_adam.csv")


if __name__ == '__main__':
    tf.app.run(main)
