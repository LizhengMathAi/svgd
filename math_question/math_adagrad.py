import numpy as np
import pandas as pd
import tensorflow as tf

tf.set_random_seed(0)
np.random.seed(0)
np.set_printoptions(precision=5, linewidth=120, suppress=True)


mat_val = np.array([[1 / (i + j + 1) for i in range(10)] for j in range(10)])
rhs_val = 0.05 * np.ones([10, 1])


mat = tf.constant(value=mat_val, dtype=tf.float32)
rhs = tf.constant(value=rhs_val, dtype=tf.float32)
x = tf.Variable(initial_value=tf.zeros_like(rhs), dtype=tf.float32)

loss = tf.reduce_sum(tf.square(tf.matmul(mat, x) - rhs))
train_op = tf.train.AdagradOptimizer(1e-2).minimize(loss)


with tf.Session() as sess:
    tf.global_variables_initializer().run()

    df = pd.DataFrame(columns=["step", "loss"])
    for i in range(100):
        if i % 5 == 0:
            loss_val = sess.run(loss)
            df.loc[i] = [str(i), str(loss_val)]
            print("step:{}\tloss:{}".format(i, loss_val))
        sess.run(train_op)

    df.to_csv("./logs/csv/math_adagrad.csv")
