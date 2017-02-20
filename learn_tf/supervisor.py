import tensorflow as tf
import numpy as np


def main(_):
    with tf.Graph().as_default() as graph:
        x = tf.placeholder(tf.float32)
        y = tf.placeholder(tf.float32)

        w = tf.get_variable('weight', shape=[], dtype=tf.float32, initializer=tf.random_normal_initializer())
        b = tf.get_variable('biase', shape=[], dtype=tf.float32, initializer=tf.zeros_initializer)

        pred = tf.multiply(x, w) + b

        loss = tf.reduce_mean(tf.square(pred - y))

        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss, global_step=global_step)

        tf.summary.scalar('weight', w)
        tf.summary.scalar('biase', b)

        sv = tf.train.Supervisor(logdir='./checkpoint')
        with sv.managed_session() as sess:
            for step in range(100000000):
                if sv.should_stop():
                    break
                train_x = np.random.randn(1)
                train_y = train_x * 10 + 20 + np.random.randn(1) * 0.1
                _, l = sess.run([train_op, loss], feed_dict={x: train_x[0], y: train_y[0]})
                if step % 1000 == 0:
                    weight, bias = sess.run([w, b])
                    print('{}: loss: {}, weight: {}, biase: {}'.format(tf.train.global_step(sess, global_step), l,
                                                                       weight, bias))


if __name__ == '__main__':
    tf.app.run()
