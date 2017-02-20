import os
import tensorflow as tf
import urllib

LOGDIR = '/tmp/mnist_tutorial/'
GIST_URL = 'https://gist.githubusercontent.com/dandelionmane/4f02ab8f1451e276fea1f165a20336f1/raw/a20c87f5e1f176e9abf677b46a74c6f2581c7bd8/'
### MNIST EMBEDDINGS ###
mnist = tf.contrib.learn.datasets.mnist.read_data_sets(train_dir=LOGDIR + 'data', one_hot=True)


### Get a sprite and labels file for the embedding projector ###
# urllib.urlretrieve(GIST_URL + 'labels_1024.tsv', LOGDIR + 'labels_1024.tsv')
# urllib.urlretrieve(GIST_URL + 'sprite_1024.png', LOGDIR + 'sprite_1024.png')


# 定义一个简单的卷积层
def conv_layer(input, channels_in, channels_out, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([5, 5, channels_in, channels_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[channels_out]), name="B")
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding='SAME')
        act = tf.nn.relu(conv + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 定义卷积层
def fc_layer(input, channels_in, channels_out, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([channels_in, channels_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[channels_out]), name="B")
        logit = tf.matmul(input, w) + b
        act = tf.nn.relu(logit)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
    return act


# 设置占位符
x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")
x_image = tf.reshape(x, [-1, 28, 28, 1])
tf.summary.image('input', x_image, 3)

# 创建网络
conv1 = conv_layer(x_image, 1, 32, name="conv1")
conv2 = conv_layer(conv1, 32, 64, name="conv2")

flattened = tf.reshape(conv2, [-1, 7 * 7 * 64])

fc1 = fc_layer(flattened, 7 * 7 * 64, 1024, "fc1")
logits = fc_layer(fc1, 1024, 10, "fc2")

# 计算损失函数
with tf.name_scope("xent"):
    xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    var_lists = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    l1_loss = 0.
    for var in var_lists:
        l1_loss += tf.reduce_mean(tf.abs(var))
    tf.summary.scalar('cross_entropy', xent)

# 设置训练器
with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(xent)

# 计算准确度
with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

sess = tf.Session()

merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("./mnist_demo/4")
writer.add_graph(sess.graph)

# 初始化变量
sess.run(tf.global_variables_initializer())

# 迭代训练2000步
for i in range(2000):
    batch = mnist.train.next_batch(100)

    if i % 5 == 0:
        s = sess.run(merged_summary, feed_dict={x: batch[0], y: batch[1]})
        writer.add_summary(s, i)

    if i % 5 == 0:
        [train_accuracy, train_xent] = sess.run([accuracy, xent], feed_dict={x: batch[0], y: batch[1]})
        print("step %d, training accuracy %g, xentropy %g" % (i, train_accuracy, train_xent))

    sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})
