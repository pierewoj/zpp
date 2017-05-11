import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

class Lenet():
    def __init__(self, x, keep_prob, conv=conv2d, device='/cpu:0', rate=1):
        inc = lambda x: np.int(np.floor(rate*x))
        with tf.device(device):
            self.x_image = tf.reshape(x, [-1,28,28,1])

            self.W_conv1 = weight_variable([5, 5, 1,inc(32)])
            self.b_conv1 = bias_variable([inc(32)])
            self.h_conv1 = tf.nn.relu(conv(self.x_image, self.W_conv1) + self.b_conv1)
            self.h_pool1 = max_pool_2x2(self.h_conv1)

            self.W_conv2 = weight_variable([5, 5, inc(32), inc(32)])
            self.b_conv2 = bias_variable([inc(32)])
            self.h_conv2 = tf.nn.relu(conv(self.h_pool1, self.W_conv2) + self.b_conv2) 
            self.h_pool2 = max_pool_2x2(self.h_conv2)

            self.W_fc1 = weight_variable([7 * 7 * inc(32), 1024])
            self.b_fc1 = bias_variable([1024])

            self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7*7*inc(32)])
            self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)

            self.h_fc1_drop = tf.nn.dropout(self.h_fc1, keep_prob)

            self.W_fc2 = weight_variable([1024, 10])
            self.b_fc2 = bias_variable([10])

            self.y = tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2
        
def train(sess, data, train_step, batch, prob, lr):
    train_step.run(session=sess, feed_dict={data[0]: batch[0], 
                                            data[1]: batch[1], 
                                            data[2]: prob, 
                                            data[3]: lr})
    
def show_acc(sess, data, accuracy, mnist, iteration):
    acc = accuracy.eval(session=sess, feed_dict={data[0]: mnist.test.images,
                                                 data[1]: mnist.test.labels, 
                                                 data[2]: 1.})
    print("step: {} accuracy: {:.4f}".format(iteration, acc))
    
def save_summaries(sess, data, train_summary, test_summary, writer, batch, mnist, iteration):
        s = train_summary.eval(session=sess, feed_dict={data[0]: batch[0],
                                                        data[1]: batch[1],
                                                        data[2]: 1})
        writer.add_summary(s, iteration)
        s = test_summary.eval(session=sess, feed_dict={data[0]: mnist.test.images,
                                                       data[1]: mnist.test.labels, 
                                                       data[2]: 1.})
        writer.add_summary(s, iteration)
        writer.flush()
        
def run(inp):

    iterations = inp['iterations']
    batch_size = inp['batch_size']
    lr = inp['lr']
    display_step = inp['display_step']
    show_step = inp['show_step']
    conv = inp['conv']
    rate = inp['rate']
    scope = inp['scope']

    mnist = input_data.read_data_sets('/tmp/dataset', one_hot=True)
    path = '/tmp/mnist/'
    writer = tf.summary.FileWriter(path)

    
    print("Running model: {}".format(scope))
    
    x = tf.placeholder(tf.float32, [None, 784])
    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)
    labels = tf.placeholder(tf.float32, [None, 10])
    
    data = [x, labels, keep_prob, learning_rate]

    net = Lenet(x, keep_prob, device='/gpu:0', conv=conv, rate=rate)
        
    predict = net.y

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=labels))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(predict,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.name_scope(scope):
        with tf.name_scope("test"):
            test_summary = tf.summary.scalar('accuracy', accuracy)

        with tf.name_scope("train"):
            train1 = tf.summary.scalar('accuracy', accuracy)
            train2 = tf.summary.scalar('loss', cross_entropy)
            train_summary = tf.summary.merge([train1, train2])

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    sess.as_default()
    sess.run(tf.global_variables_initializer())
    
    for iteration in range(iterations):
        batch = mnist.train.next_batch(batch_size)
        if iteration % show_step == 0:
            show_acc(sess, data, accuracy, mnist, iteration)
        if iteration % display_step == 0:
            save_summaries(sess, data, train_summary, test_summary, writer, batch, mnist, iteration)
        train(sess, data, train_step, batch, 0.5, lr)

    show_acc(sess, data, accuracy, mnist, iterations)
    save_summaries(sess, data, train_summary, test_summary, writer, batch, mnist, iterations)

    sess.close()

