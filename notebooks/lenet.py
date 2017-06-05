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