import os
import tensorflow as tf
import numpy as np
import time
import datetime

weight_decay = 0.0002

BN_EPSILON=0.001

def fully_connected_layer(inputs, input_dim, output_dim, nonlinearity=tf.nn.relu):
    weights = tf.Variable(
        tf.truncated_normal(
            [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5), 
        'weights')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    outputs = nonlinearity(tf.matmul(inputs, weights) + biases)
    return outputs

def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer()):
    
    regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)
   
    variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                    regularizer=regularizer)
    return variables


def batch_normalization_layer(input_layer, dimension):
    
    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    beta = tf.get_variable('beta', dimension, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable('gamma', dimension, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))
    bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)

    return bn_layer

def bn_relu_conv_layer(input_layer, filter_shape, stride, conv):

    in_channel = input_layer.get_shape().as_list()[-1]

    bn_layer = batch_normalization_layer(input_layer, in_channel)
    relu_layer = tf.nn.relu(bn_layer)

    filter = create_variables(name='conv', shape=filter_shape)
    conv_layer = conv(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    return conv_layer

def residual_block(input_layer, output_channel, conv, first_block=False):
    
    input_channel = input_layer.get_shape().as_list()[-1]

    if input_channel * 2 == output_channel:
        stride = 2
        increase_dim = True
    elif input_channel == output_channel:
        stride = 1
        increase_dim = False
    else:
        raise ValueError('Wrong input or output dimension')

    with tf.variable_scope('conv1_in_block'):
        if first_block:
            filter = create_variables(name='conv', shape=[3, 3, input_channel, output_channel])
            conv1 = conv(input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
        else:
            conv1 = bn_relu_conv_layer(input_layer, [3, 3, input_channel, output_channel], stride, conv)

    with tf.variable_scope('conv2_in_block'):
        conv2 = bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel], 1, conv)

    if increase_dim is True:
        pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='VALID')
        shortcut = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                     input_channel // 2]])
    else:
        shortcut = input_layer

    output = conv2 + shortcut
    return output


def inc(x, rate):
        y = np.int(np.floor(rate*x))
        if y % 2 == 1:
            y = y+1
        return y   

class Resnet():
    #num layers = 6n + 2
    def __init__(self, n, conv=tf.nn.conv2d, rate=1.0, IMG_HEIGHT=32,
                    IMG_WIDTH=32, IMG_DEPTH=3, NUM_CLASSES=10, train_batch_size=50):
        tf.reset_default_graph()
        self.inputs = tf.placeholder(tf.float32,[train_batch_size, IMG_HEIGHT,
                                                            IMG_WIDTH, IMG_DEPTH], 'inputs')
        self.targets = tf.placeholder(tf.float32, [None, NUM_CLASSES], 'targets')

        num_filters = inc(16,rate)
        layers = []
        
        with tf.variable_scope('conv0'):
            #first layer needs to handle input and does not need relu
            filter = create_variables(name='conv', shape=[3, 3, 3, num_filters])
            conv0 = conv(self.inputs, filter, strides=[1, 1, 1, 1], padding='SAME')
            norm_conv0 = batch_normalization_layer(conv0, num_filters) 
            layers.append(norm_conv0)

        for i in range(n):
            with tf.variable_scope('conv1_%d' %i):
                if i == 0:
                    conv11 = residual_block(layers[-1], num_filters, conv, first_block=True)
                else:
                    conv11 = residual_block(layers[-1], num_filters, conv)
                layers.append(conv11)

        for i in range(n):
            with tf.variable_scope('conv2_%d' %i):
                conv2 = residual_block(layers[-1], num_filters*2, conv)
                layers.append(conv2)

        for i in range(n):
            with tf.variable_scope('conv3_%d' %i):
                conv3 = residual_block(layers[-1], num_filters*4, conv)
                layers.append(conv3)

        with tf.variable_scope('fullyc'):
            in_channel = layers[-1].get_shape().as_list()[-1]
            bn_layer = batch_normalization_layer(layers[-1], in_channel)
            relu_layer = tf.nn.relu(bn_layer)
            global_pool = tf.reduce_mean(relu_layer, [1, 2])

            self.outputs = fully_connected_layer(global_pool, num_filters*4, NUM_CLASSES, tf.identity)
