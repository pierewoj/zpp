
from resnet import Resnet
import convolutions as c
import tensorflow as tf
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import time
import datetime
import os

def open_file_to_write(name):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    train_writer = tf.summary.FileWriter(os.path.join('log-resnet/',name, timestamp, 'train'))
    valid_writer = tf.summary.FileWriter(os.path.join('log-resnet/', name, timestamp, 'valid'))
    return train_writer, valid_writer


# In[2]:

from mlp.data_providers import CIFAR10DataProvider

train_batch_size = 50
valid_batch_size = 50

train_data = CIFAR10DataProvider('train', batch_size=train_batch_size)
re = train_data.inputs.reshape((40000, -1, 3), order='F')
train_data.inputs = re.reshape((40000, 32, 32, 3))

valid_data = CIFAR10DataProvider('valid', batch_size=valid_batch_size)
re = valid_data.inputs.reshape((10000, -1, 3), order='F')
valid_data.inputs = re.reshape((10000, 32, 32, 3))


# In[7]:

def train_or_test(is_test, data, writer, name, sess, e):
    running_error = 0.
    running_accuracy = 0.
    start = time.time()
    for b, (input_batch, target_batch) in enumerate(data):
        if is_test:
            batch_error, batch_acc, summary = sess.run(
                [error, accuracy, summary_op], 
                feed_dict={net.inputs: input_batch, net.targets: target_batch})
        else:
            _, batch_error, batch_acc, summary = sess.run(
                [train_step, error, accuracy, summary_op], 
                feed_dict={net.inputs: input_batch, net.targets: target_batch})

        running_error += batch_error
        running_accuracy += batch_acc

        writer.add_summary(summary, e * data.num_batches + b)
                
    running_error /= data.num_batches
    running_accuracy /= data.num_batches
    if(e+1) % 5 == 0:
        print('End of epoch {0:02d}: err({4})={1:.2f} acc({4})={2:.2f}, time({4})={3:.2f}'
            .format(e + 1, running_error, running_accuracy, time.time() - start, name))


# In[4]:

def run_session(net, name, learning_rate, num_epoch):
    global train_step
    global error
    global accuracy
    global summary_op
    
    train_writer, valid_writer = open_file_to_write('bla')
    init = tf.global_variables_initializer()
    
    with tf.name_scope('error'):
        error = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(net.outputs, net.targets))

    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(
                tf.equal(tf.argmax(net.outputs, 1), tf.argmax(net.targets, 1)), 
                tf.float32))

    with tf.name_scope('train'):
        opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        train_step = opt.minimize(error)
    
    tf.summary.scalar('error', error)
    tf.summary.scalar('accuracy', accuracy)
    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        valid_inputs = valid_data.inputs
        valid_targets = valid_data.to_one_of_k(valid_data.targets)
        init = tf.global_variables_initializer()
        sess.run(init)
        for e in range(num_epoch):
            train_or_test(False, train_data, train_writer, 'train', sess, e)
            if (e + 1) % 5 == 0:
                train_or_test(True, valid_data, valid_writer, 'valid', sess, e)


# In[5]:

def experiment(num_blocks, conv, learning_rate, rate, epochs, name):
    global net
    print 'Resnet ' + `num_blocks` + '-blocks, conv=' + conv.__name__ + ', learning rate=' + `learning_rate` + ', rate=' + `rate` + '\n'
    tf.reset_default_graph()
    net = None
    net = Resnet(n=num_blocks, conv=conv, rate=rate)
    run_session(net, name, learning_rate, epochs)
    print ''

