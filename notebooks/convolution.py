import tensorflow as tf

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv_bin_activ(x, W):
    G = tf.get_default_graph()
    with G.gradient_override_map({"Sign": "Identity"}):
                w_shape = tf.shape(W)
                n = tf.cast(tf.reduce_prod(w_shape[0:-1]),tf.float32) 
                abs = tf.abs(W)
                a = tf.stop_gradient(tf.reduce_sum(abs, [0,1,2])/n)  
                absX = tf.abs(x)
                k = tf.ones(w_shape, dtype=tf.float32) / n
                K = conv2d(absX, k)
                x_sign = tf.sign(x)
                w_sign = tf.sign(W/a)              
                return conv2d(x_sign, w_sign)*a*K

def conv_bin_weights_vector(x, W):
    G = tf.get_default_graph()
    with G.gradient_override_map({"Sign": "Identity"}):
                w_shape = tf.shape(W)
                n = tf.cast(tf.reduce_prod(w_shape[0:-1]),tf.float32) 
                abs = tf.abs(W)
                a = tf.stop_gradient(tf.reduce_sum(abs, [0,1,2])/n)
                return conv2d(x, tf.sign(W/a))*a

def conv_bin_weights_scalar(x, W):
    G = tf.get_default_graph()
    with G.gradient_override_map({"Sign": "Identity"}):
                w_shape = tf.shape(W)
                n = tf.cast(tf.reduce_prod(w_shape),tf.float32) 
                abs = tf.abs(W)
                a = tf.stop_gradient(tf.reduce_sum(abs, [0,1,2,3])/n)
                return conv2d(x, tf.sign(W/a))*a

convolutions = {
    'conv2d': conv2d,
    'conv_bin_activ': conv_bin_activ,
    'conv_bin_weights_vector': conv_bin_weights_vector,
    'conv_bin_weights_scalar': conv_bin_weights_scalar,
}

