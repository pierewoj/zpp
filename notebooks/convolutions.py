import tensorflow as tf

def conv2d(x, filter, strides=[1,1,1,1], padding='SAME'):
    return tf.nn.conv2d(x, filter, strides=strides, padding=padding)

def conv_bin_activ(x, filter, strides=[1,1,1,1], padding='SAME'):
    G = tf.get_default_graph()
    with G.gradient_override_map({"Sign": "Identity"}):
    	w_shape = tf.shape(filter)
    	n = tf.cast(tf.reduce_prod(w_shape[0:-1]),tf.float32) 
    	abs = tf.abs(filter)
    	a = tf.stop_gradient(tf.reduce_sum(abs, [0,1,2])/n)  
    	absX = tf.abs(x)
    	k = tf.ones(w_shape, dtype=tf.float32) / n
    	K = conv2d(absX, k, strides, padding)
    	x_sign = tf.sign(x)
    	w_sign = tf.sign(filter/a) 
    	return conv2d(x_sign, w_sign, strides, padding)*a*K

def conv_bin_weights_vector(x, filter, strides=[1,1,1,1], padding='SAME'):
    G = tf.get_default_graph()
    with G.gradient_override_map({"Sign": "Identity"}):
        w_shape = tf.shape(filter)
        n = tf.cast(tf.reduce_prod(w_shape[0:-1]),tf.float32) 
        abs = tf.abs(filter)
        a = tf.stop_gradient(tf.reduce_sum(abs, [0,1,2])/n)
        return conv2d(x, tf.sign(filter/a), strides, padding)*a

def conv_bin_weights_scalar(x, filter, strides=[1,1,1,1], padding='SAME'):
    G = tf.get_default_graph()
    with G.gradient_override_map({"Sign": "Identity"}):
        w_shape = tf.shape(filter)
        n = tf.cast(tf.reduce_prod(w_shape),tf.float32) 
        abs = tf.abs(filter)
        a = tf.stop_gradient(tf.reduce_sum(abs, [0,1,2,3])/n)
        return conv2d(x, tf.sign(filter/a), strides, padding)*a

convolutions = {
    'conv2d': conv2d,
    'conv_bin_activ': conv_bin_activ,
    'conv_bin_weights_vector': conv_bin_weights_vector,
    'conv_bin_weights_scalar': conv_bin_weights_scalar,
}
