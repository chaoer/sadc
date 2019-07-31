import tensorflow as tf
from tensorflow.layers import dense, conv2d, conv2d_transpose, max_pooling2d, flatten, batch_normalization
from tensorflow.nn import relu, tanh, leaky_relu

class Global:

    def __init__(self, images, sparse, params, reuse=False):
        with tf.variable_scope("Global", reuse=reuse) as scope:
            sparse = tf.reshape(sparse, [-1, 200, 200, 1])
            images = tf.reshape(images, [-1, 200, 200, 3])
            x = tf.concat([images, sparse], axis=3)
            x = tf.cast(x, dtype=tf.float32)
            x = conv2d(x, 32, (3, 3), strides=(2,2), padding='same')
            x = relu(x)
            x = conv2d(x, 64, (3,3), strides=(2,2), padding='same')
            x = relu(x)
            x = conv2d_transpose(x, 64, (5,5), strides=(2,2), padding='same', use_bias=False)
            x = batch_normalization(x)
            x = relu(x)
            x = conv2d_transpose(x, 3, (5,5), strides=(2,2), padding='same', use_bias=False)
            self.output = x     
        self.parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Global")
    
    def forward(self):
        return self.output
    
class Local:

    def __init__(self, sparse, guidance_map, params, reuse=False):
        with tf.variable_scope("Local", reuse=reuse) as scope:
            sparse = tf.reshape(sparse, [-1, 200, 200, 1])
            guidance_map = tf.reshape(guidance_map, [-1, 200, 200, 1])
            x = sparse + guidance_map
            
            x = conv2d(x, 32, (3, 3), strides=(2,2), padding='same')
            x = relu(x)
            x = conv2d(x, 64, (3,3), strides=(2,2), padding='same')
            x = relu(x)
            x = conv2d_transpose(x, 64, (5,5), strides=(2,2), padding='same', use_bias=False)
            x = batch_normalization(x)
            x = relu(x)
            x = conv2d_transpose(x, 2, (5,5), strides=(2,2), padding='same', use_bias=False)
            self.output = x     
        self.parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Local")
    
    def forward(self):
        return self.output
