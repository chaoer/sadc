import tensorflow as tf
from tensorflow.layers import dense, conv2d, conv2d_transpose, max_pooling2d, flatten, batch_normalization
from tensorflow.nn import relu, tanh, leaky_relu

class Network:

    def __init__(self, images, params, reuse=False):
        with tf.variable_scope("Network", reuse=reuse) as scope:
            x = tf.reshape(images, [-1, 200, 200, 4])
            x = conv2d(x, 64, (10, 10), strides=(4,4), padding='same')
            x = leaky_relu(x)
            x = max_pooling2d(x, (5, 5), strides=(2, 2), padding='same')
            x = conv2d(x, 128, (2,2), strides=(2,2), padding='same')
            x = leaky_relu(x)
            x = max_pooling2d(x, (2, 2), strides=(2, 2), padding='same')
            x = conv2d(x, 256, (3,3), strides=(1,1), padding='same')
            x = leaky_relu(x)
            x = conv2d(x, 512, (3,3), strides=(1,1), padding='same')
            x = leaky_relu(x)
            x = flatten(x)
            x = dense(x, 256, activation=leaky_relu)
            x = dense(x, 200*200, activation=None)
            x = tf.reshape(x, [-1, 200, 200])
            self.output = tf.cast(x, tf.float32)
        self.parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Network")
    
    def forward(self):
        return self.output