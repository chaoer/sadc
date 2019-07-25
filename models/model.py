import tensorflow as tf
from tensorflow.layers import dense, conv2d, conv2d_transpose, max_pooling2d, flatten, batch_normalization
from tensorflow.nn import relu, tanh, leaky_relu

class Coarse:

    def __init__(self, images, params, reuse=False):
        with tf.variable_scope("Coarse", reuse=reuse) as scope:
            x = tf.reshape(images, [-1, 200, 200, 3])
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
        self.parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Coarse")
    
    def forward(self):
        return self.output
        
class Fine:

    def __init__(self, images, coarse, params, reuse=False):
        with tf.variable_scope("Fine", reuse=reuse) as scope:
            
            x = tf.reshape(images, [-1, 200, 200, 3])
            coarse = tf.reshape(coarse, [-1, 200, 200, 1])
            x = conv2d(x, 63, (9, 9), strides=(1,1), padding='same')
            x = leaky_relu(x)
            x = max_pooling2d(x, (2, 2), strides=(1, 1), padding='same')
            
            x = tf.concat([x, coarse], axis=3)
            x = conv2d(x, 64, (5, 5), strides=(1,1), padding='same')
            x = leaky_relu(x)
            
            x = conv2d(x, 1, (5, 5), strides=(1,1), padding='same')
            self.output = x      

        self.parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Fine")
    
    def forward(self):
        return self.output

class Encoder:

    def __init__(self, images, params, reuse=False):
        with tf.variable_scope("Encoder", reuse=reuse) as scope:
            x = tf.reshape(images, [-1, 200, 200, 3])
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
            x = dense(x, 200*200*3, activation=None)
            x = tf.reshape(x, [-1, 200, 200, 3])
            self.output = tf.cast(x, tf.float32)
        self.parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Encoder")
    
    def forward(self):
        return self.output

class PolyNet:

    def __init__(self, images, params, reuse=False):
        with tf.variable_scope("PolyNet", reuse=reuse) as scope:
            x = tf.reshape(images, [-1, 200, 200, 3])
            x = conv2d(x, 63, (9, 9), strides=(1,1), padding='same')
            x = leaky_relu(x)
            x = conv2d(x, 1, (5, 5), strides=(1,1), padding='same')
            self.output = x 
            self.output = tf.cast(x, tf.float32)
        self.parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="PolyNet")
    
    def forward(self):
        return self.output