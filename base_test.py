import os
import sys
import h5py
import numpy as np
from models.model import *
import tensorflow as tf

source_dir = "results/sparse100_base"

params = dict()

sess = tf.Session()

z1_dim = 128
z2_dim = 3
x_dim = 3
d_dim = 128
num_objs = 1000

data_file = 'data/old_data/data_100.h5'
with h5py.File(data_file, 'r') as f:
    images = np.array(f['test/images'], dtype=np.float32)
    sparse = np.array(f['test/sparse'], dtype=np.float32)
    depths = np.array(f['test/depths'], dtype=np.float32)

input_imgs = tf.concat([images, tf.expand_dims(sparse, axis=3)], axis=3)
net = Network(input_imgs, params)
est_maps = tf.reshape(net.forward(), [-1, 200, 200])

loss = tf.reduce_mean(tf.abs((est_maps - depths)))

sess.run(tf.global_variables_initializer())   


saver = tf.train.Saver()
saver.restore(sess, source_dir + '/model.ckpt')

test_error = loss.eval(session=sess)
print(test_error)
