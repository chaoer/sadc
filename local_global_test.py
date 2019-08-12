import os
import sys
import h5py
import numpy as np
from models.model import *
import tensorflow as tf

source_dir = "results/lg_no_tv_100"

params = dict()

sess = tf.Session()

data_file = 'data/data_100.h5'
with h5py.File(data_file, 'r') as f:
    images = np.array(f['test/images'], dtype=np.float32)[0:100]
    sparse = np.array(f['test/sparse'], dtype=np.float32)[0:100]
    depths = np.array(f['test/depths'], dtype=np.float32)[0:100]

global_ = Global(images, sparse, params)

global_out = global_.forward()
        
guidance_maps = global_out[:, :, :, 0:1]
global_confidence = global_out[:, :, :, 1:2]
global_depth = global_out[:, :, :, 2:3]
        
local = Local(sparse, guidance_maps, params)
        
local_out = local.forward()
        
local_confidence = local_out[:, :, :, 0:1]
local_depth = local_out[:, :, :, 1:2]
        
confidence_concat = tf.concat([global_confidence, local_confidence], axis=3)
        
confidence_softmax = tf.nn.softmax(confidence_concat, axis=3)
        
global_confidence = confidence_softmax[:, :, :, 0:1]
local_confidence = confidence_softmax[:, :, :, 1:2]
        
est_maps = tf.squeeze(global_confidence * global_depth + local_confidence * local_depth)

loss = tf.reduce_mean(tf.abs((est_maps - depths)))

sess.run(tf.global_variables_initializer())   


saver = tf.train.Saver()
saver.restore(sess, source_dir + '/final.ckpt')

test_error = loss.eval(session=sess)
print(test_error)
