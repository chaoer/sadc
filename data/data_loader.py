import numpy as np
import tensorflow as tf
import h5py
from tqdm import trange


class DataLoader(object):
    def __init__(self, images, sparse_maps, depth_maps, sess, params):
        
        self.sess = sess
        
        images = images / 255

        dm_placeholder = tf.placeholder(depth_maps.dtype, depth_maps.shape)
        sm_placeholder = tf.placeholder(sparse_maps.dtype, sparse_maps.shape)
        images_placeholder = tf.placeholder(images.dtype, images.shape)
        dataset = tf.data.Dataset.from_tensor_slices((images_placeholder, sm_placeholder, dm_placeholder))
        dataset = dataset.shuffle(3000).repeat().batch(params["batch_size"])
        iterator = dataset.make_initializable_iterator()

        self.sess.run(iterator.initializer, feed_dict={images_placeholder: images, sm_placeholder: sparse_maps, dm_placeholder: depth_maps})

        bimages, bsms, bdms = iterator.get_next()
        bimages.set_shape((params["batch_size"], 200, 200, 3))
        bsms.set_shape((params["batch_size"], 200, 200))
        bdms.set_shape((params["batch_size"], 200, 200))
        self.next_batch = (bimages, bsms, bdms)


    def __iter__(self):
        return self.iterator()

    def iterator(self):
        while(True):
            yield self.sess.run(self.next_batch)
