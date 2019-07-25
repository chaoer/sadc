import numpy as np
import tensorflow as tf
import h5py
from tqdm import trange


class DataLoader(object):
    def __init__(self, images, depth_maps, sess, params):
        
        self.sess = sess
        
        print(depth_maps.shape)
        print(images.shape)
        
        images = images / 255
        mean = np.mean(depth_maps, axis=(1,2), keepdims=True)
        std = np.std(depth_maps, axis=(1,2), keepdims=True)
        depth_maps = (depth_maps - mean) / std

        dm_placeholder = tf.placeholder(depth_maps.dtype, depth_maps.shape)
        images_placeholder = tf.placeholder(images.dtype, images.shape)
        dataset = tf.data.Dataset.from_tensor_slices((images_placeholder, dm_placeholder))
        dataset = dataset.shuffle(3000).repeat().batch(params["batch_size"])
        iterator = dataset.make_initializable_iterator()

        self.sess.run(iterator.initializer, feed_dict={images_placeholder: images, dm_placeholder: depth_maps})

        bimages, bdms = iterator.get_next()
        bimages.set_shape((params["batch_size"], 200, 200, 3))
        bdms.set_shape((params["batch_size"], 200, 200))
        self.next_batch = (bimages, bdms)


    def __iter__(self):
        return self.iterator()

    def iterator(self):
        while(True):
            yield self.sess.run(self.next_batch)
