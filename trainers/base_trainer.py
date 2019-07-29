import os
import yaml
import h5py
import datetime
import numpy as np
import tensorflow as tf
from data.data_loader import DataLoader
from models.model import *
import imageio

class BaseTrainer(object):

    def __init__(self, params):

        self.params = params

        self.sess = tf.Session()

        images, sparse, maps = self.init_data().next_batch

        self.images = tf.cast(images, tf.float32)

        self.sparse = tf.cast(sparse, tf.float32)

        self.maps = tf.cast(maps, tf.float32)
        
        input_imgs = tf.concat([self.images, self.sparse], axis=3)

        self.net = Network(input_imgs, params)

        self.est_maps = tf.reshape(self.net.forward(), [-1, 200, 200])
        
        self.loss = tf.reduce_mean(tf.abs((self.est_maps - self.maps)))

        optim = tf.train.AdamOptimizer(params['learning_rate'], beta1=0.1, beta2=0.999, epsilon=1e-3)
        self.step = optim.minimize(self.loss)
        
    def init_data(self):

        with h5py.File(self.params["data_file"], 'r') as f:
            images = np.array(f["train/images"])
            depth_maps = np.array(f["train/depths"])
            sparse_maps = np.array(f["train/sparse"])
            tr_loader = DataLoader(images, sparse_maps, depth_maps, self.sess, self.params)
        return tr_loader

    def train(self):

        # Generate directory for output
        date = datetime.datetime.now()
        result_dir = "results/" + date.strftime("%a_%b_%d_%I:%M%p")
        os.mkdir(result_dir)

        # Dump configurations for current run
        config_file = open(result_dir + "/configs.yml", "w+")
        yaml.dump(self.params, config_file)
        config_file.close()

        self.sess.run(tf.global_variables_initializer())
        
        saver = tf.train.Saver()

        for i in range(self.params['num_iters']):
            
            loss, maps, ests, _ = self.sess.run([self.loss, self.maps, self.est_maps, self.step])
            
            if i % 500 == 0:
                
                _map = maps[0, :, :]
                imageio.imwrite(result_dir + "/gt" + str(i) + ".png", _map)
                _est = ests[0, :, :]
                imageio.imwrite(result_dir + "/pred" + str(i) + ".png", _est)

                save_path = saver.save(self.sess, result_dir + "/model.ckpt")
                print("Model saved in path: %s" % save_path)
