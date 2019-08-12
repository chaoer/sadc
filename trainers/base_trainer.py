import os
import yaml
import h5py
import datetime
import numpy as np
import tensorflow as tf
from data.data_loader import DataLoader

class BaseTrainer(object):

    def __init__(self, params):

        self.params = params

        self.sess = tf.Session()

        if params['optimizer'] == 'adam':
            self.optim = tf.train.AdamOptimizer(params['learning_rate'], beta1=0.1, beta2=0.999, epsilon=1e-3)
        elif params['optimizer'] == 'sgd':
            self.optim = tf.train.GradientDescentOptimizer(params['learning_rate']) 
            
        self.build()
        self.update_ops()
        
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
        self.result_dir = "results/" + date.strftime("%a_%b_%d_%I:%M%p")
        os.mkdir(self.result_dir)

        # Dump configurations for current run
        config_file = open(self.result_dir + "/configs.yml", "w+")
        yaml.dump(self.params, config_file)
        config_file.close()

        self.sess.run(tf.global_variables_initializer())
        
        saver = tf.train.Saver()

        for i in range(self.params['num_iters']):
            
            loss = self.training_step(i)
            
            if i % 10 == 0:
                print(i, loss)
            if i % 500 == 0:
                self.save_checkpoint(i)
        self.save_final()
