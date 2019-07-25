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

        images, maps = self.init_data().next_batch

        images = tf.cast(images, tf.float32)

        self.images = images

        self.maps = tf.cast(maps, tf.float32)
        
        '''
        self.coarse = Coarse(images, params)

        self.coarse_maps = self.coarse.forward()

        self.fine = Fine(images, self.coarse_maps, params)

        self.est_maps = tf.reshape(self.fine.forward(), [-1, 200, 200])

        dy_hat, dx_hat = tf.image.image_gradients(tf.expand_dims(self.est_maps, axis=3))
        dy, dx = tf.image.image_gradients(tf.expand_dims(self.maps, axis=3))

        smoothness = tf.reduce_mean(tf.abs(dy_hat) * tf.exp(-1 * dy) + tf.abs(dx_hat) * tf.exp(-1 * dx)) 

        self.stage_1_loss = tf.reduce_mean(tf.abs((self.coarse_maps - self.maps)))
        self.stage_2_loss = tf.reduce_mean(tf.abs((self.est_maps - self.maps))) #+ smoothness

        optim1 = tf.train.AdamOptimizer(params['learning_rate'], beta1=0.1, beta2=0.999, epsilon=1e-3)
        optim2 = tf.train.AdamOptimizer(params['learning_rate'], beta1=0.1, beta2=0.999, epsilon=1e-3)
        self.step_1 = optim1.minimize(self.stage_1_loss, var_list=self.coarse.parameters)
        self.step_2 = optim2.minimize(self.stage_2_loss, var_list=self.fine.parameters)
        '''
        
        Enc = Encoder(self.images, params)
        codes = Enc.forward()
        
        x_0 = self.images
        
        poly1 = PolyNet(self.images, params)
        x_1 = poly1.forward()
        
        poly2 = PolyNet(x_1, params, reuse=True)
        x_2 = poly2.forward()
        
        self.est_maps = codes[:, :, 0] * x_0 + codes[:, :, 1] * x_1 + codes[:, :, 2] * x_2
        
        self.loss = tf.reduce_mean(tf.abs((self.est_maps - self.maps)))
        optim = tf.train.AdamOptimizer(params['learning_rate'], beta1=0.1, beta2=0.999, epsilon=1e-3)
        self.step = optim.minimize(self.loss)
        
    def init_data(self):

        with h5py.File(self.params["data_file"], 'r') as f:
            images = np.array(f["train/images"])
            depth_maps = np.array(f["train/depths"])
            tr_loader = DataLoader(images, depth_maps, self.sess, self.params)
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
            '''
            if i < 10000:
                loss, coarses, maps, ests, _ = self.sess.run([self.stage_1_loss, self.coarse_maps, self.maps, self.est_maps, self.step_1])
            else:
                loss, coarses, maps, ests, _ = self.sess.run([self.stage_2_loss, self.coarse_maps, self.maps, self.est_maps, self.step_2])
            if i % 10 == 0:
                print(i, loss)
            if i % 500 == 0:

                _coarse = coarses[0, :, :]
                imageio.imwrite(result_dir + "/coarse" + str(i) + ".png", _coarse)
                _map = maps[0, :, :]
                imageio.imwrite(result_dir + "/gt" + str(i) + ".png", _map)
                _est = ests[0, :, :]
                imageio.imwrite(result_dir + "/fine" + str(i) + ".png", _est)

                save_path = saver.save(self.sess, result_dir + "/model.ckpt")
                print("Model saved in path: %s" % save_path)
            '''
            loss, maps, ests, _ = self.sess.run([self.loss, self.maps, self.est_maps, self.step])
            
            if i % 500 == 0:
                
                _map = maps[0, :, :]
                imageio.imwrite(result_dir + "/gt" + str(i) + ".png", _map)
                _est = ests[0, :, :]
                imageio.imwrite(result_dir + "/pred" + str(i) + ".png", _est)

                save_path = saver.save(self.sess, result_dir + "/model.ckpt")
                print("Model saved in path: %s" % save_path)
