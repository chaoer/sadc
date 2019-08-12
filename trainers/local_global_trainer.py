
import tensorflow as tf
from data.data_loader import DataLoader
from models.model import *
from trainers.base_trainer import BaseTrainer
import imageio

class LocalGlobalTrainer(BaseTrainer):

    def __init__(self, params):
        super(LocalGlobalTrainer, self).__init__(params)
        
    def build(self):

        images, sparse, maps = self.init_data().next_batch

        self.images = tf.cast(images, tf.float32)

        self.sparse = tf.cast(sparse, tf.float32)

        self.maps = tf.cast(maps, tf.float32)

        self.global_ = Global(images, sparse, self.params)

        self.global_out = self.global_.forward()
        
        self.guidance_maps = self.global_out[:, :, :, 0:1]
        global_confidence = self.global_out[:, :, :, 1:2]
        self.global_depth = self.global_out[:, :, :, 2:3]
        
        self.local = Local(self.sparse, self.guidance_maps, self.params)
        
        self.local_out = self.local.forward()
        
        local_confidence = self.local_out[:, :, :, 0:1]
        self.local_depth = self.local_out[:, :, :, 1:2]
        
        confidence_concat = tf.concat([global_confidence, local_confidence], axis=3)
        
        confidence_softmax = tf.nn.softmax(confidence_concat, axis=3)
        
        self.global_confidence = confidence_softmax[:, :, :, 0:1]
        self.local_confidence = confidence_softmax[:, :, :, 1:2]
        
        self.est_maps = tf.squeeze(self.global_confidence * self.global_depth + self.local_confidence * self.local_depth)

        self.loss = tf.reduce_mean(tf.abs((self.est_maps - self.maps))) #+ 1e-5 * tf.reduce_mean(tf.image.total_variation(tf.expand_dims(self.est_maps, axis=3)))

        self.saver = tf.train.Saver()

    def update_ops(self):
        self.step = self.optim.minimize(self.loss)
        
    def training_step(self, i):
        loss, _ = self.sess.run([self.loss, self.step])
        return loss
        
    def save_checkpoint(self, i):
        self.saver.save(self.sess, self.result_dir + "/" + str(i) + ".ckpt")
        if self.params["save_output"]:
            img, sparse, maps, ests = self.sess.run([self.images, self.sparse, self.maps, self.est_maps])
            _map = maps[0, :, :]
            imageio.imwrite(result_dir + "/gt" + str(i) + ".png", _map)
            _est = ests[0, :, :]
            imageio.imwrite(result_dir + "/pred" + str(i) + ".png", _est)
            _img = img[0, :, :, :]
            imageio.imwrite(result_dir + "/img" + str(i) + ".png", _img)
            _sparse = sparse[0, :, :]
            imageio.imwrite(result_dir + "/sparse" + str(i) + ".png", _sparse)
     

    def save_final(self):
        self.saver.save(self.sess, self.result_dir + "/final.ckpt")
        if self.params["save_output"]:
            img, sparse, maps, ests = self.sess.run([self.images, self.sparse, self.maps, self.est_maps])
            _map = maps[0, :, :]
            imageio.imwrite(result_dir + "/gt_final.png", _map)
            _est = ests[0, :, :]
            imageio.imwrite(result_dir + "/pred_final.png", _est)
            _img = img[0, :, :, :]
            imageio.imwrite(result_dir + "/img_final.png", _img)
            _sparse = sparse[0, :, :]
            imageio.imwrite(result_dir + "/sparse_final.png", _sparse)
