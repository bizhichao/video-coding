import tensorflow as tf
import numpy as np
import os, sys, inspect
import time

from tensorflow.contrib import layers
from tensorflow.contrib import slim

flags = tf.flags
tf.flags.DEFINE_integer("batch_size", 64, "batch size for trainning")
tf.flags.DEFINE_string("log_dir", "./checkpoint/gan", "direcotory of save checkpoint and summaries")
tf.flags.DEFINE_string("data_dir", "Data_zoo/CelebA_faces/", "path to dataset")

tf.flags.DEFINE_integer("model", "0", "Model to train. 0 - GAN, 1 - WassersteinGAN")
tf.flags.DEFINE_integer("z_dim", "100", "size of input vector to generator")
tf.flags.DEFINE_string("image_size", "108,64", "Size of actual images, Size of images to be generated at.")

tf.flags.DEFINE_string("optimizer", "Adam", "Optimizer to use for training")
tf.flags.DEFINE_float("learning_rate", "0.01", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_integer("n_epochs", 10, "number of epochs for training")

tf.flags.DEFINE_integer("gen_dimension", "16", "dimension of first layer in generator")
tf.flags.DEFINE_string("mode", "train", "train / visualize model")

FLAGS = flags.FLAGS


class Generator(object):
    def __init__(self, image_size, dims, activation=tf.nn.relu, scope=None):
        self._image_size = image_size
        self._dims = dims
        self._activation = activation
        self._scope = scope or 'Generator'

    def __call__(self, z, train_phase=True, reuse=None):
        with tf.variable_scope(self._scope, reuse=reuse) as scope:
            self._scope_name = scope.name
            N = len(self._dims)
            feat_size = self._image_size // (2 ** (N - 1))
            h_z = tf.layers.dense(z, self._dims[0] * feat_size * feat_size, activation=None, name='dense_1')
            h_z = tf.reshape(h_z, [-1, feat_size, feat_size, self._dims[0]])
            h_bnz = tf.layers.batch_normalization(h_z, training=train_phase)
            h = self._activation(h_bnz)

            for index in range(1, N - 1):
                feat_size *= 2
                h_conv_t = tf.layers.conv2d_transpose(h, self._dims[index], [5, 5], strides=[2, 2], padding='SAME')
                h_bn = tf.layers.batch_normalization(h_conv_t, training=train_phase)
                h = self._activation(h_bn)

            feat_size *= 2
            assert feat_size == self._image_size
            h_conv_t = tf.layers.conv2d_transpose(h, self._dims[-1], [5, 5], strides=[2, 2], padding='SAME')
            gen_image = tf.nn.tanh(h_conv_t)
        return gen_image

    def scope(self):
        return self._scope_name


class Discriminator(object):
    def __init__(self, image_size, dims, activation=tf.nn.relu, scope=None):
        self._image_size = image_size
        self._dims = dims
        self._activation = activation
        self._scope = scope or 'Discriminator'

    def __call__(self, image, train_phase=True, reuse=None):
        with tf.variable_scope(self._scope, reuse=reuse) as scope:
            self._scope_name = scope.name
            N = len(self._dims)
            h = image
            skip_bn = True
            for index in range(1, N - 1):
                h_conv = tf.layers.conv2d(h, self._dims[index], [5, 5], strides=[2, 2], padding='SAME')
                if skip_bn:
                    h_bn = h_conv
                    skip_bn = False
                else:
                    h_bn = tf.layers.batch_normalization(h_conv, training=train_phase)
                h = self._activation(h_bn)

            h_logits = tf.layers.dense(h, self._dims[-1], activation=None)

        return tf.nn.sigmoid(h_logits), h_logits, h

    def scope(self):
        return self._scope_name


class GAN(object):
    def __init__(self, image_size, z_dim, generator, discriminator, scope=None):
        self.image_size = image_size
        self.z_dim = z_dim
        self.generator = generator
        self.discriminator = discriminator
        self._scope = scope

    def build_network(self, optimizer="Adam", learning_rate=0.01, improve_loss=True):
        print("setting up model...")
        with tf.variable_scope(self._scope or "GAN") as scope:
            self._scope_name = scope.name
            self.train_phase = tf.placeholder(tf.bool)

            self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3])
            tf.summary.image("real_images", self.images, max_outputs=5)

            self.z_vec = tf.placeholder(tf.float32, [None, self.z_dim])
            tf.summary.histogram("z", self.z_vec)

            self.gen_images = self.generator(self.z_vec, train_phase=self.train_phase)
            self.gen_images = self.gen_images * 127.5 + 127.5
            tf.summary.image("gen_images", self.gen_images, max_outputs=5)

            img = (self.images - 127.5) / 127.5
            gen_img = (self.gen_images - 127.5) / 127.5
            prob_real, logits_real, feature_real = self.discriminator(img, self.train_phase, reuse=None)
            prob_fake, logits_fake, feature_fake = self.discriminator(gen_img, self.train_phase, reuse=True)

            # Loss calculation
            self.disc_loss, self.gen_loss = self._gan_loss(logits_real, logits_fake, improve_loss,
                                                           feature_real, feature_fake)

            self.gen_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.generator.scope())
            self.disc_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.discriminator.scope())
            self.global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._scope_name)
            self.local_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=self._scope_name)

            if optimizer == 'Adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate)
            elif optimizer == 'RMSProp':
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate)
            else:
                raise ValueError("Unknown optimizer %s" % optimizer)

            self.gen_train_op = self.optimizer.minimize(self.gen_loss, self.gen_variables)
            self.disc_train_op = self.optimizer.minimize(self.disc_loss, self.disc_variables)

            self.init_op = tf.initialize_variables(self.global_variables + self.local_variables)
            summary_ops = tf.get_collection(tf.GraphKeys.SUMMARY_OP, scope=self._scope_name)
            self.summary_op = None if not summary_ops else tf.summary.merge(summary_ops)

    def _gan_loss(self, logits_real, logits_fake, use_features=False, feature_real=None, feature_fake=None):

        disc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits_real, tf.ones_like(logits_real)))
        disc_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits_fake, tf.zeros_like(logits_fake)))
        tf.summary.scalar('disc_loss_real', disc_loss_real)
        tf.summary.scalar('disc_loss_fake', disc_loss_fake)

        disc_loss = disc_loss_fake + disc_loss_real
        tf.summary.scalar('disc_loss', disc_loss)

        # generator loss use d-trick
        gen_loss_disc = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits_fake, tf.ones_like(logits_fake)))
        tf.summary.scalar('gen_loss_disc', gen_loss_disc)

        if use_features:
            gen_loss_feat = tf.reduce_mean(tf.nn.l2_loss(feature_real - feature_fake)) / (self.image_size ** 2)
            tf.summary.scalar('gen_loss_feat', gen_loss_feat)
        else:
            gen_loss_feat = 0

        gen_loss = gen_loss_disc + 0.1 * gen_loss_feat
        tf.summary.scalar("gen_loss", gen_loss)
        return disc_loss, gen_loss
