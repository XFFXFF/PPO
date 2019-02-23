import numpy as np
import tensorflow as tf
from tensorflow import layers
from tensorflow.distributions import Categorical


init=tf.orthogonal_initializer(np.sqrt(2))


class ActorCriticModel(object):

    def __init__(self, obs, act_space):
        with tf.variable_scope('pi'):
            x = self._cnn(obs)
            logits = layers.dense(x, units=act_space.n)
            self.dist = Categorical(logits=logits)
        with tf.variable_scope('v'):
            x = self._cnn(obs)
            self.val = tf.squeeze(layers.dense(x, units=1))

    def _cnn(self, x):
        x = tf.cast(x, tf.float32) / 255.
        x = layers.conv2d(x, filters=32, kernel_size=8, strides=(4, 4), kernel_initializer=init, activation=tf.nn.relu)
        x = layers.conv2d(x, filters=64, kernel_size=4, strides=(2, 2), kernel_initializer=init, activation=tf.nn.relu)
        x = layers.conv2d(x, filters=64, kernel_size=3, strides=(1, 1), kernel_initializer=init, activation=tf.nn.relu)
        x = layers.flatten(x)
        return layers.dense(x, units=512, kernel_initializer=init, activation=tf.nn.relu)

    def output(self):
        return self.val, self.dist