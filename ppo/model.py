import tensorflow as tf
from tensorflow import layers
from tensorflow.distributions import Categorical


initializer = tf.initializers.orthogonal

class ActorCriticModel(object):

    def __init__(self, obs, act_space):
        with tf.variable_scope('pi'):
            x = self._cnn(obs)
            logits = layers.dense(x, units=act_space.n)
            self.dist = Categorical(logits=logits)
        with tf.variable_scope('old_pi'):
            x = self._cnn(obs)
            logits = layers.dense(x, units=act_space.n)
            self.old_dist = Categorical(logits=logits)
        with tf.variable_scope('v'):
            x = self._cnn(obs)
            self.val = tf.squeeze(layers.dense(x, units=1))

    def _cnn(self, x):
        x = layers.conv2d(x, filters=32, kernel_size=8, strides=(4, 4), kernel_initializer=initializer, activation=tf.nn.tanh)
        x = layers.conv2d(x, filters=64, kernel_size=4, strides=(2, 2), kernel_initializer=initializer, activation=tf.nn.tanh)
        x = layers.conv2d(x, filters=64, kernel_size=3, strides=(1, 1), kernel_initializer=initializer, activation=tf.nn.tanh)
        x = layers.flatten(x)
        return layers.dense(x, units=512, activation=tf.nn.tanh)
        # x = layers.dense(x, units=64, activation=tf.nn.tanh)
        # x = layers.dense(x, units=64, activation=tf.nn.tanh)
        # return x


    def output(self):
        return self.val, self.dist, self.old_dist