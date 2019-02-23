import os.path as osp

import numpy as np
import tensorflow as tf

from baselines.common.tf_util import make_session
from model import ActorCriticModel
from utils.checkpointer import get_latest_check_num


class Agent(object):

    def __init__(self,
                 obs_space,
                 act_space,
                 ent_coef=0.01,
                 v_coef=0.5,
                 max_grad_norm=0.5):
        self.obs_space = obs_space
        self.act_space = act_space

        self._create_placeholders()
        self._create_network()

        self.act = self.dist.sample()

        # self.pi = self.dist.prob(self.act_ph)
        # self.old_pi = tf.stop_gradient(self.old_dist.prob(self.act_ph))
        self.old_log_pi = self.dist.log_prob(self.act)
        self.log_pi = self.dist.log_prob(self.act_ph)

        # self.kl = tf.reduce_mean(self.old_dist.kl_divergence(self.dist))
        self.entropy = tf.reduce_mean(self.dist.entropy())

        # ratio = self.pi / self.old_pi
        ratio = tf.exp(self.log_pi - self.old_log_pi_ph)
        self.approx_kl = tf.reduce_mean(tf.square(self.log_pi - self.old_log_pi_ph))
        min_adv = tf.where(self.adv_ph > 0, (1 + self.clip_ratio_ph) * self.adv_ph, (1 - self.clip_ratio_ph) * self.adv_ph)
        self.pi_loss = - tf.reduce_mean(tf.minimum(ratio * self.adv_ph, min_adv))
        self.v_loss = 0.5 * tf.reduce_mean((self.ret_ph - self.val)**2)

        loss = self.pi_loss - self.entropy * ent_coef + self.v_loss * v_coef
        self.pi_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pi')
        # self.old_pi_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='old_pi')
        self.v_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='v')
        trainable_params = self.pi_params + self.v_params
        grads = tf.gradients(loss, trainable_params)
        grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, trainable_params))

        trainer = tf.train.AdamOptimizer(self.lr_ph, epsilon=1e-5)
        self.train_op = trainer.apply_gradients(grads)

        # self.sync_old_pi_params_op = tf.group([tf.assign(old_params, params)\
        #                                         for old_params, params in zip(self.old_pi_params, self.pi_params)])

        self.sess = make_session()
        # self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        # self.sync_old_pi_params()

        self.saver = tf.train.Saver(max_to_keep=3)
    
    def _create_placeholders(self):
        self.lr_ph = tf.placeholder(tf.float32, shape=None)
        self.clip_ratio_ph = tf.placeholder(tf.float32, shape=None)
        self.obs_ph = tf.placeholder(tf.float32, shape=[None] + list(self.obs_space.shape))
        self.act_ph = tf.placeholder(tf.int32, shape=[None, ])
        self.adv_ph = tf.placeholder(tf.float32, shape=[None, ])
        self.ret_ph = tf.placeholder(tf.float32, shape=[None, ]) 
        self.old_log_pi_ph = tf.placeholder(tf.float32, shape=[None]) 

    def _create_network(self):
        actor_critic = ActorCriticModel(self.obs_ph, self.act_space)
        self.val, self.dist = actor_critic.output()

    def select_action(self, obs):
        act, val, log_pi = self.sess.run([self.act, self.val, self.old_log_pi], feed_dict={self.obs_ph: obs})
        return act, val, log_pi

    def get_val(self, obs):
        val = self.sess.run(self.val, feed_dict={self.obs_ph: obs})
        return val

    def train_model(self, feed_dict):
        _, pi_loss, v_loss, kl, entropy \
                = self.sess.run([self.train_op, self.pi_loss, self.v_loss, self.approx_kl, self.entropy], feed_dict=feed_dict)
        return pi_loss, v_loss, kl, entropy

    # def sync_old_pi_params(self):
    #     self.sess.run(self.sync_old_pi_params_op)

    def save_model(self, checkpoints_dir, epoch):
        self.saver.save(self.sess, osp.join(checkpoints_dir, 'tf_ckpt'), global_step=epoch)
    
    def load_model(self, checkpoints_dir, model=None):
        if model is None:
            model = get_latest_check_num(checkpoints_dir)
        self.saver.restore(self.sess, osp.join(checkpoints_dir, 'tf_ckpt-{}'.format(model)))
