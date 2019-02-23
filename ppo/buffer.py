import numpy as np


class Buffer(object):

    def __init__(self, gamma, lam):
        self.gamma = gamma
        self.lam = lam
        self.obs_buf, self.act_buf, self.rew_buf, self.done_buf, self.val_buf, self.log_pi_buf = [], [], [], [], [], []

    def store(self, obs, act, rew, done, val, log_pi):
        self.obs_buf.append(obs)
        self.act_buf.append(act)
        self.rew_buf.append(rew)
        self.done_buf.append(done)
        self.val_buf.append(val)
        self.log_pi_buf.append(log_pi)
    
    def get(self, last_val):
        self.val_buf.append(last_val)
        obs_buf = np.asarray(self.obs_buf, dtype=np.float32)
        act_buf = np.asarray(self.act_buf, dtype=np.int32)
        rew_buf = np.asarray(self.rew_buf, dtype=np.float32)
        done_buf = np.asarray(self.done_buf, dtype=np.bool)
        val_buf = np.asarray(self.val_buf, dtype=np.float32)
        log_pi_buf = np.asarray(self.log_pi_buf, dtype=np.float32)

        adv_buf = np.zeros_like(rew_buf)
        last_gae_lam, last_ret = 0, last_val
        for i in reversed(range(len(self.rew_buf))):
            next_not_done = 1.0 - done_buf[i]
            delta = rew_buf[i] + self.gamma * val_buf[i+1] * next_not_done - val_buf[i]
            adv_buf[i] = last_gae_lam = delta + self.gamma * self.lam * next_not_done * last_gae_lam
            # ret_buf[i] = last_ret =  rew_buf[i] + self.gamma * last_ret * (1 - done_buf[i])
        ret_buf = adv_buf + val_buf[:-1]
        obs_buf, act_buf, ret_buf, adv_buf, log_pi_buf, val_buf = map(self.swap_and_flatten, (obs_buf, act_buf, ret_buf, adv_buf, log_pi_buf, val_buf[:-1]))

        adv_buf = (adv_buf - np.mean(adv_buf)) / (np.std(adv_buf) + 1e-8)

        self.obs_buf, self.act_buf, self.rew_buf, self.done_buf, self.val_buf, self.log_pi_buf = [], [], [], [], [], []

        return obs_buf, act_buf, ret_buf, adv_buf, log_pi_buf, val_buf
    
    def swap_and_flatten(self, arr):
        shape = arr.shape
        return arr.swapaxes(0, 1).reshape(shape[0]*shape[1], *shape[2:])
