import time
from collections import deque

import gym
import numpy as np
import tensorflow as tf

from ppo.agent import Agent
from baselines.common.atari_wrappers import WarpFrame, wrap_deepmind
from baselines.common.cmd_util import make_atari
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from ppo.buffer import Buffer
from ppo.utils.logx import EpochLogger
from ppo.utils.schedules import PiecewiseSchedule
from ppo.utils.wrappers import LogWrapper


def create_env(env_id, n_env, seed):
    def make_env(rank):
        def _thunk():
            env = make_atari(env_id)
            env.seed(seed + rank)
            env = LogWrapper(env)
            return wrap_deepmind(env)
        return _thunk
    env = SubprocVecEnv([make_env(i) for i in range(n_env)])
    env = VecFrameStack(env, 4)
    return env


class Runner(object):

    def __init__(self,
                 epochs,
                 env_id,
                 n_env,
                 seed,
                 gamma=0.99,
                 int_gamma=0.99,
                 lam=0.95,
                 train_epoch_len=128,
                 test_epoch_len=2000,
                 logger_kwargs=dict()):

        self.epochs = epochs
        self.env_id = env_id
        self.n_env = n_env
        self.train_epoch_len = train_epoch_len
        self.test_epoch_len = test_epoch_len
        self.logger_kwargs = logger_kwargs

        self.checkpoints_dir = self.logger_kwargs['output_dir'] + '/checkpoints'
        
        tf.set_random_seed(seed)
        np.random.seed(seed)
        self.env = create_env(env_id, n_env, seed)

        self.lr_schedule = PiecewiseSchedule(
            [
                (0, 2.5e-4),
                (2e6, 1e-4),
                (5e6, 5e-5)
            ], outside_value=5e-5,
        )

        self.clip_ratio_schedule = PiecewiseSchedule(
            [
                (0, 0.1),
                (2e6, 0.05)
            ], outside_value=0.05,
        )

        self.obs = self.env.reset()
        self.ep_info_buf = deque(maxlen=100)

        self.obs_space = self.env.observation_space
        self.act_space = self.env.action_space

        self.t = 0

        self.agent = Agent(self.obs_space, self.act_space)
        self.buffer = Buffer(gamma, lam)
    
    def _collect_rollouts(self, logger):
        for step in range(self.train_epoch_len):
            acts, vals, log_pis = self.agent.select_action(self.obs)
            logger.store(Val=vals)
            next_obs, rews, dones, infos = self.env.step(acts)
            self.t += self.n_env
            self.buffer.store(self.obs, acts, rews, dones, vals, log_pis)
            self.obs = next_obs
            for info in infos:
                if info.get('ep_r'):
                    self.ep_info_buf.append(info)
        last_vals= self.agent.get_val(self.obs)
        return last_vals

    def _run_train_phase(self, logger):
        start_time = time.time()
        last_vals = self._collect_rollouts(logger)
        obs_buf, act_buf, ret_buf, adv_buf, log_pi_buf, val_buf = self.buffer.get(last_vals)
        lr = self.lr_schedule.value(self.t)
        clip_ratio = self.clip_ratio_schedule.value(self.t)
        sample_range = np.arange(len(act_buf))
        for i in range(3):
            np.random.shuffle(sample_range)
            for j in range(int(len(act_buf) / 128)):
                sample_idx = sample_range[128 * j: 128 * (j + 1)]
                feed_dict = {
                    self.agent.lr_ph: lr,
                    self.agent.clip_ratio_ph: clip_ratio, 
                    self.agent.obs_ph: obs_buf[sample_idx],
                    self.agent.act_ph: act_buf[sample_idx],
                    self.agent.ret_ph: ret_buf[sample_idx],
                    self.agent.adv_ph: adv_buf[sample_idx],
                    self.agent.old_log_pi_ph: log_pi_buf[sample_idx],
                    self.agent.old_v_ph: val_buf[sample_idx]
                }
                pi_loss, v_loss, kl, entropy = self.agent.train_model(feed_dict)
                logger.store(PiLoss=pi_loss, VLoss=v_loss)
                logger.store(KL=kl, Entropy=entropy)
            
    def run_experiment(self):
        start_time = time.time()
        logger = EpochLogger(**self.logger_kwargs)
        for epoch in range(1, self.epochs + 1):
            self._run_train_phase(logger)
            self.agent.save_model(self.checkpoints_dir, epoch)
            ep_ret_list = [ep_info['ep_r'] for ep_info in self.ep_info_buf]
            ep_len_list = [ep_info['ep_len'] for ep_info in self.ep_info_buf]
            ep_ret_mean, ep_ret_std, ep_ret_min, ep_ret_max = logger.get_statistics_scalar(ep_ret_list, with_min_and_max=True)
            ep_len_mean, ep_len_std, ep_len_min, ep_len_max = logger.get_statistics_scalar(ep_len_list, with_min_and_max=True)
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRetMean', ep_ret_mean)
            logger.log_tabular('EpRetStd', ep_ret_std)
            logger.log_tabular('EpRetMin', ep_ret_min)
            logger.log_tabular('EpRetMax', ep_ret_max)
            logger.log_tabular('EpLenMean', ep_len_mean)
            logger.log_tabular('Val', average_only=True)
            logger.log_tabular('KL', average_only=True)
            logger.log_tabular('Entropy', average_only=True)
            logger.log_tabular('PiLoss', average_only=True)
            logger.log_tabular('VLoss', average_only=True)
            logger.log_tabular('LearningRate', self.lr_schedule.value(self.t))
            logger.log_tabular('ClipRatio', self.clip_ratio_schedule.value(self.t))
            logger.log_tabular('TotalInteractions', self.t)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()

    def _run_test_phase(self, logger, render=True):
        env = create_env(self.env_id, 1, 0)
        ep_r, ep_len = 0, 0
        obs = env.reset()
        for step in range(self.test_epoch_len):
            if render: env.render()
            act = self.agent.select_action(obs)
            next_obs, reward, done, info = env.step(act)
            # time.sleep(0.1)
            ep_r += reward
            ep_len += 1
            obs = next_obs
            
            if done:
                logger.store(TestEpRet=ep_r, TestEpLen=ep_len)

                obs = env.reset()
                ep_r, ep_len = 0, 0

    def run_test_and_render(self, model):
        logger = EpochLogger()
        self.agent.load_model(self.checkpoints_dir, model=model)
        for epoch in range(1, self.epochs + 1):
            self._run_test_phase(logger)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--env', type=str, default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--n_env', '-n', type=int, default=32)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--model', type=int, default=None)
    parser.add_argument('--exp_name', type=str, default='')
    args = parser.parse_args()

    from ppo.utils.run_utils  import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.env, args.seed, exp_name=args.exp_name)

    runner = Runner(args.epochs, args.env, args.n_env, args.seed, logger_kwargs=logger_kwargs)
    if args.test:
        runner.run_test_and_render(args.model)
    else:
        runner.run_experiment()
