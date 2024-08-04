import sys
import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.env_util import make_vec_env

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import leaphand_gym


class TensorboardCallback(BaseCallback):
    def __init__(self, reward_freq = 0 ,verbose=0):
        super().__init__(verbose=verbose)
        info_keywords = ["approach_reward", "distance_reward", "survival_reward", "total_reward"]
        self.info_keywords = info_keywords
        self.rollout_info = {}
        self.reward_freq = reward_freq
        self.n_rollout = 1

    def _on_rollout_start(self):
        self.rollout_info = {key: [] for key in self.info_keywords}

    def _on_step(self):
        for key in self.info_keywords:
            vals = [info[key] for info in self.locals["infos"]]
            self.rollout_info[key].extend(vals)
        return True

    def _on_rollout_end(self):
        if self.reward_freq != 0 and self.n_rollout % self.reward_freq == 0:
            for key in self.info_keywords:
                self.logger.record("reward/" + key, np.mean(self.rollout_info[key]))
        self.n_rollout +=1


class SaveModelCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=1):
        super(SaveModelCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # get latest model path
            log_name = self.logger.get_dir().split(os.sep)[-1]
            model_path = os.path.join(self.save_path, log_name, f'model_{self.n_calls}_steps')
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Model saved at step {self.n_calls} to {model_path}")
        return True

def make_env(env_id, rank):
    def _init():
        env = gym.make(env_id, render_mode="human")
        env = Monitor(env)  # Use Monitor wrapper to record episode statistics
        return env
    return _init

def main():
    env_id = "leaphand_gym/leaphand-v1"
    num_cpu = 1
    save_path = "./sac_leaphand_tensorboard/"
    save_freq = 10000
    reward_freq = 1000

    envs = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    model = SAC("MlpPolicy", envs, verbose=1, tensorboard_log="./sac_leaphand_tensorboard/", seed=42)

    callbacks = [TensorboardCallback(reward_freq), SaveModelCallback(save_freq, save_path)]

    model.learn(total_timesteps=10000000, progress_bar=True, callback=callbacks)

    model.save("sac_leaphand")


if __name__ == '__main__':
    main()
