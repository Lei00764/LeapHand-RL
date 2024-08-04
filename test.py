import sys
import os
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import leaphand_gym

def main():
    env_id = "leaphand_gym/leaphand-v1"
    model_path = "./sac_leaphand.zip"

    model = SAC.load(model_path)

    def make_test_env():
        env = gym.make(env_id, render_mode="human")
        return env

    test_env = DummyVecEnv([make_test_env])

    obs = test_env.reset()


    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = test_env.step(action)
        test_env.render()
        # if dones.any():
        #     obs = test_env.reset()

    test_env.close()

if __name__ == '__main__':
    main()
