import os
from gymnasium.envs.registration import register


register(
    id="leaphand_gym/leaphand-v0",
    entry_point="leaphand_gym.envs:LeaphandEnvV0",
    max_episode_steps=100,
)

register(
    id="leaphand_gym/leaphand-v1",
    entry_point="leaphand_gym.envs:LeaphandEnvV1",
    max_episode_steps=100,
)