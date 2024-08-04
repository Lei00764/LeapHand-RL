import numpy as np
import os
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 2.0, 0.0)),
    "elevation": -20.0,
}


def mass_center(model, data):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()


class LeaphandEnvV0(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }
    mj_xml_path = os.path.join(os.path.dirname(__file__), "..", "..", "mj_model", "mjmodelV0.xml")

    def __init__(
        self,
        approach_reward=1,
        healthy_reward=1,
        terminate_when_unhealthy=True,
        healthy_z_range=(-100, 100),
        reset_noise_scale=1e-2,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            approach_reward,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            reset_noise_scale,
            **kwargs,
        )
        self._approach_reward = approach_reward
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._survival_reward = 1 # 存活奖励


        # Observation Dim is in function _get_obs()
        observation_space = Box(
            low=-5, high=5, shape=(22,), dtype=np.float64
        )
        MujocoEnv.__init__(
            self,
            LeaphandEnvV0.mj_xml_path,
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )
        print("observation space shape: ", self.observation_space.shape)
        print("action space shape: ", self.action_space.shape)
        
    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        print("actuator number: ", low.shape)
        self.action_space = Box(low=low, high=high, dtype=np.float32)
        return self.action_space
    
    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(self.data.ctrl))
        return control_cost

    @property
    def is_healthy(self):
        # 死亡判断条件：
        # 手指偏离圆柱体表面
        f  = self.data.body("finger_end").xpos  # 大拇指  
        f1 = self.data.body("finger_end_1").xpos
        f2 = self.data.body("finger_end_2").xpos

        cylinder_pos = self.data.body("cylinder").xpos
        cylinder_pos += np.array([0.03, 0, -0.03]) # xpos 对应没有凹陷一面的左上角
        # print(f, cylinder_pos)
        # print(f1, cylinder_pos)
        # print(f2, cylinder_pos)
        # print(target_pos, cylinder_pos)

        # 计算手指末端到圆柱体表面的最短距离
        def distance_to_cylinder_surface(point, cylinder_center, cylinder_radius=0.03):
            px, py, pz = point
            cx, cy, cz = cylinder_center
            
            # 水平距离只考虑XZ平面
            horizontal_distance = np.linalg.norm([px - cx, pz - cz])
            # print(horizontal_distance)
            surface_distance = abs(horizontal_distance - cylinder_radius)
            return surface_distance
        
        f_d  = distance_to_cylinder_surface(f, cylinder_pos)
        f1_d = distance_to_cylinder_surface(f1, cylinder_pos)
        f2_d = distance_to_cylinder_surface(f2, cylinder_pos)
        # print(f_d, f1_d, f2_d)
        threshold = 0.034

        is_healthy =  f_d < threshold + 0.02 and (f1_d < threshold) and (f2_d < threshold)
    
        return is_healthy

    @property
    def terminated(self):
        terminated = (not self.is_healthy) if self._terminate_when_unhealthy else False
        return terminated

    def _get_obs(self):
        # dimesion=22
        position = self.data.qpos.flat.copy()
        return np.concatenate(
            (
                position,
            )
        )

    def step(self, action):
        # print(action)
        action += np.array([0.99616, 0, 0.486265, 0.4166, 0.99616, 0, 0.486265, 0.4166, 0.00, -0.1047, 0.462355, 0.50088, 1.61351, -0.00, -0.2855, 0.7852])
        self.do_simulation(action, self.frame_skip)

        survival_reward = self._survival_reward if self.is_healthy else 0

        distance_reward = (-np.linalg.norm(self.data.body("red_ball").xpos - self.data.body("target_obj").xpos)) * 10
        # 获取手指末端和圆柱体的位置
        f  = self.data.body("finger_end").xpos  # 大拇指  
        f1 = self.data.body("finger_end_1").xpos
        f2 = self.data.body("finger_end_2").xpos
        cylinder_pos = self.data.body("cylinder").xpos + np.array([0.03, 0, -0.03])

        # 计算手指末端到圆柱体表面的最短距离
        def distance_to_cylinder_surface(point, cylinder_center, cylinder_radius=0.03):
            px, py, pz = point
            cx, cy, cz = cylinder_center
            # 水平距离只考虑XZ平面
            print(point, cylinder_center)
            horizontal_distance = np.linalg.norm([px - cx, pz - cz])
            surface_distance = abs(horizontal_distance - cylinder_radius)
            return surface_distance

        f_d  = distance_to_cylinder_surface(f, cylinder_pos)
        f1_d = distance_to_cylinder_surface(f1, cylinder_pos)
        f2_d = distance_to_cylinder_surface(f2, cylinder_pos)

        approach_reward = (-f_d - f1_d - f2_d) * 10
        
        # print("reward: ", approach_reward, distance_reward, time_step_reward)
        
        reward =  approach_reward + distance_reward + survival_reward
        observation = self._get_obs()
        terminated = self.terminated
        info = {
            "approach_reward": approach_reward,
            "distance_reward": distance_reward,
            "survival_reward": survival_reward,
            "total_reward": reward,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        """
        <key qpos='0.99616 0 0.486265 0.4166 0.8944 0.0 0.54604 0.56108 0.00 -0.1047 0.462355 0.50088 1.61351 -0.00 -0.2855 0.7852 0 0 0 0 0 0'/>
        """
        qpos = np.array([0.99616, 0, 0.486265, 0.4166, 0.99616, 0, 0.486265, 0.4166, 0.00, -0.1047, 0.462355, 0.50088, 1.61351, -0.00, -0.2855, 0.7852, 0, 0, 0, 0, 0, 0])
        qvel = self.init_qvel 
        self.data.ctrl[:] = np.array([0.99616, 0, 0.486265, 0.4166, 0.99616, 0, 0.486265, 0.4166, 0.00, -0.1047, 0.462355, 0.50088, 1.61351, -0.00, -0.2855, 0.7852])
        
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation
