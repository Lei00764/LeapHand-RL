import numpy as np
import os
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import mujoco

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


class LeaphandEnvV1(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }
    mj_xml_path = os.path.join(os.path.dirname(__file__), "..", "..", "mj_model", "mjmodel.xml")

    def __init__(
        self,
        approach_reward=1,
        healthy_reward=1,
        terminate_when_unhealthy=True,
        healthy_z_range=(-100, 100),
        reset_noise_scale=1e-2,
        seed=42,
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
        self._survival_reward = 80 # 存活奖励

        self._is_healthy = True

        self.time_step = 0
        self.seed(seed)
        # self.prev_action = np.array([0.99616, 0, 0.486265, 0.4166, 0.99616, 0, 0.486265, 0.4166, 0.00, -0.1047, 0.462355, 0.50088, 1.61351, -0.00, -0.2855, 0.7852])

        # Observation Dim is in function _get_obs()
        observation_space = Box(
            low=-5, high=5, shape=(78,), dtype=np.float64
        )
        MujocoEnv.__init__(
            self,
            LeaphandEnvV1.mj_xml_path,
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )
        print("observation space shape: ", self.observation_space.shape)
        print("action space shape: ", self.action_space.shape)
    
    def seed(self, seed=None):
        np.random.seed(seed)
        
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

    def check_healthy(self):
        # 死亡判断条件：
        # 手指偏离圆柱体表面
        f  = self.data.body("finger_end").xpos.copy()  # 大拇指  
        f1 = self.data.body("finger_end_1").xpos.copy()
        f2 = self.data.body("finger_end_2").xpos.copy()

        # print("xxx3", self.data.body("cylinder").xpos)
        # print(f, cylinder_pos)
        # print(f1, cylinder_pos)
        # print(f2, cylinder_pos)
        # print(target_pos, cylinder_pos)
        # print("xxx2", self.data.body("cylinder").xpos)

        cylinder_bottom_pos = self.data.body("cylinder_bottom_ball").xpos.copy()
        cylinder_upper_pos = self.data.body("red_ball").xpos.copy()
        # print("xxx1", self.data.body("cylinder").xpos)
        cylinder_axis = cylinder_upper_pos - cylinder_bottom_pos
        f_d  = np.linalg.norm(np.cross(f-cylinder_bottom_pos, cylinder_axis))/np.linalg.norm(cylinder_axis) - 0.028
        f1_d = np.linalg.norm(np.cross(f1-cylinder_bottom_pos, cylinder_axis))/np.linalg.norm(cylinder_axis) - 0.028
        f2_d = np.linalg.norm(np.cross(f2-cylinder_bottom_pos, cylinder_axis))/np.linalg.norm(cylinder_axis) - 0.028
        
        threshold = 0.04

        self._is_healthy = (f_d < threshold + 0.01) and (f1_d < threshold) and (f2_d < threshold)
        # 限制三根手指的z坐标范围，跟圆柱体圆心的z坐标差值不能超过0.005
        # delta_z = 0.02
        # if abs(f[2] - cylinder_pos[2]) > delta_z or abs(f1[2] - cylinder_pos[2]) > delta_z or abs(f2[2] - cylinder_pos[2]) > delta_z:
        #     self._is_healthy = False

        return self._is_healthy

    @property
    def terminated(self):
        terminated = (not self.check_healthy()) if self._terminate_when_unhealthy else False
        return terminated

    def _get_obs(self):
        # dimesion=47
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()
        delta_pos = self.data.qpos.flat.copy() - np.array([0.71632,0, 0.964465, 0, 0.74176, 0, 1.01229, 0, 0, 0, 0, -0.1252, 1.64794, 0, -0.27, 0.8657, 0, 0, 0, 0, 0, 0, 0 ,0 ,0])
        # position = position[:16]
        src = self.data.body("red_ball").xpos.copy()
        dst = self.data.body("target_obj").xpos.copy()
        # # 相对偏差
        delta = dst - src
        observation = np.concatenate((position, delta_pos, delta,velocity))

        # TODO：可能需要更改 target
        # print(position)
        return observation
        

    def step(self, action):
        # self.prev_action = np.array([0.99616, 0, 0.486265, 0.4166, 0.99616, 0, 0.486265, 0.4166, 0.00, -0.1047, 0.462355, 0.50088, 1.61351, -0.00, -0.2855, 0.7852])
        # max_delta = 0.01  # 每次 action 最大变化量
        # action = np.clip(action, self.prev_action - max_delta, self.prev_action + max_delta)
        # self.prev_action = action
        # print(action)
        action += np.array([0.71632,0, 0.964465, 0, 0.74176, 0, 1.01229, 0, 0, 0, 0, -0.1252, 1.64794, 0, -0.27, 0.8657])
        self.do_simulation(action, self.frame_skip)

        self.time_step += 1

        survival_reward = self._survival_reward if self._is_healthy else 0

        distance = np.linalg.norm(self.data.body("red_ball").xpos.copy() - self.data.body("target_obj").xpos.copy())
        distance_reward = - distance * 1000
        velocity_penalty = - 0.1 * np.square(self.data.qvel.copy()).sum()
        # 获取手指末端和圆柱体的位置
        f  = self.data.body("finger_end").xpos.copy()  # 大拇指  
        f1 = self.data.body("finger_end_1").xpos.copy()
        f2 = self.data.body("finger_end_2").xpos.copy()

        cylinder_bottom_pos = self.data.body("cylinder_bottom_ball").xpos.copy()
        cylinder_upper_pos = self.data.body("red_ball").xpos.copy()
        # print("xxx1", self.data.body("cylinder").xpos)
        cylinder_axis = cylinder_upper_pos - cylinder_bottom_pos
        f_d  = np.linalg.norm(np.cross(f-cylinder_bottom_pos, cylinder_axis))/np.linalg.norm(cylinder_axis) - 0.028
        f1_d = np.linalg.norm(np.cross(f1-cylinder_bottom_pos, cylinder_axis))/np.linalg.norm(cylinder_axis) - 0.028
        f2_d = np.linalg.norm(np.cross(f2-cylinder_bottom_pos, cylinder_axis))/np.linalg.norm(cylinder_axis) - 0.028
        approach_reward = (- f_d - f1_d - f2_d) * 1000
        
        reward = distance_reward + approach_reward + survival_reward + velocity_penalty
        observation = self._get_obs()
        terminated = self.terminated
        info = {
            "approach_reward": approach_reward,
            "distance_reward": distance_reward,
            "survival_reward": survival_reward,
            "velocity_penalty": velocity_penalty,
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
        qpos = np.array([0.71632,0, 0.964465, 0, 0.74176, 0, 1.01229, 0, 0, 0, 0, -0.1252, 1.64794, 0, -0.27, 0.8657, 0, 0, 0, 0, 0, 0, 0 ,0 ,0])
        qvel = self.init_qvel 
        self.data.ctrl[:] = np.array([0.71632,0, 0.964465, 0, 0.74176, 0, 1.01229, 0, 0, 0, 0, -0.1252, 1.64794, 0, -0.27, 0.8657])
        self.set_state(qpos, qvel)
        self.data.qpos[-3:] = np.random.uniform(-0.03, 0.03, size=3)
        # if self.time_step % 100000 == 0:
        # target_x = np.random.uniform(-0.01, 0.02)
        # target_y = np.random.uniform(-0.01, 0.02)
        # target_z = np.random.uniform(-0.02, 0)
        # self.data.qpos[19:] = np.array([target_x, target_y, target_z])
        #     target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_obj")
        #     self.model.body_pos[target_body_id] = np.array([target_x, target_y, target_z])
        
        observation = self._get_obs()
        return observation
