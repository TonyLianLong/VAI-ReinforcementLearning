import numpy as np
from gym.spaces import Box

from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerHammerEnvV2(SawyerXYZEnv):
    def __init__(self):

        liftThresh = 0.09
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.4, 0.0)
        obj_high = (0.1, 0.5, 0.0)
        goal_low = (0.2399, .7399, 0.109)
        goal_high = (0.2401, .7401, 0.111)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'hammer_init_pos': np.array([0, 0.5, 0.0]),
            'hand_init_pos': np.array([0, 0.4, 0.2]),
        }
        self.goal = self.init_config['hammer_init_pos']
        self.hammer_init_pos = self.init_config['hammer_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self.liftThresh = liftThresh
        self.max_path_length = 200

        self._random_reset_space = Box(np.array(obj_low), np.array(obj_high))
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.max_nail_dist = None
        self.max_hammer_dist = None

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_hammer.xml')

    @_assert_task_is_set
    def step(self, action):
        ob = super().step(action)
        reward, _, reachDist, pickRew, _, _, screwDist = self.compute_reward(action, ob)
        self.curr_path_length += 1

        info = {
            'reachDist': reachDist,
            'pickRew': pickRew,
            'epRew': reward,
            'goalDist': screwDist,
            'success': float(screwDist <= 0.05)
        }

        return ob, reward, False, info

    def _get_pos_objects(self):
        return np.hstack((
            self.get_body_com('hammer').copy(),
            self.get_body_com('nail_link').copy()
        ))

    def _set_hammer_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()

        # Set position of box & nail (these are not randomized)
        self.sim.model.body_pos[self.model.body_name2id(
            'box'
        )] = np.array([0.24, 0.85, 0.0])
        # Update _target_pos
        self._target_pos = self._get_site_pos('goal')

        # Randomize hammer position
        self.hammer_init_pos = self._get_state_rand_vec() if self.random_init \
            else self.init_config['hammer_init_pos']
        self._set_hammer_xyz(self.hammer_init_pos)

        # Update heights (for use in reward function)
        self.hammerHeight = self.get_body_com('hammer').copy()[2]
        self.heightTarget = self.hammerHeight + self.liftThresh

        # Update distances (for use in reward function)
        nail_init_pos = self._get_site_pos('nailHead')
        self.max_nail_dist = (self._target_pos - nail_init_pos)[1]
        self.max_hammer_dist = np.linalg.norm(
            np.array([
                self.hammer_init_pos[0],
                self.hammer_init_pos[1],
                self.heightTarget
            ]) - nail_init_pos + self.heightTarget + np.abs(self.max_nail_dist)
        )

        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand()
        self.pickCompleted = False

    def compute_reward(self, actions, obs):

        hammer_pos = obs[3:6]
        nail_pos = obs[6:9]

        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        fingerCOM = (rightFinger + leftFinger) / 2.0

        hammerDist = np.linalg.norm(nail_pos - hammer_pos)
        screwDist = np.abs(nail_pos[1] - self._target_pos[1])
        reachDist = np.linalg.norm(hammer_pos - fingerCOM)

        def reachReward():
            reachRew = -reachDist
            # incentive to close fingers when reachDist is small
            if reachDist < 0.05:
                reachRew = -reachDist + max(actions[-1], 0)/50
            return reachRew, reachDist

        def pickCompletionCriteria():
            tolerance = 0.01
            if hammer_pos[2] >= (self.heightTarget - tolerance):
                return True
            else:
                return False

        if pickCompletionCriteria():
            self.pickCompleted = True

        def objDropped():
            return (hammer_pos[2] < (self.hammerHeight + 0.005)) and (hammerDist >0.02) and (reachDist > 0.02)
            # Object on the ground, far away from the goal, and from the gripper
            # Can tweak the margin limits

        def orig_pickReward():
            hScale = 100

            if self.pickCompleted and not(objDropped()):
                return hScale*self.heightTarget
            elif (reachDist < 0.1) and (hammer_pos[2] > (self.hammerHeight + 0.005)):
                return hScale * min(self.heightTarget, hammer_pos[2])
            else:
                return 0

        def hammerReward():
            c1 = 1000
            c2 = 0.01
            c3 = 0.001

            cond = self.pickCompleted and (reachDist < 0.1) and not(objDropped())
            if cond:
                hammerRew = 1000*(self.max_hammer_dist - hammerDist - screwDist) + c1*(np.exp(-((hammerDist+screwDist)**2)/c2) + np.exp(-((hammerDist+screwDist)**2)/c3))
                hammerRew = max(hammerRew, 0)
                return [hammerRew, hammerDist, screwDist]
            else:
                return [0, hammerDist, screwDist]

        reachRew, reachDist = reachReward()
        pickRew = orig_pickReward()
        hammerRew , hammerDist, screwDist = hammerReward()
        assert ((hammerRew >=0) and (pickRew>=0))
        reward = reachRew + pickRew + hammerRew

        return [reward, reachRew, reachDist, pickRew, hammerRew, hammerDist, screwDist]
