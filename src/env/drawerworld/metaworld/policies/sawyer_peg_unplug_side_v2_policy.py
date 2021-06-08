import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerPegUnplugSideV2Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'peg_pos': obs[3:6],
            'unused_info': obs[6:],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=25.)
        action['grab_effort'] = self._grab_effort(o_d)

        return action.array

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_peg = o_d['peg_pos'] + np.array([-.02, .0, .035])

        if np.linalg.norm(pos_curr[:2] - pos_peg[:2]) > 0.04:
            return pos_peg + np.array([0., 0., 0.2])
        elif abs(pos_curr[2] - .15) > 0.02:
            return np.array([*pos_peg[:2], .15])
        else:
            return pos_curr + np.array([.01, .0, .0])

    @staticmethod
    def _grab_effort(o_d):
        pos_curr = o_d['hand_pos']
        pos_peg = o_d['peg_pos'] + np.array([-.02, .0, .035])

        if np.linalg.norm(pos_curr[:2] - pos_peg[:2]) > 0.04 \
            or abs(pos_curr[2] - pos_peg[2]) > 0.15:
            return -1.
        else:
            return .1
