# Copyright 2019 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""A CMU humanoid walker."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import os

from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_control.locomotion.walkers import base
from dm_control.locomotion.walkers import legacy_base
from dm_control.locomotion.walkers import scaled_actuators
from dm_control.mujoco import wrapper as mj_wrapper
import numpy as np
import six
from six.moves import zip

_XML_PATH = os.path.join(os.path.dirname(__file__), 'assets/humanoid_CMU.xml')

_WALKER_GEOM_GROUP = 2

_CMU_MOCAP_JOINTS = (
    'lfemurrz', 'lfemurry', 'lfemurrx', 'ltibiarx', 'lfootrz', 'lfootrx',
    'ltoesrx', 'rfemurrz', 'rfemurry', 'rfemurrx', 'rtibiarx', 'rfootrz',
    'rfootrx', 'rtoesrx', 'lowerbackrz', 'lowerbackry', 'lowerbackrx',
    'upperbackrz', 'upperbackry', 'upperbackrx', 'thoraxrz', 'thoraxry',
    'thoraxrx', 'lowerneckrz', 'lowerneckry', 'lowerneckrx', 'upperneckrz',
    'upperneckry', 'upperneckrx', 'headrz', 'headry', 'headrx', 'lclaviclerz',
    'lclaviclery', 'lhumerusrz', 'lhumerusry', 'lhumerusrx', 'lradiusrx',
    'lwristry', 'lhandrz', 'lhandrx', 'lfingersrx', 'lthumbrz', 'lthumbrx',
    'rclaviclerz', 'rclaviclery', 'rhumerusrz', 'rhumerusry', 'rhumerusrx',
    'rradiusrx', 'rwristry', 'rhandrz', 'rhandrx', 'rfingersrx', 'rthumbrz',
    'rthumbrx')


# pylint: disable=bad-whitespace
PositionActuatorParams = collections.namedtuple(
    'PositionActuatorParams', ['name', 'forcerange', 'kp'])
_POSITION_ACTUATORS = [
    PositionActuatorParams('headrx',      [-20,   20 ], 20 ),
    PositionActuatorParams('headry',      [-20,   20 ], 20 ),
    PositionActuatorParams('headrz',      [-20,   20 ], 20 ),
    PositionActuatorParams('lclaviclery', [-20,   20 ], 20 ),
    PositionActuatorParams('lclaviclerz', [-20,   20 ], 20 ),
    PositionActuatorParams('lfemurrx',    [-120,  120], 120),
    PositionActuatorParams('lfemurry',    [-80,   80 ], 80 ),
    PositionActuatorParams('lfemurrz',    [-80,   80 ], 80 ),
    PositionActuatorParams('lfingersrx',  [-20,   20 ], 20 ),
    PositionActuatorParams('lfootrx',     [-50,   50 ], 50 ),
    PositionActuatorParams('lfootrz',     [-50,   50 ], 50 ),
    PositionActuatorParams('lhandrx',     [-20,   20 ], 20 ),
    PositionActuatorParams('lhandrz',     [-20,   20 ], 20 ),
    PositionActuatorParams('lhumerusrx',  [-60,   60 ], 60 ),
    PositionActuatorParams('lhumerusry',  [-60,   60 ], 60 ),
    PositionActuatorParams('lhumerusrz',  [-60,   60 ], 60 ),
    PositionActuatorParams('lowerbackrx', [-120,  120], 150),
    PositionActuatorParams('lowerbackry', [-120,  120], 150),
    PositionActuatorParams('lowerbackrz', [-120,  120], 150),
    PositionActuatorParams('lowerneckrx', [-20,   20 ], 20 ),
    PositionActuatorParams('lowerneckry', [-20,   20 ], 20 ),
    PositionActuatorParams('lowerneckrz', [-20,   20 ], 20 ),
    PositionActuatorParams('lradiusrx',   [-60,   60 ], 60 ),
    PositionActuatorParams('lthumbrx',    [-20,   20 ], 20) ,
    PositionActuatorParams('lthumbrz',    [-20,   20 ], 20 ),
    PositionActuatorParams('ltibiarx',    [-80,   80 ], 80 ),
    PositionActuatorParams('ltoesrx',     [-20,   20 ], 20 ),
    PositionActuatorParams('lwristry',    [-20,   20 ], 20 ),
    PositionActuatorParams('rclaviclery', [-20,   20 ], 20 ),
    PositionActuatorParams('rclaviclerz', [-20,   20 ], 20 ),
    PositionActuatorParams('rfemurrx',    [-120,  120], 120),
    PositionActuatorParams('rfemurry',    [-80,   80 ], 80 ),
    PositionActuatorParams('rfemurrz',    [-80,   80 ], 80 ),
    PositionActuatorParams('rfingersrx',  [-20,   20 ], 20 ),
    PositionActuatorParams('rfootrx',     [-50,   50 ], 50 ),
    PositionActuatorParams('rfootrz',     [-50,   50 ], 50 ),
    PositionActuatorParams('rhandrx',     [-20,   20 ], 20 ),
    PositionActuatorParams('rhandrz',     [-20,   20 ], 20 ),
    PositionActuatorParams('rhumerusrx',  [-60,   60 ], 60 ),
    PositionActuatorParams('rhumerusry',  [-60,   60 ], 60 ),
    PositionActuatorParams('rhumerusrz',  [-60,   60 ], 60 ),
    PositionActuatorParams('rradiusrx',   [-60,   60 ], 60 ),
    PositionActuatorParams('rthumbrx',    [-20,   20 ], 20 ),
    PositionActuatorParams('rthumbrz',    [-20,   20 ], 20 ),
    PositionActuatorParams('rtibiarx',    [-80,   80 ], 80 ),
    PositionActuatorParams('rtoesrx',     [-20,   20 ], 20 ),
    PositionActuatorParams('rwristry',    [-20,   20 ], 20 ),
    PositionActuatorParams('thoraxrx',    [-80,   80 ], 100),
    PositionActuatorParams('thoraxry',    [-80,   80 ], 100),
    PositionActuatorParams('thoraxrz',    [-80,   80 ], 100),
    PositionActuatorParams('upperbackrx', [-80,   80 ], 80 ),
    PositionActuatorParams('upperbackry', [-80,   80 ], 80 ),
    PositionActuatorParams('upperbackrz', [-80,   80 ], 80 ),
    PositionActuatorParams('upperneckrx', [-20,   20 ], 20 ),
    PositionActuatorParams('upperneckry', [-20,   20 ], 20 ),
    PositionActuatorParams('upperneckrz', [-20,   20 ], 20 ),
]
# pylint: enable=bad-whitespace

_UPRIGHT_POS = (0.0, 0.0, 0.94)
_UPRIGHT_QUAT = (0.859, 1.0, 1.0, 0.859)

# Height of head above which the humanoid is considered standing.
_STAND_HEIGHT = 1.5

_TORQUE_THRESHOLD = 60


@six.add_metaclass(abc.ABCMeta)
class _CMUHumanoidBase(legacy_base.Walker):
  """The abstract base class for walkers compatible with the CMU humanoid."""

  def _build(self,
             name='walker',
             marker_rgba=None,
             initializer=None):
    self._mjcf_root = mjcf.from_path(self._xml_path)
    if name:
      self._mjcf_root.model = name

    # Set corresponding marker color if specified.
    if marker_rgba is not None:
      for geom in self.marker_geoms:
        geom.set_attributes(rgba=marker_rgba)

    self._actuator_order = np.argsort(_CMU_MOCAP_JOINTS)
    self._inverse_order = np.argsort(self._actuator_order)

    super(_CMUHumanoidBase, self)._build(initializer=initializer)

  def _build_observables(self):
    return CMUHumanoidObservables(self)

  @abc.abstractproperty
  def _xml_path(self):
    raise NotImplementedError

  @composer.cached_property
  def mocap_joints(self):
    return tuple(
        self._mjcf_root.find('joint', name) for name in _CMU_MOCAP_JOINTS)

  @property
  def actuator_order(self):
    """Index of joints from the CMU mocap dataset sorted alphabetically by name.

    Actuators in this walkers are ordered alphabetically by name. This property
    provides a mapping between from actuator ordering to canonical CMU ordering.

    Returns:
      A list of integers corresponding to joint indices from the CMU dataset.
      Specifically, the n-th element in the list is the index of the CMU joint
      index that corresponds to the n-th actuator in this walker.
    """
    return self._actuator_order

  @property
  def actuator_to_joint_order(self):
    """Index of actuators corresponding to each CMU mocap joint.

    Actuators in this walkers are ordered alphabetically by name. This property
    provides a mapping between from canonical CMU ordering to actuator ordering.

    Returns:
      A list of integers corresponding to actuator indices within this walker.
      Specifically, the n-th element in the list is the index of the actuator
      in this walker that corresponds to the n-th joint from the CMU mocap
      dataset.
    """
    return self._inverse_order

  @property
  def upright_pose(self):
    return base.WalkerPose(xpos=_UPRIGHT_POS, xquat=_UPRIGHT_QUAT)

  @property
  def mjcf_model(self):
    return self._mjcf_root

  @composer.cached_property
  def actuators(self):
    return tuple(self._mjcf_root.find_all('actuator'))

  @composer.cached_property
  def root_body(self):
    return self._mjcf_root.find('body', 'root')

  @composer.cached_property
  def head(self):
    return self._mjcf_root.find('body', 'head')

  @composer.cached_property
  def left_arm_root(self):
    return self._mjcf_root.find('body', 'lclavicle')

  @composer.cached_property
  def right_arm_root(self):
    return self._mjcf_root.find('body', 'rclavicle')

  @composer.cached_property
  def ground_contact_geoms(self):
    return tuple(self._mjcf_root.find('body', 'lfoot').find_all('geom') +
                 self._mjcf_root.find('body', 'rfoot').find_all('geom'))

  @composer.cached_property
  def standing_height(self):
    return _STAND_HEIGHT

  @composer.cached_property
  def end_effectors(self):
    return (self._mjcf_root.find('body', 'rradius'),
            self._mjcf_root.find('body', 'lradius'),
            self._mjcf_root.find('body', 'rfoot'),
            self._mjcf_root.find('body', 'lfoot'))

  @composer.cached_property
  def observable_joints(self):
    return tuple(actuator.joint for actuator in self.actuators
                 if actuator.joint is not None)

  @composer.cached_property
  def bodies(self):
    return tuple(self._mjcf_root.find_all('body'))

  @composer.cached_property
  def egocentric_camera(self):
    return self._mjcf_root.find('camera', 'egocentric')

  @composer.cached_property
  def body_camera(self):
    return self._mjcf_root.find('camera', 'bodycam')

  @property
  def marker_geoms(self):
    return (self._mjcf_root.find('geom', 'rradius'),
            self._mjcf_root.find('geom', 'lradius'))


class CMUHumanoid(_CMUHumanoidBase):
  """A CMU humanoid walker."""

  @property
  def _xml_path(self):
    return _XML_PATH


class CMUHumanoidPositionControlled(CMUHumanoid):
  """A position-controlled CMU humanoid with control range scaled to [-1, 1]."""

  def _build(self, *args, **kwargs):
    super(CMUHumanoidPositionControlled, self)._build(*args, **kwargs)
    self._mjcf_root.default.general.forcelimited = 'true'
    self._mjcf_root.actuator.motor.clear()
    for actuator_params in _POSITION_ACTUATORS:
      associated_joint = self._mjcf_root.find('joint', actuator_params.name)
      scaled_actuators.add_position_actuator(
          name=actuator_params.name,
          target=associated_joint,
          kp=actuator_params.kp,
          qposrange=associated_joint.range,
          ctrlrange=(-1, 1),
          forcerange=actuator_params.forcerange)
    limits = zip(*(actuator.joint.range for actuator in self.actuators))  # pylint: disable=not-an-iterable
    lower, upper = (np.array(limit) for limit in limits)
    self._scale = upper - lower
    self._offset = upper + lower

  def cmu_pose_to_actuation(self, target_pose):
    """Creates the control signal corresponding a CMU mocap joints pose.

    Args:
      target_pose: An array containing the target position for each joint.
        These must be given in "canonical CMU order" rather than "qpos order",
        i.e. the order of `target_pose[self.actuator_order]` should correspond
        to the order of `physics.bind(self.actuators).ctrl`.

    Returns:
      An array of the same shape as `target_pose` containing inputs for position
      controllers. Writing these values into `physics.bind(self.actuators).ctrl`
      will cause the actuators to drive joints towards `target_pose`.
    """
    return (2 * target_pose[self.actuator_order] - self._offset) / self._scale


class CMUHumanoidObservables(legacy_base.WalkerObservables):
  """Observables for the Humanoid."""

  @composer.observable
  def body_camera(self):
    options = mj_wrapper.MjvOption()

    # Don't render this walker's geoms.
    options.geomgroup[_WALKER_GEOM_GROUP] = 0
    return observable.MJCFCamera(
        self._entity.body_camera, width=64, height=64, scene_option=options)

  @composer.observable
  def head_height(self):
    return observable.MJCFFeature('xpos', self._entity.head)[2]

  @composer.observable
  def sensors_torque(self):
    return observable.MJCFFeature(
        'sensordata', self._entity.mjcf_model.sensor.torque,
        corruptor=lambda v, random_state: np.tanh(2 * v / _TORQUE_THRESHOLD))

  @composer.observable
  def actuator_activation(self):
    return observable.MJCFFeature('act',
                                  self._entity.mjcf_model.find_all('actuator'))

  @composer.observable
  def appendages_pos(self):
    """Equivalent to `end_effectors_pos` with the head's position appended."""
    def relative_pos_in_egocentric_frame(physics):
      end_effectors_with_head = (
          self._entity.end_effectors + (self._entity.head,))
      end_effector = physics.bind(end_effectors_with_head).xpos
      torso = physics.bind(self._entity.root_body).xpos
      xmat = np.reshape(physics.bind(self._entity.root_body).xmat, (3, 3))
      return np.reshape(np.dot(end_effector - torso, xmat), -1)
    return observable.Generic(relative_pos_in_egocentric_frame)

  @property
  def proprioception(self):
    return [
        self.joints_pos,
        self.joints_vel,
        self.actuator_activation,
        self.body_height,
        self.end_effectors_pos,
        self.appendages_pos,
        self.world_zaxis
    ] + self._collect_from_attachments('proprioception')
