# Copyright 2018 The dm_control Authors.
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

"""Tests for composer.Entity."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
# Internal dependencies.

from absl.testing import absltest
from absl.testing import parameterized
from dm_control import mjcf
from dm_control.composer import arena
from dm_control.composer import define
from dm_control.composer import entity
from dm_control.composer.observation.observable import base as observable
import numpy as np
import six
from six.moves import range

_NO_ROTATION = (1, 0, 0, 0)  # Tests support for non-arrays and non-floats.
_NINETY_DEGREES_ABOUT_X = np.array(
    [np.cos(np.pi / 4), np.sin(np.pi / 4), 0., 0.])
_NINETY_DEGREES_ABOUT_Y = np.array(
    [np.cos(np.pi / 4), 0., np.sin(np.pi / 4), 0.])
_NINETY_DEGREES_ABOUT_Z = np.array(
    [np.cos(np.pi / 4), 0., 0., np.sin(np.pi / 4)])
_FORTYFIVE_DEGREES_ABOUT_X = np.array(
    [np.cos(np.pi / 8), np.sin(np.pi / 8), 0., 0.])

_TEST_ROTATIONS = [
    # Triplets of original rotation, new rotation and final rotation.
    (None, _NO_ROTATION, _NO_ROTATION),
    (_NO_ROTATION, _NINETY_DEGREES_ABOUT_Z, _NINETY_DEGREES_ABOUT_Z),
    (_FORTYFIVE_DEGREES_ABOUT_X, _NINETY_DEGREES_ABOUT_Y,
     np.array([0.65328, 0.2706, 0.65328, -0.2706])),
]


def _param_product(**param_lists):
  keys, values = zip(*param_lists.items())
  for combination in itertools.product(*values):
    yield dict(zip(keys, combination))


class TestEntity(entity.Entity):
  """Simple test entity that does nothing but declare some observables."""

  def _build(self, name='test_entity'):
    self._mjcf_root = mjcf.element.RootElement(model=name)
    self._mjcf_root.worldbody.add('geom', type='sphere', size=(0.1,))

  def _build_observables(self):
    return TestEntityObservables(self)

  @property
  def mjcf_model(self):
    return self._mjcf_root


class TestEntityObservables(entity.Observables):
  """Trivial observables for the test entity."""

  @define.observable
  def observable0(self):
    return observable.Generic(lambda phys: 0.0)

  @define.observable
  def observable1(self):
    return observable.Generic(lambda phys: 1.0)


class EntityTest(parameterized.TestCase):

  def setUp(self):
    super(EntityTest, self).setUp()
    self.entity = TestEntity()

  def testNumObservables(self):
    """Tests that the observables dict has the right number of entries."""
    self.assertLen(self.entity.observables.as_dict(), 2)

  def testObservableNames(self):
    """Tests that the observables dict keys correspond to the observable names.
    """
    obs = self.entity.observables.as_dict()
    self.assertIn('observable0', obs)
    self.assertIn('observable1', obs)

    subentity = TestEntity(name='subentity')
    self.entity.attach(subentity)
    self.assertIn('subentity/observable0', subentity.observables.as_dict())
    self.assertEqual(subentity.observables.dict_keys.observable0,
                     'subentity/observable0')
    self.assertIn('observable0', dir(subentity.observables.dict_keys))
    self.assertIn('subentity/observable1', subentity.observables.as_dict())
    self.assertEqual(subentity.observables.dict_keys.observable1,
                     'subentity/observable1')
    self.assertIn('observable1', dir(subentity.observables.dict_keys))

  def testEnableDisableObservables(self):
    """Test the enabling and disable functionality for observables."""
    all_obs = self.entity.observables.as_dict()

    self.entity.observables.enable_all()
    for obs in all_obs.values():
      self.assertTrue(obs.enabled)

    self.entity.observables.disable_all()
    for obs in all_obs.values():
      self.assertFalse(obs.enabled)

    self.entity.observables.observable0.enabled = True
    self.assertTrue(all_obs['observable0'].enabled)

  def testObservableDefaultOptions(self):
    corruptor = lambda x: x
    options = {
        'update_interval': 2,
        'buffer_size': 10,
        'delay': 1,
        'aggregator': 'max',
        'corruptor': corruptor,
        'enabled': True
    }
    self.entity.observables.set_options(options)

    for obs in self.entity.observables.as_dict().values():
      self.assertEqual(obs.update_interval, 2)
      self.assertEqual(obs.delay, 1)
      self.assertEqual(obs.buffer_size, 10)
      self.assertEqual(obs.aggregator, observable.AGGREGATORS['max'])
      self.assertEqual(obs.corruptor, corruptor)
      self.assertTrue(obs.enabled)

  def testObservablePartialDefaultOptions(self):
    options = {'update_interval': 2, 'delay': 1}
    self.entity.observables.set_options(options)

    for obs in self.entity.observables.as_dict().values():
      self.assertEqual(obs.update_interval, 2)
      self.assertEqual(obs.delay, 1)
      self.assertEqual(obs.buffer_size, None)
      self.assertEqual(obs.aggregator, None)
      self.assertEqual(obs.corruptor, None)

  def testObservableOptionsInvalidName(self):
    options = {'asdf': None}
    with six.assertRaisesRegex(
        self, KeyError, 'No observable with name \'asdf\''):
      self.entity.observables.set_options(options)

  def testObservableInvalidOptions(self):
    options = {'observable0': {'asdf': 2}}
    with six.assertRaisesRegex(self, AttributeError,
                               'Cannot add attribute asdf in configure.'):
      self.entity.observables.set_options(options)

  def testObservableOptions(self):
    options = {
        'observable0': {
            'update_interval': 2,
            'delay': 3
        },
        'observable1': {
            'update_interval': 4,
            'delay': 5
        }
    }
    self.entity.observables.set_options(options)
    observables = self.entity.observables.as_dict()
    self.assertEqual(observables['observable0'].update_interval, 2)
    self.assertEqual(observables['observable0'].delay, 3)
    self.assertEqual(observables['observable0'].buffer_size, None)
    self.assertEqual(observables['observable0'].aggregator, None)
    self.assertEqual(observables['observable0'].corruptor, None)
    self.assertFalse(observables['observable0'].enabled)

    self.assertEqual(observables['observable1'].update_interval, 4)
    self.assertEqual(observables['observable1'].delay, 5)
    self.assertEqual(observables['observable1'].buffer_size, None)
    self.assertEqual(observables['observable1'].aggregator, None)
    self.assertEqual(observables['observable1'].corruptor, None)
    self.assertFalse(observables['observable1'].enabled)

  def testObservableOptionsEntityConstructor(self):
    options = {
        'observable0': {
            'update_interval': 2,
            'delay': 3
        },
        'observable1': {
            'update_interval': 4,
            'delay': 5
        }
    }
    ent = TestEntity(observable_options=options)
    observables = ent.observables.as_dict()
    self.assertEqual(observables['observable0'].update_interval, 2)
    self.assertEqual(observables['observable0'].delay, 3)
    self.assertEqual(observables['observable0'].buffer_size, None)
    self.assertEqual(observables['observable0'].aggregator, None)
    self.assertEqual(observables['observable0'].corruptor, None)
    self.assertFalse(observables['observable0'].enabled)

    self.assertEqual(observables['observable1'].update_interval, 4)
    self.assertEqual(observables['observable1'].delay, 5)
    self.assertEqual(observables['observable1'].buffer_size, None)
    self.assertEqual(observables['observable1'].aggregator, None)
    self.assertEqual(observables['observable1'].corruptor, None)
    self.assertFalse(observables['observable1'].enabled)

  def testObservablePartialOptions(self):
    options = {'observable0': {'update_interval': 2, 'delay': 3}}
    self.entity.observables.set_options(options)
    observables = self.entity.observables.as_dict()
    self.assertEqual(observables['observable0'].update_interval, 2)
    self.assertEqual(observables['observable0'].delay, 3)
    self.assertEqual(observables['observable0'].buffer_size, None)
    self.assertEqual(observables['observable0'].aggregator, None)
    self.assertEqual(observables['observable0'].corruptor, None)
    self.assertFalse(observables['observable0'].enabled)

    self.assertEqual(observables['observable1'].update_interval, 1)
    self.assertEqual(observables['observable1'].delay, None)
    self.assertEqual(observables['observable1'].buffer_size, None)
    self.assertEqual(observables['observable1'].aggregator, None)
    self.assertEqual(observables['observable1'].corruptor, None)
    self.assertFalse(observables['observable1'].enabled)

  def testAttach(self):
    entities = [TestEntity() for _ in range(4)]
    entities[0].attach(entities[1])
    entities[1].attach(entities[2])
    entities[0].attach(entities[3])

    self.assertIsNone(entities[0].parent)
    self.assertIs(entities[1].parent, entities[0])
    self.assertIs(entities[2].parent, entities[1])
    self.assertIs(entities[3].parent, entities[0])

    self.assertIsNone(entities[0].mjcf_model.parent_model)
    self.assertIs(entities[1].mjcf_model.parent_model, entities[0].mjcf_model)
    self.assertIs(entities[2].mjcf_model.parent_model, entities[1].mjcf_model)
    self.assertIs(entities[3].mjcf_model.parent_model, entities[0].mjcf_model)

    self.assertEqual(list(entities[0].iter_entities()), entities)

  def testDetach(self):
    entities = [TestEntity() for _ in range(4)]
    entities[0].attach(entities[1])
    entities[1].attach(entities[2])
    entities[0].attach(entities[3])

    entities[1].detach()
    with six.assertRaisesRegex(self, RuntimeError, 'not attached'):
      entities[1].detach()

    self.assertIsNone(entities[0].parent)
    self.assertIsNone(entities[1].parent)
    self.assertIs(entities[2].parent, entities[1])
    self.assertIs(entities[3].parent, entities[0])

    self.assertIsNone(entities[0].mjcf_model.parent_model)
    self.assertIsNone(entities[1].mjcf_model.parent_model)
    self.assertIs(entities[2].mjcf_model.parent_model, entities[1].mjcf_model)
    self.assertIs(entities[3].mjcf_model.parent_model, entities[0].mjcf_model)

    self.assertEqual(list(entities[0].iter_entities()),
                     [entities[0], entities[3]])

  def testIterEntitiesExcludeSelf(self):
    entities = [TestEntity() for _ in range(4)]
    entities[0].attach(entities[1])
    entities[1].attach(entities[2])
    entities[0].attach(entities[3])
    self.assertEqual(
        list(entities[0].iter_entities(exclude_self=True)), entities[1:])

  def testGlobalVectorToLocalFrame(self):
    parent = TestEntity()
    parent.mjcf_model.worldbody.add(
        'site', xyaxes=[0, 1, 0, -1, 0, 0]).attach(self.entity.mjcf_model)
    physics = mjcf.Physics.from_mjcf_model(parent.mjcf_model)

    # 3D vectors
    np.testing.assert_allclose(
        self.entity.global_vector_to_local_frame(physics, [0, 1, 0]),
        [1, 0, 0], atol=1e-10)
    np.testing.assert_allclose(
        self.entity.global_vector_to_local_frame(physics, [-1, 0, 0]),
        [0, 1, 0], atol=1e-10)
    np.testing.assert_allclose(
        self.entity.global_vector_to_local_frame(physics, [0, 0, 1]),
        [0, 0, 1], atol=1e-10)

    # 2D vectors; z-component is ignored
    np.testing.assert_allclose(
        self.entity.global_vector_to_local_frame(physics, [0, 1]),
        [1, 0], atol=1e-10)
    np.testing.assert_allclose(
        self.entity.global_vector_to_local_frame(physics, [-1, 0]),
        [0, 1], atol=1e-10)

  def testGlobalMatrixToLocalFrame(self):
    parent = TestEntity()
    parent.mjcf_model.worldbody.add(
        'site', xyaxes=[0, 1, 0, -1, 0, 0]).attach(self.entity.mjcf_model)
    physics = mjcf.Physics.from_mjcf_model(parent.mjcf_model)

    rotation_atob = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
    ego_rotation_atob = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])

    np.testing.assert_allclose(
        self.entity.global_xmat_to_local_frame(physics, rotation_atob),
        ego_rotation_atob, atol=1e-10)

    flat_rotation_atob = np.reshape(rotation_atob, -1)
    flat_rotation_ego_atob = np.reshape(ego_rotation_atob, -1)
    np.testing.assert_allclose(
        self.entity.global_xmat_to_local_frame(
            physics, flat_rotation_atob),
        flat_rotation_ego_atob, atol=1e-10)

  @parameterized.parameters(*_param_product(
      position=[None, [1., 0., -1.]],
      quaternion=[None, _FORTYFIVE_DEGREES_ABOUT_X, _NINETY_DEGREES_ABOUT_Z],
      freejoint=[False, True],
  ))
  def testSetPose(self, position, quaternion, freejoint):
    # Setup entity.
    test_arena = arena.Arena()
    subentity = TestEntity(name='subentity')
    frame = test_arena.attach(subentity)
    if freejoint:
      frame.add('freejoint')

    physics = mjcf.Physics.from_mjcf_model(test_arena.mjcf_model)

    if quaternion is None:
      ground_truth_quat = _NO_ROTATION
    else:
      ground_truth_quat = quaternion

    if position is None:
      ground_truth_pos = np.zeros(shape=(3,))
    else:
      ground_truth_pos = position

    subentity.set_pose(physics, position=position, quaternion=quaternion)

    np.testing.assert_array_equal(physics.bind(frame).xpos, ground_truth_pos)
    np.testing.assert_array_equal(physics.bind(frame).xquat, ground_truth_quat)

  @parameterized.parameters(*_param_product(
      original_position=[[-2, -1, -1.], [1., 0., -1.]],
      position=[None, [1., 0., -1.]],
      original_quaternion=_TEST_ROTATIONS[0],
      quaternion=_TEST_ROTATIONS[1],
      expected_quaternion=_TEST_ROTATIONS[2],
      freejoint=[False, True],
  ))
  def testShiftPose(self, original_position, position, original_quaternion,
                    quaternion, expected_quaternion, freejoint):
    # Setup entity.
    test_arena = arena.Arena()
    subentity = TestEntity(name='subentity')
    frame = test_arena.attach(subentity)
    if freejoint:
      frame.add('freejoint')

    physics = mjcf.Physics.from_mjcf_model(test_arena.mjcf_model)

    # Set the original position
    subentity.set_pose(
        physics, position=original_position, quaternion=original_quaternion)

    if position is None:
      ground_truth_pos = original_position
    else:
      ground_truth_pos = original_position + np.array(position)
    subentity.shift_pose(physics, position=position, quaternion=quaternion)
    np.testing.assert_array_equal(physics.bind(frame).xpos, ground_truth_pos)

    updated_quat = physics.bind(frame).xquat
    np.testing.assert_array_almost_equal(updated_quat, expected_quaternion,
                                         1e-4)

  @parameterized.parameters(False, True)
  def testShiftPoseWithVelocity(self, rotate_velocity):
    # Setup entity.
    test_arena = arena.Arena()
    subentity = TestEntity(name='subentity')
    frame = test_arena.attach(subentity)
    frame.add('freejoint')

    physics = mjcf.Physics.from_mjcf_model(test_arena.mjcf_model)

    # Set the original position
    subentity.set_pose(physics, position=[0., 0., 0.])

    # Set velocity in y dim.
    subentity.set_velocity(physics, [0., 1., 0.])

    # Rotate the entity around the z axis.
    subentity.shift_pose(
        physics, quaternion=[0., 0., 0., 1.], rotate_velocity=rotate_velocity)

    physics.forward()
    updated_position, _ = subentity.get_pose(physics)
    if rotate_velocity:
      # Should not have moved in the y dim.
      np.testing.assert_array_almost_equal(updated_position[1], 0.)
    else:
      # Should not have moved in the x dim.
      np.testing.assert_array_almost_equal(updated_position[0], 0.)


if __name__ == '__main__':
  absltest.main()
