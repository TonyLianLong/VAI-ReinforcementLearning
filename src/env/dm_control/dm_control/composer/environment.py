# Copyright 2018-2019 The dm_control Authors.
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

"""RL environment classes for Composer tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import weakref

from absl import logging
from dm_control import mjcf
from dm_control.composer import observation
from dm_control.rl import control
import dm_env
import numpy as np
from six.moves import range

warnings.simplefilter('always', DeprecationWarning)

_STEPS_LOGGING_INTERVAL = 10000

HOOK_NAMES = ('initialize_episode_mjcf',
              'after_compile',
              'initialize_episode',
              'before_step',
              'before_substep',
              'after_substep',
              'after_step')

_empty_function = lambda: None


def _empty_function_with_docstring():
  """Some docstring."""

_EMPTY_CODE = _empty_function.__code__.co_code
_EMPTY_WITH_DOCSTRING_CODE = _empty_function_with_docstring.__code__.co_code


def _callable_is_trivial(f):
  return (f.__code__.co_code == _EMPTY_CODE or
          f.__code__.co_code == _EMPTY_WITH_DOCSTRING_CODE)


class EpisodeInitializationError(RuntimeError):
  """Raised by a `composer.Task` when it fails to initialize an episode."""


class _Hook(object):

  __slots__ = ('entity_hooks', 'extra_hooks')

  def __init__(self):
    self.entity_hooks = []
    self.extra_hooks = []


class _EnvironmentHooks(object):
  """Helper object that scans and memoizes various hooks in a task.

  This object exist to ensure that we do not incur a substantial overhead in
  calling empty entity hooks in more complicated tasks.
  """

  __slots__ = (('_task', '_episode_step_count') +
               tuple('_' + hook_name for hook_name in HOOK_NAMES))

  def __init__(self, task):
    self._task = task
    self._episode_step_count = 0
    for hook_name in HOOK_NAMES:
      slot_name = '_' + hook_name
      setattr(self, slot_name, _Hook())
    self.refresh_entity_hooks()

  def refresh_entity_hooks(self):
    """Scans and memoizes all non-trivial entity hooks."""
    for hook_name in HOOK_NAMES:
      hooks = []
      for entity in self._task.root_entity.iter_entities():
        entity_hook = getattr(entity, hook_name)
        # Ignore any hook that is a no-op to avoid function call overhead.
        if not _callable_is_trivial(entity_hook):
          hooks.append(entity_hook)
      getattr(self, '_' + hook_name).entity_hooks = hooks

  def add_extra_hook(self, hook_name, hook_callable):
    if hook_name not in HOOK_NAMES:
      raise ValueError('{!r} is not a valid hook name'.format(hook_name))
    if not callable(hook_callable):
      raise ValueError('{!r} is not a callable'.format(hook_callable))
    getattr(self, '_' + hook_name).extra_hooks.append(hook_callable)

  def initialize_episode_mjcf(self, random_state):
    self._task.initialize_episode_mjcf(random_state)
    for entity_hook in self._initialize_episode_mjcf.entity_hooks:
      entity_hook(random_state)
    for extra_hook in self._initialize_episode_mjcf.extra_hooks:
      extra_hook(random_state)

  def after_compile(self, physics, random_state):
    self._task.after_compile(physics, random_state)
    for entity_hook in self._after_compile.entity_hooks:
      entity_hook(physics, random_state)
    for extra_hook in self._after_compile.extra_hooks:
      extra_hook(physics, random_state)

  def initialize_episode(self, physics, random_state):
    self._episode_step_count = 0
    self._task.initialize_episode(physics, random_state)
    for entity_hook in self._initialize_episode.entity_hooks:
      entity_hook(physics, random_state)
    for extra_hook in self._initialize_episode.extra_hooks:
      extra_hook(physics, random_state)

  def before_step(self, physics, action, random_state):
    self._episode_step_count += 1
    if self._episode_step_count % _STEPS_LOGGING_INTERVAL == 0:
      logging.info('The current episode has been running for %d steps.',
                   self._episode_step_count)
    self._task.before_step(physics, action, random_state)
    for entity_hook in self._before_step.entity_hooks:
      entity_hook(physics, random_state)
    for extra_hook in self._before_step.extra_hooks:
      extra_hook(physics, action, random_state)

  def before_substep(self, physics, action, random_state):
    self._task.before_substep(physics, action, random_state)
    for entity_hook in self._before_substep.entity_hooks:
      entity_hook(physics, random_state)
    for extra_hooks in self._before_substep.extra_hooks:
      extra_hooks(physics, action, random_state)

  def after_substep(self, physics, random_state):
    self._task.after_substep(physics, random_state)
    for entity_hook in self._after_substep.entity_hooks:
      entity_hook(physics, random_state)
    for extra_hook in self._after_substep.extra_hooks:
      extra_hook(physics, random_state)

  def after_step(self, physics, random_state):
    self._task.after_step(physics, random_state)
    for entity_hook in self._after_step.entity_hooks:
      entity_hook(physics, random_state)
    for extra_hook in self._after_step.extra_hooks:
      extra_hook(physics, random_state)


class _CommonEnvironment(object):
  """Common components for RL environments."""

  def __init__(self, task, time_limit=float('inf'), random_state=None,
               n_sub_steps=None,
               raise_exception_on_physics_error=True,
               strip_singleton_obs_buffer_dim=False):
    """Initializes an instance of `_CommonEnvironment`.

    Args:
      task: Instance of `composer.base.Task`.
      time_limit: (optional) A float, the time limit in seconds beyond which an
        episode is forced to terminate.
      random_state: Optional, either an int seed or an `np.random.RandomState`
        object. If None (default), the random number generator will self-seed
        from a platform-dependent source of entropy.
      n_sub_steps: (DEPRECATED) An integer, number of physics steps to take per
        agent control step. New code should instead override the
        `control_substep` property of the task.
      raise_exception_on_physics_error: (optional) A boolean, indicating whether
        `PhysicsError` should be raised as an exception. If `False`, physics
        errors will result in the current episode being terminated with a
        warning logged, and a new episode started.
      strip_singleton_obs_buffer_dim: (optional) A boolean, if `True`,
        the array shape of observations with `buffer_size == 1` will not have a
        leading buffer dimension.
    """
    self._task = task
    if not isinstance(random_state, np.random.RandomState):
      self._random_state = np.random.RandomState(random_state)
    else:
      self._random_state = random_state
    self._hooks = _EnvironmentHooks(self._task)
    self._time_limit = time_limit
    self._raise_exception_on_physics_error = raise_exception_on_physics_error
    self._strip_singleton_obs_buffer_dim = strip_singleton_obs_buffer_dim

    if n_sub_steps is not None:
      warnings.simplefilter('once', DeprecationWarning)
      warnings.warn('The `n_sub_steps` argument is deprecated. Please override '
                    'the `control_timestep` property of the task instead.',
                    DeprecationWarning)
    self._overridden_n_sub_steps = n_sub_steps

    self._recompile_physics_and_update_observables()

  def add_extra_hook(self, hook_name, hook_callable):
    self._hooks.add_extra_hook(hook_name, hook_callable)

  def _recompile_physics_and_update_observables(self):
    """Sets up the environment for latest MJCF model from the task."""
    self._physics_proxy = None
    self._recompile_physics()
    if isinstance(self._physics, weakref.ProxyType):
      self._physics_proxy = self._physics
    else:
      self._physics_proxy = weakref.proxy(self._physics)

    if self._overridden_n_sub_steps is not None:
      self._n_sub_steps = self._overridden_n_sub_steps
    else:
      self._n_sub_steps = self._task.physics_steps_per_control_step

    self._hooks.refresh_entity_hooks()
    self._hooks.after_compile(self._physics_proxy, self._random_state)
    self._observation_updater = self._make_observation_updater()
    self._observation_updater.reset(self._physics_proxy, self._random_state)

  def _recompile_physics(self):
    """Creates a new Physics using the latest MJCF model from the task."""
    if getattr(self, '_physics', None):
      self._physics.free()
    self._physics = mjcf.Physics.from_mjcf_model(
        self._task.root_entity.mjcf_model)

  def _make_observation_updater(self):
    return observation.Updater(
        self._task.observables, self._task.physics_steps_per_control_step,
        self._strip_singleton_obs_buffer_dim)

  @property
  def physics(self):
    """Returns a `weakref.ProxyType` pointing to the current `mjcf.Physics`.

    Note that the underlying `mjcf.Physics` will be destroyed whenever the MJCF
    model is recompiled. It is therefore unsafe for external objects to hold a
    reference to `environment.physics`. Attempting to access attributes of a
    dead `Physics` instance will result in a `ReferenceError`.
    """
    return self._physics_proxy

  @property
  def task(self):
    return self._task

  @property
  def random_state(self):
    return self._random_state

  def control_timestep(self):
    """Returns the interval between agent actions in seconds."""
    if self._overridden_n_sub_steps is not None:
      return self.physics.timestep() * self._overridden_n_sub_steps
    else:
      return self.task.control_timestep


class Environment(_CommonEnvironment, dm_env.Environment):
  """Reinforcement learning environment for Composer tasks."""

  def __init__(self, task, time_limit=float('inf'), random_state=None,
               n_sub_steps=None,
               raise_exception_on_physics_error=True,
               strip_singleton_obs_buffer_dim=False,
               max_reset_attempts=1):
    """Initializes an instance of `Environment`.

    Args:
      task: Instance of `composer.base.Task`.
      time_limit: (optional) A float, the time limit in seconds beyond which
        an episode is forced to terminate.
      random_state: (optional) an int seed or `np.random.RandomState` instance.
      n_sub_steps: (DEPRECATED) An integer, number of physics steps to take per
        agent control step. New code should instead override the
        `control_substep` property of the task.
      raise_exception_on_physics_error: (optional) A boolean, indicating whether
        `PhysicsError` should be raised as an exception. If `False`, physics
        errors will result in the current episode being terminated with a
        warning logged, and a new episode started.
      strip_singleton_obs_buffer_dim: (optional) A boolean, if `True`,
        the array shape of observations with `buffer_size == 1` will not have a
        leading buffer dimension.
      max_reset_attempts: (optional) Maximum number of times to try resetting
        the environment. If an `EpisodeInitializationError` is raised
        during this process, an environment reset is reattempted up to this
        number of times. If this count is exceeded then the most recent
        exception will be allowed to propagate. Defaults to 1, i.e. no failure
        is allowed.
    """
    super(Environment, self).__init__(
        task=task,
        time_limit=time_limit,
        random_state=random_state,
        n_sub_steps=n_sub_steps,
        raise_exception_on_physics_error=raise_exception_on_physics_error,
        strip_singleton_obs_buffer_dim=strip_singleton_obs_buffer_dim)
    self._max_reset_attempts = max_reset_attempts
    self._reset_next_step = True

  def reset(self):
    failed_attempts = 0
    while True:
      try:
        return self._reset_attempt()
      except EpisodeInitializationError as e:
        failed_attempts += 1
        if failed_attempts < self._max_reset_attempts:
          logging.error('Error during episode reset: %s', repr(e))
        else:
          raise

  def _reset_attempt(self):
    self._hooks.initialize_episode_mjcf(self._random_state)
    self._recompile_physics_and_update_observables()
    with self._physics.reset_context():
      self._hooks.initialize_episode(self._physics_proxy, self._random_state)
    self._observation_updater.reset(self._physics_proxy, self._random_state)
    self._reset_next_step = False
    return dm_env.TimeStep(
        step_type=dm_env.StepType.FIRST,
        reward=None,
        discount=None,
        observation=self._observation_updater.get_observation())

  # TODO(b/129061424): Remove this method.
  def step_spec(self):
    """DEPRECATED: please use `reward_spec` and `discount_spec` instead."""
    warnings.warn('`step_spec` is deprecated, please use `reward_spec` and '
                  '`discount_spec` instead.', DeprecationWarning)
    if (self._task.get_reward_spec() is None or
        self._task.get_discount_spec() is None):
      raise NotImplementedError
    return dm_env.TimeStep(
        step_type=None,
        reward=self._task.get_reward_spec(),
        discount=self._task.get_discount_spec(),
        observation=self._observation_updater.observation_spec(),
    )

  def step(self, action):
    """Updates the environment using the action and returns a `TimeStep`."""
    if self._reset_next_step:
      self._reset_next_step = False
      return self.reset()

    self._hooks.before_step(self._physics_proxy, action, self._random_state)
    self._observation_updater.prepare_for_next_control_step()

    try:
      for i in range(self._n_sub_steps):
        self._hooks.before_substep(self._physics_proxy, action,
                                   self._random_state)
        self._physics.step()
        self._hooks.after_substep(self._physics_proxy, self._random_state)
        # The final observation update must happen after all the hooks in
        # `self._hooks.after_step` is called. Otherwise, if any of these hooks
        # modify the physics state then we might capture an observation that is
        # inconsistent with the final physics state.
        if i < self._n_sub_steps - 1:
          self._observation_updater.update()
      physics_is_divergent = False
    except control.PhysicsError as e:
      if not self._raise_exception_on_physics_error:
        logging.warning(e)
        physics_is_divergent = True
      else:
        raise

    self._hooks.after_step(self._physics_proxy, self._random_state)
    self._observation_updater.update()

    if not physics_is_divergent:
      reward = self._task.get_reward(self._physics_proxy)
      discount = self._task.get_discount(self._physics_proxy)
      terminating = (
          self._task.should_terminate_episode(self._physics_proxy)
          or self._physics.time() >= self._time_limit
      )
    else:
      reward = 0.0
      discount = 0.0
      terminating = True

    obs = self._observation_updater.get_observation()

    if not terminating:
      return dm_env.TimeStep(dm_env.StepType.MID, reward, discount, obs)
    else:
      self._reset_next_step = True
      return dm_env.TimeStep(dm_env.StepType.LAST, reward, discount, obs)

  def action_spec(self):
    """Returns the action specification for this environment."""
    return self._task.action_spec(self._physics_proxy)

  def reward_spec(self):
    """Describes the reward returned by this environment.

    This will be the output of `self.task.reward_spec()` if it is not None,
    otherwise it will be the default spec returned by
    `dm_env.Environment.reward_spec()`.

    Returns:
      A `specs.Array` instance, or a nested dict, list or tuple of
      `specs.Array`s.
    """
    task_reward_spec = self._task.get_reward_spec()
    if task_reward_spec is not None:
      return task_reward_spec
    else:
      return super(Environment, self).reward_spec()

  def discount_spec(self):
    """Describes the discount returned by this environment.

    This will be the output of `self.task.discount_spec()` if it is not None,
    otherwise it will be the default spec returned by
    `dm_env.Environment.discount_spec()`.

    Returns:
      A `specs.Array` instance, or a nested dict, list or tuple of
      `specs.Array`s.
    """
    task_discount_spec = self._task.get_discount_spec()
    if task_discount_spec is not None:
      return task_discount_spec
    else:
      return super(Environment, self).discount_spec()

  def observation_spec(self):
    """Returns the observation specification for this environment.

    Returns:
      An `OrderedDict` mapping observation name to `specs.Array` containing
      observation shape and dtype.
    """
    return self._observation_updater.observation_spec()
