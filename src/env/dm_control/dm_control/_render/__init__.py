# Copyright 2017-2018 The dm_control Authors.
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

"""OpenGL context management for rendering MuJoCo scenes.

By default, the `Renderer` class will try to load one of the following rendering
APIs, in descending order of priority: GLFW > EGL > OSMesa.

It is also possible to select a specific backend by setting the `MUJOCO_GL=`
environment variable to 'glfw', 'egl', or 'osmesa'.
"""

import collections
import os

from absl import logging
from dm_control._render import constants

BACKEND = os.environ.get(constants.MUJOCO_GL)


# pylint: disable=g-import-not-at-top
def _import_egl():
  from dm_control._render.pyopengl.egl_renderer import EGLContext
  return EGLContext


def _import_glfw():
  from dm_control._render.glfw_renderer import GLFWContext
  return GLFWContext


def _import_osmesa():
  from dm_control._render.pyopengl.osmesa_renderer import OSMesaContext
  return OSMesaContext
# pylint: enable=g-import-not-at-top

_ALL_RENDERERS = collections.OrderedDict([
    (constants.GLFW, _import_glfw),
    (constants.EGL, _import_egl),
    (constants.OSMESA, _import_osmesa),
])


if BACKEND is not None:
  # If a backend was specified, try importing it and error if unsuccessful.
  try:
    import_func = _ALL_RENDERERS[BACKEND]
  except KeyError:
    raise RuntimeError(
        'Environment variable {} must be one of {!r}: got {!r}.'
        .format(constants.MUJOCO_GL, _ALL_RENDERERS.keys(), BACKEND))
  logging.info('MUJOCO_GL=%s, attempting to import specified OpenGL backend.',
               BACKEND)
  Renderer = import_func()  # pylint: disable=invalid-name
else:
  logging.info('MUJOCO_GL is not set, so an OpenGL backend will be chosen '
               'automatically.')
  # Otherwise try importing them in descending order of priority until
  # successful.
  for name, import_func in _ALL_RENDERERS.items():
    try:
      Renderer = import_func()
      BACKEND = name
      logging.info('Successfully imported OpenGL backend: %s', name)
      break
    except ImportError:
      logging.info('Failed to import OpenGL backend: %s', name)
  if BACKEND is None:
    logging.info('No OpenGL backend could be imported. Attempting to create a '
                 'rendering context will result in a RuntimeError.')

    def Renderer(*args, **kwargs):  # pylint: disable=function-redefined,invalid-name
      del args, kwargs
      raise RuntimeError('No OpenGL rendering backend is available.')

USING_GPU = BACKEND in (constants.EGL, constants.GLFW)
