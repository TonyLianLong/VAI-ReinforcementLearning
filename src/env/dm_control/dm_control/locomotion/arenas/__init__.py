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
"""Arenas for Locomotion tasks."""

from dm_control.locomotion.arenas.bowl import Bowl
from dm_control.locomotion.arenas.corridors import EmptyCorridor
from dm_control.locomotion.arenas.corridors import GapsCorridor
from dm_control.locomotion.arenas.corridors import WallsCorridor
from dm_control.locomotion.arenas.floors import Floor
from dm_control.locomotion.arenas.labmaze_textures import FloorTextures
from dm_control.locomotion.arenas.labmaze_textures import SkyBox
from dm_control.locomotion.arenas.labmaze_textures import WallTextures
from dm_control.locomotion.arenas.mazes import MazeWithTargets
from dm_control.locomotion.arenas.mazes import RandomMazeWithTargets
