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

"""Tests for base variation operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import operator
from absl.testing import absltest
from absl.testing import parameterized
from dm_control.composer import variation
from dm_control.composer.variation import deterministic


class VariationTest(parameterized.TestCase):

  def setUp(self):
    self.value_1 = 3
    self.variation_1 = deterministic.Constant(self.value_1)
    self.value_2 = 5
    self.variation_2 = deterministic.Constant(self.value_2)

  @parameterized.parameters(['add', 'sub', 'mul', 'truediv', 'floordiv', 'pow'])
  def test_operator(self, name):
    func = getattr(operator, name)
    self.assertEqual(
        variation.evaluate(func(self.value_1, self.variation_2)),
        func(self.value_1, self.value_2))
    self.assertEqual(
        variation.evaluate(func(self.variation_1, self.value_2)),
        func(self.value_1, self.value_2))
    self.assertEqual(
        variation.evaluate(func(self.variation_1, self.variation_2)),
        func(self.value_1, self.value_2))


if __name__ == '__main__':
  absltest.main()
