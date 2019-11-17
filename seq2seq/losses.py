# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Operations related to calculating sequence losses.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def cross_entropy_sequence_loss(logits, targets, sequence_length):
  """Calculates the per-example cross-entropy loss for a sequence of logits and
    masks out all losses passed the sequence length.

  Args:
    logits: Logits of shape `[T, B, vocab_size]`
    targets: Target classes of shape `[T, B]`
    sequence_length: An int32 tensor of shape `[B]` corresponding
      to the length of each input

  Returns:
    A tensor of shape [T, B] that contains the loss per example, per time step.
  """
  with tf.compat.v1.name_scope("cross_entropy_sequence_loss"):
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=targets)

    # Mask out the losses we don't care about
    loss_mask = tf.sequence_mask(
        tf.cast(sequence_length, dtype=tf.int32), tf.cast(tf.shape(input=targets)[0], dtype=tf.int32))
    losses = losses * tf.transpose(a=tf.cast(loss_mask, dtype=tf.float32), perm=[1, 0])

    return losses
