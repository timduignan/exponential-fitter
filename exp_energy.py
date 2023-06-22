# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Definitions of various standard energy functions."""

from functools import wraps, partial

from typing import Callable, Tuple, TextIO, Dict, Any, Optional

import re

import jax
import jax.numpy as jnp
from jax import ops
from jax.tree_util import tree_map
from jax import vmap
import haiku as hk
from jax.scipy.special import erfc  # error function
from jax_md import space, smap, partition, nn, quantity, interpolate, util, energy

from ml_collections import ConfigDict


# Define aliases different neural network primitives.
bp = nn.behler_parrinello
gnome = nn.gnome
nequip = nn.nequip

maybe_downcast = util.maybe_downcast


# Types


f32 = util.f32
f64 = util.f64
Array = util.Array

PyTree = Any
Box = space.Box
DisplacementFn = space.DisplacementFn
DisplacementOrMetricFn = space.DisplacementOrMetricFn

NeighborFn = partition.NeighborFn
NeighborList = partition.NeighborList
NeighborListFormat = partition.NeighborListFormat



def exponential_potential(dr: Array,
                          A: Array = 1,
                          B: Array = 1,
                          **unused_kwargs) -> Array:
    """Exponential interaction potential between particles.

    Args:
      dr: An ndarray of shape `[n, m]` of pairwise distances between particles.
      A: Interaction energy scale. Should either be a floating point scalar or
        an ndarray whose shape is `[n, m]`.
      B: Exponential decay constant. Should either be a floating point scalar or
        an ndarray whose shape is `[n, m]`.
      unused_kwargs: Allows extra data (e.g. time) to be passed to the energy.

    Returns:
      Matrix of energies of shape `[n, m]`.
    """
    return A * jnp.exp(- dr/B)

def exponential_potential_pair(displacement_or_metric: DisplacementOrMetricFn,
                               species: Optional[Array] = None,
                               A: Array = 1.0,
                               B: Array = 1.0,
                               r_onset: Array = 2.0,
                               r_cutoff: Array = 2.5,
                               per_particle: bool = False) -> Callable[[Array], Array]:
    """Convenience wrapper to compute the exponential potential over a system."""
    A = maybe_downcast(A)
    B = maybe_downcast(B)
    r_onset = maybe_downcast(r_onset)
    r_cutoff = maybe_downcast(r_cutoff)
    return smap.pair(
        energy.multiplicative_isotropic_cutoff(exponential_potential, r_onset, r_cutoff),
        space.canonicalize_displacement_or_metric(displacement_or_metric),
        ignore_unused_parameters=True,
        species=species,
        A=A,
        B=B,
        reduce_axis=(1,) if per_particle else None)
