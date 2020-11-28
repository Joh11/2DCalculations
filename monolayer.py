#!/usr/bin/env python3
"""Generate TB for monolayer graphene."""

import numpy as np
from math import sqrt

from sheet import Sheet
from intralayers import Intralayer

N = 20 # TODO check what's enough

a = 2.46
a1 = np.array([a, 0, 0])
a2 = np.array([a/2, sqrt(3)/2 * a, 0])
a3 = np.array([0, 0, 1])

sites = np.array([0   * a1 + 0   * a2,
                  1/3 * a1 + 1/3 * a2])

sheet = Sheet([a1,a2,a3], ['C','C'], sites, [0], [-N,-N], [N,N], 'graphene')
