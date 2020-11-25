import numpy as np
from math import sqrt

from heterostructure import Heterostructure
from sheet import Sheet

# -----------------------------------------------------------------------------
# Input parameters
# -----------------------------------------------------------------------------

supercell_lattice = ...
theta = ...

# -----------------------------------------------------------------------------
# Beginning of the script
# -----------------------------------------------------------------------------

# Generate the heterostructure
N = 15

a = 2.46
a1 = np.array([sqrt(3) / 2 * a, 1/2, 0])
a2 = np.array([sqrt(3) / 2 * a, -1/2, 0])
a3 = np.array([0, 0, 1])

sites_A = np.array([0   * a1 + 0   * a2,
                    2/3 * a1 + 2/3 * a2])

sites_B = np.array([1/3 * a1 + 1/3  * a2,
                    2/3 * a1 + 2/3 * a2])

s1 = Sheet([a1,a2,a3], ['C','C'], sites_A, [0], [-N,-N], [N,N], 'graphene')
s2 = Sheet([a1,a2,a3], ['C','C'], sites_B, [0], [-N,-N], [N,N], 'graphene')

h = Heterostructure([s1, s2], [0, theta], [0, 3.35])

# Generate the TB hoppings
