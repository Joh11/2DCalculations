import numpy as np
from math import sqrt
import pymatgen as mg
from pymatgen.io.vasp import Poscar

from heterostructure import Heterostructure
from sheet import Sheet

# -----------------------------------------------------------------------------
# Input parameters
# -----------------------------------------------------------------------------

supercell_lattice = ...
m = 7
theta = np.arccos((3 * m**2 + 3 * m + 1/2) / (3 * m**2 + 3 * m + 1))


# -----------------------------------------------------------------------------
# Beginning of the script
# -----------------------------------------------------------------------------

# Generate the heterostructure
N = 15

a = 2.46
a1 = np.array([a, 0, 0])
a2 = np.array([a/2, sqrt(3)/2 * a, 0])
a3 = np.array([0, 0, 1])

sites_A = np.array([0   * a1 + 0   * a2,
                    1/3 * a1 + 1/3 * a2])

sites_B = np.array([1/3 * a1 + 1/3 * a2,
                    2/3 * a1 + 2/3 * a2])

s1 = Sheet([a1,a2,a3], ['C','C'], sites_A, [0], [-N,-N], [N,N], 'graphene')
s2 = Sheet([a1,a2,a3], ['C','C'], sites_B, [0], [-N,-N], [N,N], 'graphene')

h = Heterostructure([s1, s2], [0, -theta], [0, 3.35])

# Generate the TB hoppings
# /!\ Special case: format: [[intra-bottom, inter-bottom-top], [intra-top, intra-top-bottom]]
xs = h.totalHamiltonian()
intra_bottom, inter_bottom_top = xs[0]
intra_top = xs[1][0]

# Find the position of each one of these indices
def get_positions(h, s):
    max_index = h.sheets[s].max_index
    grid_indices = [h.sheets[s].indexToGrid(k) for k in range(max_index)]
    return np.array([h.posAtomGrid(grid_index, s) for grid_index in grid_indices])

bottom_coords = get_positions(h, s=0)
top_coords = get_positions(h, s=1)

# Match these positions to ones of the reference POSCAR
struct = mg.Structure.from_file('m7.vasp')

# Shift the sites s.t. bottom -> bottom of the struct
bottom_z_struct = np.min(struct.cart_coords[:, 2])
bottom_coords[:, 2] = bottom_coords[:, 2] + bottom_z_struct
top_coords[:, 2] = top_coords[:, 2] + bottom_z_struct

def hetstruct_to_poscar(h, filename='scarr.vasp'):
    coords = [get_positions(h, s) for s in range(h.max_sheets)]
    coords = np.concatenate(coords, axis=0)
    nsites = len(coords)
    s = mg.Structure(np.array([[100, 0, 0],
                               [0, 100, 0],
                               [0, 0, 20]]),
                     nsites * ['C'],
                     coords + np.array([0, 0, 10 - 3.335 / 2]),# + np.array([50, 50, 2.5]),
                     coords_are_cartesian=True)
    Poscar(s).write_file(filename)
hetstruct_to_poscar(h)
