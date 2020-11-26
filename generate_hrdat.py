import numpy as np
from math import sqrt
import pymatgen as mg
from pymatgen.io.vasp import Poscar

from heterostructure import Heterostructure
from sheet import Sheet

# -----------------------------------------------------------------------------
# Input parameters
# -----------------------------------------------------------------------------

m = 7

# -----------------------------------------------------------------------------
# Beginning of the script
# -----------------------------------------------------------------------------

# Generate the heterostructure
N = 40 # TODO choose a big enough N

a = 2.46
interlayer_dist = 3.35
a1 = np.array([a, 0, 0])
a2 = np.array([a/2, sqrt(3)/2 * a, 0])
a3 = np.array([0, 0, 1])

sites_A = np.array([0   * a1 + 0   * a2,
                    1/3 * a1 + 1/3 * a2])

sites_B = np.array([1/3 * a1 + 1/3 * a2,
                    2/3 * a1 + 2/3 * a2])

theta = np.arccos((3 * m**2 + 3 * m + 1/2) / (3 * m**2 + 3 * m + 1))
natoms_per_layer = 2 * (3 * m**2 + 3 * m + 1)
natoms = 2 * natoms_per_layer

s1 = Sheet([a1,a2,a3], ['C','C'], sites_A, [0], [-N,-N], [N,N], 'graphene')
s2 = Sheet([a1,a2,a3], ['C','C'], sites_B, [0], [-N,-N], [N,N], 'graphene')

h = Heterostructure([s1, s2], [0, theta], [0, interlayer_dist])

# Superlattice
t1 = -(m+1) * a1 + (2*m+1) * a2
t2 = (2*m+1) * a1 - m * a2
t3 = interlayer_dist + 20 * np.array([0, 0, 1])

# Find the position of each one of these indices
def get_positions(h, s):
    max_index = h.sheets[s].max_index
    grid_indices = [h.sheets[s].indexToGrid(k) for k in range(max_index)]
    return np.array([h.posAtomGrid(grid_index, s) for grid_index in grid_indices])

def indices_in_supercell(h, t1, t2, t3, origin=[0, 0, 0]):
    """Returns a list of all (global) indices inside the given supercell. 
    
    Arguments: 
    h          -- Heterostructure object
    t1, t2, t3 -- supercell lattice vectors
    origin     -- position of the supercell
    """
    origin = np.array(origin)
    idcs = np.arange(h.max_index) # shape: (k,)
    coords = np.concatenate([get_positions(h, s) for s in range(h.max_sheets)]) # shape: (k,3)
    # translate
    coords = coords - origin
    # convert to frac coords
    mat = np.linalg.inv(np.column_stack([t1, t2, t3])) # shape: (3,3)
    # for broadcasting reasons do it in a transposed way
    frac_coords = coords @ mat.transpose()

    mask = ((frac_coords >= 0) * (frac_coords <= 1)).all(axis=1)
    return idcs[mask]

# shift it so that the center of the two sheets is at 1/3 t3
idcs = indices_in_supercell(h, t1, t2, t3, origin=[0, 0, interlayer_dist/2 - t3[2]/2])

print(len(idcs))
assert(len(idcs) == natoms)

# Generate the TB hoppings
# /!\ Special case: format: [[intra-bottom, inter-bottom-top], [intra-top, intra-top-bottom]]
xs = h.totalHamiltonian()
intra_bottom, inter_bottom_top = xs[0]
intra_top = xs[1][0]

bottom_coords = get_positions(h, s=0)
top_coords = get_positions(h, s=1)



# Find all the indices that are used

