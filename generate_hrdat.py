import numpy as np
from math import sqrt
import pymatgen as mg
from pymatgen.io.vasp import Poscar
import scipy

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
N = 10 # TODO choose a big enough N

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
s2 = Sheet([a1,a2,a3], ['C','C'], sites_A, [0], [-N,-N], [N,N], 'graphene')

h = Heterostructure([s1, s2], [0, theta], [0, interlayer_dist])

# Superlattice
t1 = -(m+1) * a1 + (2*m+1) * (a2 - a1)
t2 = (2*m+1) * a1 - m * (a2 - a1)
t3 = interlayer_dist + 20 * np.array([0, 0, 1])

# Find the position of each one of these indices
def get_positions(h, s):
    max_index = h.sheets[s].max_index
    grid_indices = [h.sheets[s].indexToGrid(k) for k in range(max_index)]
    return np.array([h.posAtomGrid(grid_index, s) for grid_index in grid_indices])

def indices_to_supercell(h, t1, t2, t3, origin=[0, 0, 0]):
    """Returns a list of all  inside the given supercell. 
    
    Arguments: 
    h          -- Heterostructure object
    t1, t2, t3 -- supercell lattice vectors
    origin     -- position of the supercell

    Returns:
    the position of each atom in the unit cell
    the image of each Heterostructure site
    the index of each Heterostructure site
    """
    origin = np.array(origin)
    delta = np.full(3, np.pi * 1e-5)
    coords = np.concatenate([get_positions(h, s) for s in range(h.max_sheets)]) # shape: (k,3)
    # translate
    coords = coords - origin + delta
    # convert to frac coords
    frac2cart = np.column_stack([t1, t2, t3])
    cart2frac = np.linalg.inv(frac2cart) # shape: (3,3)
    # for broadcasting reasons do it in a transposed way
    frac_coords = coords @ cart2frac.transpose()

    Rs, unit_frac_coords = np.divmod(frac_coords, 1) # (k, 3), (k, 3)
    
    # Go back to cart coords to remove duplicates
    coords = unit_frac_coords @ frac2cart.transpose() - delta
    coords, inverses = np.unique(coords.round(decimals=1), return_inverse=True, axis=0)
    
    return coords, Rs, inverses

# shift it so that the center of the two sheets is at 1/3 t3
coords, Rs, js = indices_to_supercell(h, t1, t2, t3, origin=[0, 0, interlayer_dist/2 - t3[2]/2])

print(len(coords), natoms)
# assert(len(coords) == natoms)

# Save the POSCAR
struct = mg.Structure([t1, t2, t3], len(coords) * ['C'], coords)
Poscar(struct).write_file("test.vasp")

# Generate the TB hoppings
# /!\ Special case: format: [[intra-bottom, inter-bottom-top], [intra-top, intra-top-bottom]]
xs = h.totalHamiltonian()
intra_bottom, inter_bottom_top = xs[0]
intra_top = xs[1][0]

bottom_coords = get_positions(h, s=0)
top_coords = get_positions(h, s=1)
