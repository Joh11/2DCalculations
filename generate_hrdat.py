import numpy as np
from math import sqrt
import pymatgen as mg
from pymatgen.io.vasp import Poscar
import scipy
from collections import deque
import sys

from heterostructure import Heterostructure
from sheet import Sheet

# -----------------------------------------------------------------------------
# Input parameters
# -----------------------------------------------------------------------------

m = 7 if len(sys.argv) <= 1 else sys.argv[1]
poscar_path = 'stcarr.vasp' if len(sys.argv) <= 2 else sys.argv[2]
hrdat_path = 'stcarr_hr.dat' if len(sys.argv) <= 3 else sys.argv[3]

# -----------------------------------------------------------------------------
# Beginning of the script
# -----------------------------------------------------------------------------

print('Usage: ./generate_hrdat.py [m] [poscar] [hrdat]')

# Generate the heterostructure
N = 50 # TODO choose a big enough N

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
t1 = -(m+1) * a1 + (2*m+1) * (a2)
t2 = (2*m+1) * a1 - m * (a2)
t3 = np.array([0, 0, interlayer_dist + 20])

# Find the position of each one of these indices
def get_positions(h, s):
    max_index = h.sheets[s].max_index
    grid_indices = [h.sheets[s].indexToGrid(k) for k in range(max_index)]
    return np.array([h.posAtomGrid(grid_index, s) for grid_index in grid_indices])

def unique_coords(coords):
    """Returns the coords without duplicates, along with the inverses
    array"""
    uniques = np.zeros_like(coords)
    inverses = np.zeros(len(coords))
    next_free = 0

    # first step
    uniques[0] = coords[0]
    inverses[0] = 0
    next_free += 1
    
    for i, c in enumerate(coords[1:], start=1):
        dists = np.linalg.norm(uniques[:next_free] - c, axis=1)
        j = np.argmin(dists)
        if dists[j] > 1:
            # append it
            uniques[next_free] = c
            next_free += 1
        inverses[i] = j

    return uniques[:next_free], inverses

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

    coords, inverses = unique_coords(coords)
    return coords, Rs, inverses

# shift it so that the center of the two sheets is at 1/3 t3
coords, Rs, js = indices_to_supercell(h, t1, t2, t3, origin=[0, 0, interlayer_dist/2 - t3[2]/2])

print(len(coords), natoms)
assert(len(coords) == natoms)

# Save the POSCAR
struct = mg.Structure([t1, t2, t3], len(coords) * ['C'], coords, coords_are_cartesian=True)
Poscar(struct).write_file(poscar_path)

# Generate the TB hoppings
# /!\ Special case: format: [[intra-bottom, inter-bottom-top], [inter-top-bottom, intra-top]]
xs = h.totalHamiltonian()
intra_bottom, inter_bottom_top = xs[0]
intra_top = xs[1][1]

# Save the hoppings
def save_sparse_hoppings(path, hoppings, num_wann, comment='Generated from stcarr\'s code'):
    """Save the given hoppings in a file with the sparse hrdat format. 

    Notes about the sparse hrdat format:
    - the fourth line, the number of rpoints, must only count the R
      points with at least one entry
    - the R points must be sorted lexicographically (i.e. first (-1,
      -1, 0), then (-1, 0, 0))
    - index for i and j starts at 1 (but this functions assumes it
      starts at 0 like in Python)

    Arguments: 
    path     -- target path for the hrdat file
    hoppings -- (Ra, Rb, Rc) -> [(i, j, hr, hi)] dictionnary
    num_wann -- number of Wannier functions per unit cell (i.e number
                of sites)
    comment  -- optional comment, first line of the file

    """
    nrpts = len([R for R in hoppings if len(hoppings[R]) != 0])
    with open(path, 'w') as f:
        num_lines = sum([len(hoppings[R]) for R in hoppings])
        f.write(comment+'\n')
        f.write(f'{num_lines}\n')
        f.write(f'{num_wann}\n')
        f.write(f'{nrpts}\n')
        f.write(' '.join(nrpts * ['1']) + '\n')
        for R in sorted(hoppings.keys()):
            for i, j, hr, hi in hoppings[R]:
                f.write(f'{int(R[0])} {int(R[1])} {int(R[2])} {int(i+1)} {int(j+1)} {hr} {hi}\n')

def fill_hoppings(hoppings, hamiltonian, row_shift=0, col_shift=0, include_transpose=False):
    """Fill the hoppings dictionary using the given hamiltonian sparse
    matrix.

    Arguments: 
    hoppings          -- (Ra, Rb, Rc) -> [(i, j, hr, hi)] dictionnary
    hamiltonian       -- (Nr, Nc) matrix
    row_shift         -- shift to convert row idcs to global idcs
    col_shift         -- shift to convert col idcs to global idcs
    include_transpose -- if True also include the -R, j, i bond
    """
    rows, cols, vals = scipy.sparse.find(hamiltonian)
    rows += row_shift
    cols += col_shift
    for r, c, v in zip(rows, cols, vals):
        # skip if rows outside of the unit cell
        R = Rs[r]
        if (R != [0, 0, 0]).any():
            continue
        R, i, j = Rs[c],js[r], js[c]
        R = (R[0], R[1], R[2]) # convert to tuple for key hash
        hoppings.setdefault(R, deque()).append((i, j, v, 0))
        if include_transpose:
            R = (-R[0], -R[1], -R[2])
            hoppings.setdefault(R, deque()).append((j, i, v, 0))

hoppings = {}

# bottom
fill_hoppings(hoppings, intra_bottom)
print('done bottom')
# top
fill_hoppings(hoppings, intra_top,
              row_shift=h.sheets[0].max_index,
              col_shift=h.sheets[0].max_index)
print('done top')
# intra
fill_hoppings(hoppings, inter_bottom_top,
              col_shift=h.sheets[0].max_index,
              include_transpose=True)
print('done inter')
    
save_sparse_hoppings(hrdat_path, hoppings, len(coords))
