#!/usr/bin/env python3
"""Generate TB for monolayer graphene."""

import numpy as np
from math import sqrt
import sys
import scipy.sparse as sparse
from collections import deque


from sheet import Sheet

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

def fill_hoppings(xs, hoppings, i, idx, Rai=0, Rbi=0):
    for _, idxbond, valbond in zip(*sparse.find(xs.getrow(idx))):
        Ra, Rb, j = sheet.indexToGrid(idxbond)
        Ra -= Rai
        Rb -= Rbi
        hoppings.setdefault((Ra, Rb, 0), deque()).append((i, j, valbond, 0))

                
# -----------------------------------------------------------------------------

hrdat_path = 'monolayer-stcarr.dat' if len(sys.argv) <= 1 else sys.argv[1]

N = 20 # TODO check what's enough

a = 2.46
a1 = np.array([a, 0, 0])
a2 = np.array([a/2, sqrt(3)/2 * a, 0])
a3 = np.array([0, 0, 1])

sites = np.array([0   * a1 + 0   * a2,
                  1/3 * a1 + 1/3 * a2])

sheet = Sheet([a1,a2,a3], ['C','C'], sites, [0], [-N,-N], [N,N], 'graphene')
xs = sheet.intraHamiltonian(0) # this argument is useless btw

# Now generate the hoppings
hoppings = {}

# Generate for the A site
fill_hoppings(xs, hoppings, 0, sheet.gridToIndex((0, 0, 0)))
# Generate for the B site
fill_hoppings(xs, hoppings, 1, sheet.gridToIndex((0, 0, 1)))
    
# save_sparse_hoppings(hrdat_path, hoppings, num_wann)


# -----------------------------------------------------------------------------
# Plot the atoms with their labels in 2D
# -----------------------------------------------------------------------------
