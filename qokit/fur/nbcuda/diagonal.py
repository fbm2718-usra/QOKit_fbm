###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import math
import numba.cuda
from qokit.fur.nbcuda.fbm_monkey_patch import __global_grid_size__


@numba.cuda.jit
def apply_diagonal_kernel(sv, gamma, diag):
    n = len(sv)
    tid = numba.cuda.grid(__global_grid_size__)
    if tid < n:
        x = 0.5 * gamma * diag[tid]
        sv[tid] *= math.cos(x) - 1j * math.sin(x)


def apply_diagonal(sv, gamma, diag):
    apply_diagonal_kernel.forall(len(sv))(sv, gamma, diag)
