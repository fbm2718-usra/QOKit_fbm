###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
# pyright: reportGeneralTypeIssues=false
# pyright seems to be upset about numba code
import numba.cuda
from qokit.fur.nbcuda.fbm_monkey_patch import __global_grid_size__

@numba.cuda.jit
def zero_init_kernel(x):
    tid = numba.cuda.grid(__global_grid_size__)

    if tid < len(x):
        x[tid] = 0


def zero_init(x):
    zero_init_kernel.forall(len(x))(x)


@numba.cuda.jit
def compute_costs_kernel(costs, coef: float, pos_mask: int, offset: int):
    tid = numba.cuda.grid(__global_grid_size__)

    if tid < len(costs):
        parity = numba.cuda.popc((tid + offset) & pos_mask) & 1
        if parity:
            costs[tid] -= coef
        else:
            costs[tid] += coef


def compute_costs(rank: int,
                  n_local_qubits: int,
                  terms, 
                  out,
                  #TODO(FBM): I modified this so non-qiskit convention can be used
                  first_qubit_first_bit = False
                  ):
    offset = rank << n_local_qubits
    n = len(out)
    zero_init_kernel.forall(n)(out)

    if first_qubit_first_bit:
        for coef, pos in terms:
            pos_mask = sum(2**(n_local_qubits-x-1) for x in pos)
            compute_costs_kernel.forall(n)(out, coef, pos_mask, offset)
    else:
        for coef, pos in terms:
            pos_mask = sum(2**x for x in pos)
            compute_costs_kernel.forall(n)(out, coef, pos_mask, offset)

