"""
stability.py
Stability of a clamped elastica equilibrium under prescribed end displacement.

The solver finds stationary points of

    Pi = 1/2 int theta'^2 ds + fx (int cos(theta) ds - d) + fy int sin(theta) ds,

with theta(0) = phi1, theta(L) = phi2. The equilibrium is stable when the
second variation

    Q[v] = int v'^2 - (fx cos(theta) + fy sin(theta)) v^2 ds

is positive for every v with v(0) = v(L) = 0 that keeps the two constraints,
int sin(theta) v = int cos(theta) v = 0. Newton converges to unstable
equilibria as readily as to stable ones; a jump in p shows when it lands on a
different branch, but not when it follows one branch smoothly past the point
where it loses stability. That happens on the anti-diagonal phi1 = -phi2,
where the rod is symmetric (fy = 0) and the bent-up branch loses stability at
a symmetry-breaking bifurcation instead of a fold.

Q is discretised with P1 elements (lumped mass) on the beam nodes. The number
of negative eigenvalues of the constrained form follows from the inertia of
the bordered matrix [[A, C], [C^T, 0]] (Haynsworth):

    neg(constrained) = neg(A) + pos(C^T A^-1 C) - 2,

where A is the tridiagonal unconstrained form and C holds the two constraint
vectors. neg(A) is the number of negative pivots of A's LDL^T factorisation,
so the whole test is O(N).
"""

import numpy as np
from scipy.linalg import solve_banded


def unstable_directions(theta, s, fx, fy):
    """Number of independent directions in which the equilibrium is unstable.

    theta: values at increasing arc lengths s (ends included); fx, fy: the
    Lagrange multipliers of the solver's weak form. 0 means stable.
    """
    theta = np.asarray(theta, dtype=float)
    h = np.diff(s)
    lumped = 0.5 * (h[:-1] + h[1:])                     # interior nodes
    t = theta[1:-1]
    diagonal = 1.0 / h[:-1] + 1.0 / h[1:] - lumped * (fx * np.cos(t) + fy * np.sin(t))
    off = -1.0 / h[1:-1]
    # Negative pivots of LDL^T (Sylvester's law of inertia).
    negative = 0
    pivot = diagonal[0]
    for k in range(1, len(diagonal)):
        negative += pivot < 0.0
        pivot = diagonal[k] - off[k - 1] ** 2 / pivot
    negative += pivot < 0.0
    banded = np.zeros((3, len(diagonal)))
    banded[0, 1:] = off
    banded[1] = diagonal
    banded[2, :-1] = off
    C = np.column_stack([lumped * np.cos(t), lumped * np.sin(t)])
    S = C.T @ solve_banded((1, 1), banded, C)
    positive = int(np.count_nonzero(np.linalg.eigvalsh(0.5 * (S + S.T)) > 0.0))
    return int(negative) + positive - 2
