"""
solver.py
Elastica solver based on working_elastica_v2.ipynb
"""

import uuid
from dataclasses import dataclass
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.fem.petsc import NonlinearProblem


def first_mode_cold_start(phi1, phi2, d, x0, xL):
    """
    Build a first-buckling-mode initial guess for theta(s).

    A plain linear ramp between the two clamped end angles (the old
    default) is a constant-curvature guess -- a circular arc. That is a
    fine initial guess only when the rod is nearly straight (d close to
    L). Once it is genuinely compressed, a circular arc is far from any
    equilibrium shape, and starting Newton's method there routinely
    converges to a spurious/higher buckling mode or fails to converge at
    all -- this is the main source of the solver's numerical trouble.

    Instead we add a half-sine bump on top of the linear ramp:

        theta0(s) = phi1 + (phi2 - phi1) * s/L + A * sin(pi * s / L)

    sin(pi*s/L) is exactly the n=1 buckling shape: a single hump with no
    internal zero crossing, and it vanishes at both ends so the clamped
    BCs stay satisfied exactly by the initial guess. The amplitude A
    comes from the small-angle elastica relation cos(theta) ~= 1 -
    theta^2/2, integrated against this mode shape, which gives
    d/L ~= 1 - A^2/4, i.e.

        A = 2 * sqrt(max(0.0, 1 - d/L))

    so the initial bow grows as the rod is compressed further and
    shrinks to ~0 as d -> L (nearly straight) -- always the fundamental
    mode, never a higher harmonic.
    """
    L = xL - x0
    if L <= 0:
        raise ValueError("Degenerate beam domain: xL <= x0")
    amplitude = 2.0 * np.sqrt(max(0.0, 1.0 - d / L))

    def profile(x):
        s = (x[0] - x0) / L
        return phi1 + (phi2 - phi1) * s + amplitude * np.sin(np.pi * s)

    return profile


@dataclass
class ElasticaSolution:
    theta: fem.Function   # angle field on V
    fx: float             # horizontal force multiplier
    fy: float             # vertical   force multiplier
    reason: int           # SNES converged reason (>0 = converged)
    iterations: int       # Newton iterations


class ElasticaSolver:
    """Reuse the nonlinear problem for a sequence of parameter points."""

    def __init__(self, V, R, petsc_options=None):
        self.V = V
        domain, comm = V.mesh, V.mesh.comm
        xg = domain.geometry.x[:, 0]
        self.x0 = comm.allreduce(xg.min(), op=MPI.MIN)
        self.xL = comm.allreduce(xg.max(), op=MPI.MAX)

        self.theta = fem.Function(V)
        self.fx = fem.Function(R)
        self.fy = fem.Function(R)
        self.d_value = fem.Constant(domain, PETSc.ScalarType(0.0))
        self.phi1_value = fem.Constant(domain, PETSc.ScalarType(0.0))
        self.phi2_value = fem.Constant(domain, PETSc.ScalarType(0.0))

        v, qx, qy = ufl.TestFunction(V), ufl.TestFunction(R), ufl.TestFunction(R)
        F0 = (ufl.inner(ufl.grad(self.theta), ufl.grad(v))
              - self.fx * ufl.sin(self.theta) * v
              + self.fy * ufl.cos(self.theta) * v) * ufl.dx
        F1 = (ufl.cos(self.theta) - self.d_value) * qx * ufl.dx
        F2 = ufl.sin(self.theta) * qy * ufl.dx

        dofs_l = fem.locate_dofs_geometrical(
            V, lambda x: np.isclose(x[0], self.x0)
        )
        dofs_r = fem.locate_dofs_geometrical(
            V, lambda x: np.isclose(x[0], self.xL)
        )
        bcs = [fem.dirichletbc(self.phi1_value, dofs_l, V),
               fem.dirichletbc(self.phi2_value, dofs_r, V)]

        opts = {"snes_type": "newtonls", "snes_linesearch_type": "bt",
                "snes_linesearch_order": 3, "snes_rtol": 1e-8,
                "snes_atol": 1e-10, "snes_max_it": 100,
                "ksp_type": "preonly", "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps"}
        if petsc_options is not None:
            opts.update(petsc_options)
        self.problem = NonlinearProblem(
            [F0, F1, F2], [self.theta, self.fx, self.fy],
            petsc_options_prefix=f"elastica_{uuid.uuid4().hex[:8]}_",
            bcs=bcs, petsc_options=opts,
        )


    def solve(self, d, phi1, phi2, theta0=None, fx0=0.0, fy0=0.0):
        self.d_value.value = PETSc.ScalarType(d)
        self.phi1_value.value = PETSc.ScalarType(phi1)
        self.phi2_value.value = PETSc.ScalarType(phi2)

        if theta0 is None:
            # Cold start: seed with the first buckling mode, not a bare
            # linear ramp (see first_mode_cold_start docstring).
            self.theta.interpolate(
                first_mode_cold_start(phi1, phi2, d, self.x0, self.xL)
            )
            self.theta.x.scatter_forward()

        elif np.isscalar(theta0):
            self.theta.x.array[:] = float(theta0)
            self.theta.x.scatter_forward()

        elif isinstance(theta0, np.ndarray):
            self.theta.x.array[:] = theta0
            self.theta.x.scatter_forward()
        else:
            self.theta.x.array[:] = theta0.x.array
            self.theta.x.scatter_forward()

        # Initial guesses for the Lagrange multipliers
        self.fx.x.array[:] = fx0
        self.fy.x.array[:] = fy0

        # Solve nonlinear system
        self.problem.solve()

        reason = self.problem.solver.getConvergedReason()

        if reason <= 0:
            raise RuntimeError(
                f"SNES did not converge (reason={reason})"
            )

        # Copy solution so it is not overwritten by the next solve
        theta_result = fem.Function(self.V)
        theta_result.x.array[:] = self.theta.x.array
        theta_result.x.scatter_forward()

        return ElasticaSolution(
            theta_result,
            float(self.fx.x.array[0]),
            float(self.fy.x.array[0]),
            int(reason),
            int(self.problem.solver.getIterationNumber()),
        )



def solve_elastica(V, R, d, phi1, phi2, theta0=None, fx0=0.0, fy0=0.0, petsc_options=None):
    """Solve the constrained planar elastica on prebuilt function spaces.

    Minimizes  0.5*int(theta'^2)  subject to  int(cos theta) = d  and  int(sin theta) = 0,
    with clamped angles  theta = phi1 at s=0  and  theta = phi2 at s=L
    (the endpoints are read from the mesh extent).

    Parameters
    ----------
    V, R : dolfinx.fem.FunctionSpace
        V : Lagrange space for the angle theta.
        R : real (global-scalar) space -- one dof -- used for each force multiplier.
        Both are built once outside and reused across solves (same mesh assumed).
    d, phi1, phi2 : float
        Constraint values: target end-to-end distance and the two clamped angles.
    theta0 : fem.Function | callable | float, optional
        Initial guess for theta. Pass a previous solution `sol.theta` (a Function on
        the same mesh) to warm-start; a callable x->array or a scalar also work.
        If None (default), theta is seeded with the first buckling mode
        (see `first_mode_cold_start`), not a bare linear ramp.
    fx0, fy0 : float
        Initial guesses for the force multipliers.
    petsc_options : dict, optional
        Overrides/extends the default SNES options.

    Returns
    -------
    ElasticaSolution
        theta (solved Function), fx, fy (floats), SNES reason and iteration count.
        Raises RuntimeError if the solve does not converge.

    Notes
    -----
    This is a thin convenience wrapper around `ElasticaSolver` for one-off
    solves. It used to duplicate the whole weak form / BC / SNES setup
    inline, which meant the cold-start logic could (and did) drift out of
    sync between the two code paths. If you're solving many (d, phi1,
    phi2) points, build an `ElasticaSolver` once yourself and call
    `.solve()` repeatedly -- it reuses the assembled problem instead of
    rebuilding it every call.
    """
    return ElasticaSolver(V, R, petsc_options=petsc_options).solve(
        d, phi1, phi2, theta0=theta0, fx0=fx0, fy0=fy0
    )