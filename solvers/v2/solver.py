"""
solver.py
Elastica solver based on working_elastica_v2.ipynb
"""

import uuid
from dataclasses import dataclass
import numpy as np
import ufl, basix.ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem
from dolfinx.fem.petsc import NonlinearProblem


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
                "snes_linesearch_order": 3, "snes_rtol": 1e-10,
                "snes_atol": 1e-12, "snes_max_it": 100,
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
            self.theta.interpolate(
                lambda x: phi1 + (phi2 - phi1) * (x[0] - self.x0)
                / (self.xL - self.x0)
            )
        elif np.isscalar(theta0):
            self.theta.x.array[:] = float(theta0)
        else:
            self.theta.x.array[:] = theta0.x.array
            self.theta.x.scatter_forward()
        self.fx.x.array[:] = fx0
        self.fy.x.array[:] = fy0

        self.problem.solve()
        reason = self.problem.solver.getConvergedReason()
        if reason <= 0:
            raise RuntimeError(f"SNES did not converge (reason={reason})")

        theta_result = fem.Function(self.V)
        theta_result.x.array[:] = self.theta.x.array
        theta_result.x.scatter_forward()
        return ElasticaSolution(
            theta_result, float(self.fx.x.array[0]), float(self.fy.x.array[0]),
            int(reason), int(self.problem.solver.getIterationNumber()),
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
        If None (default), a linear phi1->phi2 ramp is used -- a non-flat guess that
        keeps the Jacobian nonsingular. (A flat theta0 == 0 is singular; see docs.)
    fx0, fy0 : float
        Initial guesses for the force multipliers.
    petsc_options : dict, optional
        Overrides/extends the default SNES options.

    Returns
    -------
    ElasticaSolution
        theta (solved Function), fx, fy (floats), SNES reason and iteration count.
        Raises RuntimeError if the solve does not converge.
    """
    domain, comm = V.mesh, V.mesh.comm

    # endpoints of the beam (MPI-safe), needed for the BCs and the default ramp
    xg = domain.geometry.x[:, 0]
    x0 = comm.allreduce(xg.min(), op=MPI.MIN)
    xL = comm.allreduce(xg.max(), op=MPI.MAX)

    # initial guess: a Function (warm start) or callable is interpolated; a scalar fills
    theta, fx, fy = fem.Function(V), fem.Function(R), fem.Function(R)
    if theta0 is None:
        theta0 = lambda x: phi1 + (phi2 - phi1) * (x[0] - x0) / (xL - x0)  # linear ramp
    if np.isscalar(theta0):
        theta.x.array[:] = float(theta0)
    else:
        theta.interpolate(theta0)
    fx.x.array[:] = fx0
    fy.x.array[:] = fy0

    # --- blocked weak form: one residual per test function ---
    v, qx, qy = ufl.TestFunction(V), ufl.TestFunction(R), ufl.TestFunction(R)
    F0 = (ufl.inner(ufl.grad(theta), ufl.grad(v))
          - fx * ufl.sin(theta) * v + fy * ufl.cos(theta) * v) * ufl.dx  # d/dtheta
    F1 = (ufl.cos(theta) - d) * qx * ufl.dx                              # d/dfx
    F2 = ufl.sin(theta) * qy * ufl.dx                                    # d/dfy

    # --- clamped-angle BCs at the mesh endpoints ---
    dofs_l = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], x0))
    dofs_r = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], xL))
    bcs = [fem.dirichletbc(PETSc.ScalarType(phi1), dofs_l, V),
           fem.dirichletbc(PETSc.ScalarType(phi2), dofs_r, V)]

    # --- SNES: Newton LS w/ cubic backtracking; preonly + LU (MUMPS pivots the saddle block) ---
    opts = {"snes_type": "newtonls", "snes_linesearch_type": "bt",
            "snes_linesearch_order": 3, "snes_rtol": 1e-10, "snes_atol": 1e-12,
            "snes_max_it": 100, "ksp_type": "preonly", "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps"}
    if petsc_options is not None:
        opts.update(petsc_options)

    problem = NonlinearProblem([F0, F1, F2], [theta, fx, fy],
                               petsc_options_prefix=f"elastica_{uuid.uuid4().hex[:8]}_",
                               bcs=bcs, petsc_options=opts)
    problem.solve()
    reason = problem.solver.getConvergedReason()
    if reason <= 0:
        raise RuntimeError(f"SNES did not converge (reason={reason})")
    return ElasticaSolution(theta, float(fx.x.array[0]), float(fy.x.array[0]),
                            int(reason), int(problem.solver.getIterationNumber()))