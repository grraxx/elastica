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
from scipy.optimize import brentq
from scipy.special import ellipe, ellipj, ellipk


def first_clamped_mode(d_over_L, s):
    """
    Exact first clamped-clamped mode of the elastica for phi1 = phi2 = 0.

    With theta(0) = theta(L) = 0 the equilibrium equation reduces to the
    pendulum equation theta'' + fx sin(theta) = 0 (fy = 0). Its first mode
    is four quarter-waves of
        theta(s) = 2 asin(p sn(4 K s / L | m)),   m = p^2,
    which satisfies int(sin theta) = 0 exactly, with
        d / L = 2 E(m) / K(m) - 1,   fx = (4 K(m) / L)^2.
    m is found from d / L by root finding; this holds for any -1 < d/L < 1,
    not only for small deflections.

    Parameters
    ----------
    d_over_L : float
        End-to-end distance over rod length.
    s : np.ndarray
        Normalised arc length s / L in [0, 1] at which to evaluate theta.

    Returns
    -------
    (theta, fx * L^2) : (np.ndarray, float)
    """
    if d_over_L >= 1.0:
        return np.zeros_like(s), 4.0 * np.pi ** 2
    if d_over_L <= -1.0:
        raise ValueError("d / L must be greater than -1")
    m = brentq(lambda m: 2.0 * ellipe(m) / ellipk(m) - 1.0 - d_over_L,
               0.0, 1.0 - 1e-15, xtol=1e-15)
    K = ellipk(m)
    sn = ellipj(4.0 * K * s, m)[0]
    return 2.0 * np.arcsin(np.sqrt(m) * sn), 16.0 * K ** 2


@dataclass
class ElasticaSolution:
    theta: np.ndarray     # angle dof values on V (a copy)
    fx: float             # horizontal force multiplier
    fy: float             # vertical   force multiplier
    energy: float         # bending energy 0.5 * int(theta'^2)
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
        self.L = self.xL - self.x0
        # Normalised arc length s / L of every theta dof.
        self.s = (V.tabulate_dof_coordinates()[:, 0] - self.x0) / self.L

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
        self.energy_form = fem.form(
            0.5 * ufl.inner(ufl.grad(self.theta), ufl.grad(self.theta)) * ufl.dx
        )

        dofs_l = fem.locate_dofs_geometrical(
            V, lambda x: np.isclose(x[0], self.x0)
        )
        dofs_r = fem.locate_dofs_geometrical(
            V, lambda x: np.isclose(x[0], self.xL)
        )
        self.bcs = [fem.dirichletbc(self.phi1_value, dofs_l, V),
                    fem.dirichletbc(self.phi2_value, dofs_r, V)]
        self.forms = [F0, F1, F2]

        self.opts = {"snes_type": "newtonls", "snes_linesearch_type": "bt",
                     "snes_linesearch_order": 3, "snes_rtol": 1e-8,
                     "snes_atol": 1e-10, "snes_max_it": 100,
                     "ksp_type": "preonly", "pc_type": "lu",
                     "pc_factor_mat_solver_type": "mumps"}
        if petsc_options is not None:
            self.opts.update(petsc_options)
        self._create_problem()

    def _create_problem(self):
        self.problem = NonlinearProblem(
            self.forms, [self.theta, self.fx, self.fy],
            petsc_options_prefix=f"elastica_{uuid.uuid4().hex[:8]}_",
            bcs=self.bcs, petsc_options=self.opts,
        )

    def solve(self, d, phi1, phi2, theta0, fx0=0.0, fy0=0.0, max_it=None):
        """Solve at (d, phi1, phi2) from the initial guess theta0 (dof values on V).

        max_it overrides the SNES iteration limit for this solve only.
        Raises RuntimeError if SNES diverges.
        """
        self.d_value.value = PETSc.ScalarType(d)
        self.phi1_value.value = PETSc.ScalarType(phi1)
        self.phi2_value.value = PETSc.ScalarType(phi2)

        self.theta.x.array[:] = theta0
        self.theta.x.scatter_forward()
        self.fx.x.array[:] = fx0
        self.fy.x.array[:] = fy0

        snes = self.problem.solver
        snes.setTolerances(max_it=int(max_it or self.opts["snes_max_it"]))
        try:
            self.problem.solve()
        except PETSc.Error as error:
            # A PETSc error aborts SNESSolve with its vectors still locked, so
            # every later solve would fail too; start again from a fresh problem.
            self._create_problem()
            raise RuntimeError(f"PETSc error {error.ierr} during SNES solve") from error
        reason = snes.getConvergedReason()
        if reason <= 0:
            raise RuntimeError(f"SNES did not converge (reason={reason})")

        return ElasticaSolution(
            self.theta.x.array.copy(),
            float(self.fx.x.array[0]),
            float(self.fy.x.array[0]),
            float(fem.assemble_scalar(self.energy_form)),
            int(reason),
            int(snes.getIterationNumber()),
        )

    def cold_solve(self, d, phi1, phi2, max_step=0.25, min_step=0.02,
                   step_max_it=15):
        """Solve at (d, phi1, phi2) with no prior solution to start from.

        Starts on the exact first clamped mode at (d, 0, 0) (see
        first_clamped_mode), then continues the clamp angles along a straight
        line to (phi1, phi2), in steps of at most max_step radians. A step
        that does not converge within step_max_it Newton iterations is
        halved, down to min_step. Each step's guess is the previous solution
        plus a linear ramp for the change in end angles.

        Both signs of the first mode (hump up and hump down) are continued,
        and the lower-energy result is returned, so the answer does not
        depend on which branch the ramp happens to follow.
        Raises RuntimeError if neither branch reaches the target.
        """
        theta_mode, fx_mode = first_clamped_mode(d / self.L, self.s)
        fx_mode /= self.L ** 2
        target = np.array([phi1, phi2], dtype=float)
        distance = float(np.linalg.norm(target))
        best, errors = None, []
        for sign in (1.0, -1.0):
            try:
                solution = self.solve(d, 0.0, 0.0, sign * theta_mode, fx_mode, 0.0)
                t = 0.0
                step = 1.0 if distance == 0.0 else min(1.0, max_step / distance)
                full_step = step
                while t < 1.0:
                    t_next = min(1.0, t + step)
                    delta = (t_next - t) * target
                    guess = (solution.theta + delta[0] * (1.0 - self.s)
                             + delta[1] * self.s)
                    try:
                        solution = self.solve(d, *(t_next * target), guess,
                                              solution.fx, solution.fy,
                                              max_it=step_max_it)
                    except RuntimeError:
                        step /= 2.0
                        if step * distance < min_step:
                            raise
                        continue
                    t = t_next
                    step = min(full_step, 2.0 * step)
            except RuntimeError as error:
                errors.append(str(error))
                continue
            if best is None or solution.energy < best.energy:
                best = solution
        if best is None:
            raise RuntimeError("cold start failed on both first-mode branches: "
                               + "; ".join(errors))
        return best
