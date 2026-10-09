"""CPU PETSc solver for the constrained planar elastica."""

import uuid
from dataclasses import dataclass

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.fem.petsc import NonlinearProblem


def first_mode_cold_start(phi1, phi2, d, x0, xL, mode=0):
    """Build a parameterized elastica initial guess for theta(s)."""
    length = xL - x0
    if length <= 0:
        raise ValueError("Degenerate beam domain: xL <= x0")
    amplitude = 2.0 * np.sqrt(max(0.0, 1.0 - d / length))
    modes = {
        0: lambda t: 0.0,
        1: lambda t: amplitude * np.sin(np.pi * t),
        2: lambda t: -amplitude * np.sin(np.pi * t),
        3: lambda t: 0.5 * amplitude * np.sin(np.pi * t),
        4: lambda t: -0.5 * amplitude * np.sin(np.pi * t),
        5: lambda t: 0.5 * amplitude * np.sin(2.0 * np.pi * t),
        6: lambda t: -0.5 * amplitude * np.sin(2.0 * np.pi * t),
    }
    if mode not in modes:
        raise ValueError(f"Unknown cold-start mode {mode}")

    def profile(x):
        normalized_coordinate = (x[0] - x0) / length
        return (phi1 + (phi2 - phi1) * normalized_coordinate
                + modes[mode](normalized_coordinate))

    return profile


@dataclass
class ElasticaSolution:
    theta: np.ndarray
    fx: float
    fy: float
    reason: int
    iterations: int


class ElasticaSolver:
    """Reuse one CPU PETSc nonlinear problem for a parameter sequence."""

    def __init__(self, V, R, petsc_options=None):
        self.V = V
        domain, comm = V.mesh, V.mesh.comm
        coordinates = domain.geometry.x[:, 0]
        self.x0 = comm.allreduce(coordinates.min(), op=MPI.MIN)
        self.xL = comm.allreduce(coordinates.max(), op=MPI.MAX)
        self.beam_order = np.argsort(coordinates)

        self.theta = fem.Function(V)
        self.fx = fem.Function(R)
        self.fy = fem.Function(R)
        self.d_value = fem.Constant(domain, PETSc.ScalarType(0.0))
        self.phi1_value = fem.Constant(domain, PETSc.ScalarType(0.0))
        self.phi2_value = fem.Constant(domain, PETSc.ScalarType(0.0))

        test_theta = ufl.TestFunction(V)
        test_fx = ufl.TestFunction(R)
        test_fy = ufl.TestFunction(R)
        forms = [
            (ufl.inner(ufl.grad(self.theta), ufl.grad(test_theta))
             - self.fx * ufl.sin(self.theta) * test_theta
             + self.fy * ufl.cos(self.theta) * test_theta) * ufl.dx,
            (ufl.cos(self.theta) - self.d_value) * test_fx * ufl.dx,
            ufl.sin(self.theta) * test_fy * ufl.dx,
        ]
        dofs_l = fem.locate_dofs_geometrical(
            V, lambda x: np.isclose(x[0], self.x0)
        )
        dofs_r = fem.locate_dofs_geometrical(
            V, lambda x: np.isclose(x[0], self.xL)
        )
        bcs = [
            fem.dirichletbc(self.phi1_value, dofs_l, V),
            fem.dirichletbc(self.phi2_value, dofs_r, V),
        ]

        options = {
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_linesearch_order": 3,
            "snes_rtol": 1e-10,
            "snes_atol": 1e-12,
            "snes_max_it": 100,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
        if petsc_options is not None:
            options.update(petsc_options)
        self.problem = NonlinearProblem(
            forms,
            [self.theta, self.fx, self.fy],
            petsc_options_prefix=f"elastica_{uuid.uuid4().hex[:8]}_",
            bcs=bcs,
            petsc_options=options,
        )

    def solve(
        self, d, phi1, phi2, theta0=None, fx0=0.0, fy0=0.0, cold_start_mode=1
    ) -> ElasticaSolution:
        self.d_value.value = PETSc.ScalarType(d)
        self.phi1_value.value = PETSc.ScalarType(phi1)
        self.phi2_value.value = PETSc.ScalarType(phi2)

        if theta0 is None:
            self.theta.interpolate(
                first_mode_cold_start(
                    phi1, phi2, d, self.x0, self.xL, cold_start_mode
                )
            )
        elif np.isscalar(theta0):
            self.theta.x.array[:] = float(theta0)
        else:
            initial_values = theta0.x.array if hasattr(theta0, "x") else theta0
            self.theta.x.array[self.beam_order] = np.asarray(initial_values)
        self.theta.x.scatter_forward()
        self.fx.x.array[:] = fx0
        self.fy.x.array[:] = fy0

        try:
            self.problem.solve()
        except PETSc.Error as error:
            try:
                self.problem.solver.setUp()
            except PETSc.Error:
                pass
            raise RuntimeError(f"PETSc nonlinear solve failed: {error}") from error
        reason = self.problem.solver.getConvergedReason()
        if reason <= 0:
            raise RuntimeError(f"SNES did not converge (reason={reason})")

        theta_result = self.theta.x.array[self.beam_order].copy()
        return ElasticaSolution(
            theta_result,
            float(self.fx.x.array[0]),
            float(self.fy.x.array[0]),
            int(reason),
            int(self.problem.solver.getIterationNumber()),
        )


def solve_elastica(
    V, R, d, phi1, phi2, theta0=None, fx0=0.0, fy0=0.0,
    petsc_options=None,
):
    """Solve one constrained planar elastica on prebuilt spaces."""
    return ElasticaSolver(V, R, petsc_options=petsc_options).solve(
        d, phi1, phi2, theta0=theta0, fx0=fx0, fy0=fy0

    )
