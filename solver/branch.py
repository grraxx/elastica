"""
branch.py
Bent-up / bent-down classification of elastica solutions.

The transverse shape y(s) = int_0^s sin(theta) is projected onto the first
clamped-clamped buckling mode w1(s) = 1 - cos(2 pi s / L):

    p = int_0^L y w1 ds / ||w1||,     p > 0: bent up,  p < 0: bent down.

The second mode is antisymmetric and carries no up/down information. p is
unchanged by the mirror (phi1, phi2) -> (-phi2, -phi1), which maps y(s) to
y(L - s), and changes sign under (phi1, phi2) -> (-phi1, -phi2).

Between neighbouring grid points p changes in two ways. On one branch it
changes smoothly: by a small fraction of its size, or, where the rod passes
through an S-shape, through zero. Across a snap the rod jumps to another
equilibrium and p jumps by at least half its size: from bent up to bent down,
or onto the unstable S-shaped family (p near 0, high energy) that Newton can
also land on past a fold. A jump is

    |p_a - p_b| > tol   and   |p_a - p_b| > max(|p_a|, |p_b|) / 2;

the snap surface is where p changes sign with a jump.

p shrinks like sqrt(1 - d) as the rod straightens, so between d layers near
d = 1 a smooth change of p is large compared with p itself. The solver
therefore applies the jump rule to q = p / a(d), where a(d) is |p| of the
exact first clamped mode at (d, 0, 0): q = +-1 there at every d, and a tol on
q is the same relative tolerance at every d (0.17 on q is 0.02 on p at d = 0.9).
"""

from functools import lru_cache

import numpy as np
from scipy.optimize import brentq
from scipy.special import ellipe, ellipj, ellipk


def first_mode_projection(theta, s):
    """p for theta sampled at increasing arc lengths s (last axis of theta)."""
    theta = np.asarray(theta, dtype=float)
    s = np.asarray(s, dtype=float)
    increments = 0.5 * (np.sin(theta[..., 1:]) + np.sin(theta[..., :-1])) * np.diff(s)
    y = np.concatenate(
        (np.zeros(theta.shape[:-1] + (1,)), np.cumsum(increments, axis=-1)), axis=-1
    )
    w1 = 1.0 - np.cos(2.0 * np.pi * (s - s[0]) / (s[-1] - s[0]))
    w1 /= np.sqrt(np.trapezoid(w1 * w1, s))
    return np.trapezoid(y * w1, s, axis=-1)


def is_jump(p_a, p_b, tol):
    """Elementwise: True where p jumps between neighbouring solutions."""
    p_a, p_b = np.asarray(p_a), np.asarray(p_b)
    change = np.abs(p_a - p_b)
    return (change > tol) & (change > 0.5 * np.maximum(np.abs(p_a), np.abs(p_b)))


@lru_cache(maxsize=None)
def _first_mode_amplitude(d):
    m = brentq(lambda m: 2.0 * ellipe(m) / ellipk(m) - 1.0 - d, 0.0, 1.0 - 1e-15,
               xtol=1e-15)
    s = np.linspace(0.0, 1.0, 20001)
    theta = 2.0 * np.arcsin(np.sqrt(m) * ellipj(4.0 * ellipk(m) * s, m)[0])
    return abs(float(first_mode_projection(theta, s)))


def first_mode_amplitude(d):
    """|p| of the exact first clamped mode at (d, 0, 0), for 0 < d < 1 (L = 1)."""
    d = np.asarray(d, dtype=float)
    return np.vectorize(_first_mode_amplitude, otypes=[float])(d)
