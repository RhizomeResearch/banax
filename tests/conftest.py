import jax

jax.config.update("jax_numpy_rank_promotion", "raise")
jax.config.update("jax_numpy_dtype_promotion", "strict")


from banax.solver import Picard, Relaxed, Reversible as ReversibleSolver
from banax.adjoint import BPTT, JFB, Implicit, UnrollPhantom, NeumannPhantom
from banax.adjoint import Reversible

# ── Constants ──────────────────────────────────────────────────────────────

ATOL = 1e-6
RTOL = 0.0
MAX_STEPS = 200
# damp=0.5 → 2× amplification per step; 20 steps → 2^20 ≈ 1e6, within float32 range.
# Primal converges to error ~0.002 at 20 steps; gradient error vs analytic ~0.12.
# (damp=0.8 at 5 steps: primal error ~0.044, gradient error ~1.2, cliff at step 10.)
REV_DAMP = 0.5
REV_MAX_STEPS = 20
KEY = jax.random.PRNGKey(0)

# ── Test functions ─────────────────────────────────────────────────────────


def linear(x, a, b):
    """f(x) = a*x + b, fixed point: b/(1-a), requires |a| < 1."""
    return a * x + b


def affine(x, A, b):
    """f(x) = A @ x + b, fixed point: (I-A)^{-1}b, requires ||A|| < 1."""
    return A @ x + b


def pytree_f(x):
    """Fixed point: {a: 2.0, b: 4.0}."""
    return {"a": 0.5 * x["a"] + 1.0, "b": 0.25 * x["b"] + 3.0}


# ── Solver instances ───────────────────────────────────────────────────────

P = Picard(atol=ATOL, rtol=RTOL, max_steps=MAX_STEPS)
P_BWD = Picard(atol=ATOL, rtol=RTOL, max_steps=MAX_STEPS, loop_kind="checkpointed")

# ── Adjoint factories ──────────────────────────────────────────────────────


def bptt():
    return BPTT(solver=P_BWD)


def jfb():
    return JFB(solver=P)


def implicit():
    return Implicit(solver=P, b_solver=P)


def rev():
    return Reversible(
        solver=ReversibleSolver(
            damp=REV_DAMP,
            atol=ATOL,
            rtol=RTOL,
            max_steps=REV_MAX_STEPS,
            loop_kind="lax",
        )
    )


def phantom(n):
    return UnrollPhantom(
        solver=P,
        b_solver=Relaxed(
            damp=REV_DAMP, atol=0.0, rtol=0.0, max_steps=n, loop_kind="checkpointed"
        ),
    )


def neumann(n):
    return NeumannPhantom(
        solver=P,
        b_solver=Relaxed(damp=REV_DAMP, atol=0.0, rtol=0.0, max_steps=n),
    )


# ── Parametrize groups ─────────────────────────────────────────────────────

ALL_ADJOINTS = [bptt, jfb, implicit, rev, lambda: phantom(10), lambda: neumann(10)]
ALL_ADJOINT_IDS = ["bptt", "jfb", "implicit", "rev", "phantom10", "neumann10"]

EXACT_ADJOINTS = [bptt, implicit, rev]
EXACT_ADJOINT_IDS = ["bptt", "implicit", "rev"]

APPROX_ADJOINTS = [jfb, lambda: phantom(10), lambda: neumann(10)]
APPROX_ADJOINT_IDS = ["jfb", "phantom10", "neumann10"]
