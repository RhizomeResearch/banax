"""Adjoint tests: primal correctness and gradient correctness.

Solver internals (norms, convergence, aux) belong in test_solvers.py.
"""

import jax
import jax.numpy as jnp
import pytest

from banax.solver import Picard, Relaxed
from banax.adjoint import BPTT, JFB, UnrollPhantom, NeumannPhantom, GDEQ
from banax.adjoint import Reversible
from banax.solver import Broyden as BroydenSolver

from conftest import (
    ATOL,
    RTOL,
    P_BWD,
    linear,
    affine,
    pytree_f,
    ALL_ADJOINTS,
    ALL_ADJOINT_IDS,
    EXACT_ADJOINTS,
    EXACT_ADJOINT_IDS,
    APPROX_ADJOINTS,
    APPROX_ADJOINT_IDS,
)


# ── TestPrimal ────────────────────────────────────────────────────────────


class TestPrimal:
    @pytest.mark.parametrize("make_adj", ALL_ADJOINTS, ids=ALL_ADJOINT_IDS)
    def test_scalar(self, make_adj):
        a, b = 0.5, 1.0
        sol = make_adj()((linear, (a, b)), jnp.array(0.0))
        assert jnp.allclose(sol.value, b / (1 - a), atol=0.01)

    @pytest.mark.parametrize("make_adj", ALL_ADJOINTS, ids=ALL_ADJOINT_IDS)
    def test_vector(self, make_adj):
        A = jnp.array([[0.0, 0.25], [0.25, 0.0]])
        b = jnp.array([1.0, 2.0])
        sol = make_adj()((affine, (A, b)), jnp.zeros(2))
        expected = jnp.linalg.solve(jnp.eye(2) - A, b)
        assert jnp.allclose(sol.value, expected, atol=0.01)

    @pytest.mark.parametrize("make_adj", ALL_ADJOINTS, ids=ALL_ADJOINT_IDS)
    def test_pytree(self, make_adj):
        x0 = {"a": jnp.array(0.0), "b": jnp.array(0.0)}
        sol = make_adj()(pytree_f, x0)
        assert jnp.allclose(sol.value["a"], 2.0, atol=0.01)
        assert jnp.allclose(sol.value["b"], 4.0, atol=0.01)

    @pytest.mark.parametrize("make_adj", ALL_ADJOINTS, ids=ALL_ADJOINT_IDS)
    def test_n_steps_positive(self, make_adj):
        sol = make_adj()((linear, (0.5, 1.0)), jnp.array(0.0))
        assert sol.stats.steps > 0

    @pytest.mark.parametrize("make_adj", ALL_ADJOINTS, ids=ALL_ADJOINT_IDS)
    def test_step_budget_limits_steps(self, make_adj):
        budget = jnp.array(4)
        sol = make_adj()((linear, (0.5, 1.0)), jnp.array(0.0), step_budget=budget)
        assert int(sol.stats.steps) <= budget

    @pytest.mark.parametrize("make_adj", ALL_ADJOINTS, ids=ALL_ADJOINT_IDS)
    def test_step_budget_none_unchanged(self, make_adj):
        adj = make_adj()
        sol_default = adj((linear, (0.5, 1.0)), jnp.array(0.0))
        sol_none = adj((linear, (0.5, 1.0)), jnp.array(0.0), step_budget=None)
        assert int(sol_default.stats.steps) == int(sol_none.stats.steps)


# ── TestExactGradients ────────────────────────────────────────────────────


class TestExactGradients:
    """Exact methods must reproduce the analytic gradient of x*(a,b) = b/(1-a).

    d(x*²)/da = 2·x*·b/(1-a)² = 16 at a=0.5, b=1
    d(x*²)/db = 2·x*·1/(1-a)  =  8 at a=0.5, b=1

    rev() uses damp=0.5, max_steps=20: primal error ~0.002, gradient error ~0.12.
    """

    @pytest.mark.parametrize("make_adj", EXACT_ADJOINTS, ids=EXACT_ADJOINT_IDS)
    def test_grad_matches_analytic(self, make_adj):
        def loss(a, b):
            sol = make_adj()((linear, (a, b)), jnp.array(0.0))
            return jnp.sum(sol.value**2)

        grad_a, grad_b = jax.grad(loss, argnums=(0, 1))(jnp.array(0.5), jnp.array(1.0))
        assert jnp.allclose(grad_a, 16.0, atol=0.5), f"grad_a={grad_a}"
        assert jnp.allclose(grad_b, 8.0, atol=0.5), f"grad_b={grad_b}"

    def test_exact_adjoints_agree(self):
        """All exact methods must agree on gradients (atol=0.5 covers rev's truncation)."""
        a, b = jnp.array(0.5), jnp.array(1.0)

        def loss(a, b, make_adj):
            sol = make_adj()((linear, (a, b)), jnp.array(0.0))
            return jnp.sum(sol.value**2)

        grads = [jax.grad(loss, argnums=(0, 1))(a, b, m) for m in EXACT_ADJOINTS]
        g_ref = grads[0]  # bptt as reference
        for g in grads[1:]:
            assert jnp.allclose(g[0], g_ref[0], atol=0.5)
            assert jnp.allclose(g[1], g_ref[1], atol=0.5)


# ── TestApproximateGradients ──────────────────────────────────────────────


class TestApproximateGradients:
    @pytest.mark.parametrize("make_adj", APPROX_ADJOINTS, ids=APPROX_ADJOINT_IDS)
    def test_grad_finite_and_nonzero(self, make_adj):
        def loss(a, b):
            sol = make_adj()((linear, (a, b)), jnp.array(0.0))
            return jnp.sum(sol.value**2)

        grad_a, grad_b = jax.grad(loss, argnums=(0, 1))(jnp.array(0.5), jnp.array(1.0))
        assert jnp.isfinite(grad_a) and grad_a != 0.0
        assert jnp.isfinite(grad_b) and grad_b != 0.0

    @pytest.mark.parametrize("make_adj", APPROX_ADJOINTS, ids=APPROX_ADJOINT_IDS)
    def test_primal_unaffected(self, make_adj):
        """Approximate backward must not change the forward fixed point."""
        sol = make_adj()((linear, (0.5, 1.0)), jnp.array(0.0))
        assert jnp.allclose(sol.value, 2.0, atol=1e-4)


# ── TestGradientBias ──────────────────────────────────────────────────────


class TestGradientBias:
    def test_jfb_differs_from_bptt(self):
        """On a nonlinear f, JFB and BPTT must give different gradients."""

        def nonlinear(x, w):
            return jnp.tanh(w * x + 0.5)

        def loss(make_adj, w):
            solver = Picard(
                atol=1e-6, rtol=0.0, max_steps=100, loop_kind="checkpointed"
            )
            sol = make_adj(solver)((nonlinear, (w,)), jnp.array(0.0))
            return jnp.sum(sol.value**2)

        g_bptt = jax.grad(loss, argnums=1)(lambda s: BPTT(solver=s), jnp.array(1.0))
        g_jfb = jax.grad(loss, argnums=1)(
            lambda s: JFB(solver=Picard(atol=1e-6, rtol=0.0, max_steps=100)),
            jnp.array(1.0),
        )
        assert jnp.isfinite(g_bptt)
        assert jnp.isfinite(g_jfb)
        assert not jnp.allclose(g_bptt, g_jfb, atol=1e-3)

    def test_phantom_converges_to_bptt(self):
        """More UnrollPhantom backward steps → gradient closer to BPTT."""
        p = Picard(atol=ATOL, rtol=RTOL, max_steps=50)
        p_bwd = Picard(atol=ATOL, rtol=RTOL, max_steps=50, loop_kind="checkpointed")

        def loss_bptt(w):
            sol = BPTT(solver=p_bwd)((linear, (w, jnp.array(1.0))), jnp.array(0.0))
            return jnp.sum(sol.value**2)

        def loss_phantom(w, n):
            sol = UnrollPhantom(
                solver=p,
                b_solver=Relaxed(
                    damp=1.0, atol=0.0, rtol=0.0, max_steps=n, loop_kind="checkpointed"
                ),
            )((linear, (w, jnp.array(1.0))), jnp.array(0.0))
            return jnp.sum(sol.value**2)

        w = jnp.array(0.5)
        g_true = jax.grad(loss_bptt)(w)
        g_few = jax.grad(loss_phantom)(w, 3)
        g_many = jax.grad(loss_phantom)(w, 20)
        assert jnp.abs(g_many - g_true) < jnp.abs(g_few - g_true)

    def test_neumann_converges_to_bptt(self):
        """More NeumannPhantom backward steps → gradient closer to BPTT."""
        p = Picard(atol=ATOL, rtol=RTOL, max_steps=50)
        p_bwd = Picard(atol=ATOL, rtol=RTOL, max_steps=50, loop_kind="checkpointed")

        def loss_bptt(w):
            sol = BPTT(solver=p_bwd)((linear, (w, jnp.array(1.0))), jnp.array(0.0))
            return jnp.sum(sol.value**2)

        def loss_neumann(w, n):
            sol = NeumannPhantom(
                solver=p,
                b_solver=Relaxed(damp=1.0, atol=0.0, rtol=0.0, max_steps=n),
            )((linear, (w, jnp.array(1.0))), jnp.array(0.0))
            return jnp.sum(sol.value**2)

        w = jnp.array(0.5)
        g_true = jax.grad(loss_bptt)(w)
        g_few = jax.grad(loss_neumann)(w, 3)
        g_many = jax.grad(loss_neumann)(w, 20)
        assert jnp.abs(g_many - g_true) < jnp.abs(g_few - g_true)


# ── TestGDEQ ─────────────────────────────────────────────────────────────


class TestGDEQ:
    def test_grad_better_than_jfb(self):
        """GDEQ gradient should be closer to BPTT than JFB on a nonlinear f.

        We use f(x, w) = tanh(w * x), which has a spectral radius < 1 near x=0
        for |w| < 1. Broyden accumulates rank-1 updates during the forward solve,
        so GDEQ's B^T preconditioning carries more Jacobian information than
        the identity (JFB baseline).
        """

        def nonlinear(x, w):
            return jnp.tanh(w * x + 0.3)

        w = jnp.array(0.7)
        x0 = jnp.array(0.0)

        p_bwd = BPTT(
            solver=Picard(atol=1e-7, rtol=0.0, max_steps=200, loop_kind="checkpointed")
        )

        def loss(adj, w):
            sol = adj((nonlinear, (w,)), x0)
            return jnp.sum(sol.value**2)

        g_bptt = jax.grad(loss, argnums=1)(p_bwd, w)
        g_jfb = jax.grad(loss, argnums=1)(
            JFB(solver=Picard(atol=1e-7, rtol=0.0, max_steps=200)), w
        )
        g_gdeq = jax.grad(loss, argnums=1)(
            GDEQ(solver=BroydenSolver(atol=1e-7, rtol=0.0, max_steps=200)), w
        )

        assert jnp.isfinite(g_gdeq), f"GDEQ gradient is not finite: {g_gdeq}"
        assert jnp.abs(g_gdeq - g_bptt) < jnp.abs(g_jfb - g_bptt), (
            f"GDEQ error={jnp.abs(g_gdeq - g_bptt):.4f} >= "
            f"JFB error={jnp.abs(g_jfb - g_bptt):.4f}"
        )

    def test_rejects_non_broyden_solver(self):
        with pytest.raises(TypeError, match="Broyden"):
            GDEQ(solver=Picard())  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


# ── TestValidation ────────────────────────────────────────────────────────


class TestValidation:
    def test_bptt_rejects_lax_solver(self):
        with pytest.raises(TypeError):
            BPTT(solver=Picard(loop_kind="lax"))

    def test_reversible_adjoint_rejects_picard(self):
        with pytest.raises(TypeError):
            Reversible(solver=Picard())  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    def test_shape_mismatch_raises(self):
        """f_spec that returns different shape from x0 raises ValueError."""

        def bad_f(x):
            return jnp.array([x, x])  # scalar in, vector out

        with pytest.raises(ValueError, match="same structure"):
            BPTT(solver=P_BWD)(bad_f, jnp.array(0.0))
