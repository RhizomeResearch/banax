"""Solver-internals tests: norms, convergence, per-solver primal, aux.

Gradient correctness belongs in test_adjoints.py.
"""

import jax.numpy as jnp
import pytest

from banax._core import Result
from banax.solver import _abs_err, _rel_err
from banax.solver import Picard, Relaxed, Reversible as ReversibleSolver
from banax.adjoint import BPTT
from banax.adjoint import Reversible

from conftest import (
    ATOL,
    RTOL,
    MAX_STEPS,
    REV_DAMP,
    P_BWD,
    linear,
    affine,
    pytree_f,
    ALL_ADJOINTS,
    ALL_ADJOINT_IDS,
)


# ── TestNorms ─────────────────────────────────────────────────────────────


class TestNorms:
    def test_abs_err_scalar(self):
        x = jnp.array(1.0)
        fx = jnp.array(3.0)
        assert jnp.allclose(_abs_err(x, fx), 2.0)

    def test_abs_err_vector(self):
        x = jnp.array([1.0, 0.0])
        fx = jnp.array([1.0, 3.0])
        assert jnp.allclose(_abs_err(x, fx), 3.0)

    def test_abs_err_pytree(self):
        x = {"a": jnp.array(0.0), "b": jnp.array([0.0, 0.0])}
        fx = {"a": jnp.array(5.0), "b": jnp.array([0.0, 0.0])}
        assert jnp.allclose(_abs_err(x, fx), 5.0)

    def test_abs_err_zero(self):
        x = jnp.array([1.0, 2.0, 3.0])
        assert _abs_err(x, x) == 0.0

    def test_rel_err_scalar(self):
        x = jnp.array(2.0)
        fx = jnp.array(4.0)
        expected = 2.0 / (4.0 + 1e-8)
        assert jnp.allclose(_rel_err(x, fx), expected)

    def test_rel_err_eps_prevents_div_zero(self):
        x = jnp.array(1.0)
        fx = jnp.array(0.0)  # fx ≈ 0 → denominator would be 0 without eps
        err = _rel_err(x, fx)
        assert jnp.isfinite(err)


# ── TestPicard ────────────────────────────────────────────────────────────


class TestPicard:
    def test_scalar_fixed_point(self):
        a, b = 0.5, 1.0
        sol = BPTT(solver=P_BWD)((linear, (a, b)), jnp.array(0.0))
        assert jnp.allclose(sol.value, b / (1 - a), atol=ATOL)

    def test_vector_fixed_point(self):
        A = jnp.array([[0.0, 0.25], [0.25, 0.0]])
        b = jnp.array([1.0, 2.0])
        sol = BPTT(solver=P_BWD)((affine, (A, b)), jnp.zeros(2))
        expected = jnp.linalg.solve(jnp.eye(2) - A, b)
        assert jnp.allclose(sol.value, expected, atol=1e-5)

    def test_pytree_fixed_point(self):
        x0 = {"a": jnp.array(0.0), "b": jnp.array(0.0)}
        sol = BPTT(solver=P_BWD)(pytree_f, x0)
        assert jnp.allclose(sol.value["a"], 2.0, atol=1e-5)
        assert jnp.allclose(sol.value["b"], 4.0, atol=1e-5)


# ── TestRelaxed ───────────────────────────────────────────────────────────


class TestRelaxed:
    def test_scalar_fixed_point(self):
        solver = Relaxed(damp=0.5, atol=ATOL, rtol=RTOL, max_steps=MAX_STEPS)
        sol, _ = solver._solve((linear, (0.5, 1.0)), x0=jnp.array(0.0))
        assert jnp.allclose(sol.value, 2.0, atol=ATOL)

    def test_damp_one_matches_picard(self):
        """Relaxed(damp=1.0) is identical to Picard."""
        picard = Picard(atol=ATOL, rtol=RTOL, max_steps=MAX_STEPS)
        relaxed = Relaxed(damp=1.0, atol=ATOL, rtol=RTOL, max_steps=MAX_STEPS)
        sol_p, _ = picard._solve((linear, (0.5, 1.0)), x0=jnp.array(0.0))
        sol_r, _ = relaxed._solve((linear, (0.5, 1.0)), x0=jnp.array(0.0))
        assert jnp.allclose(sol_p.value, sol_r.value, atol=1e-6)


# ── TestReversibleSolver ──────────────────────────────────────────────────


class TestReversibleSolver:
    def _solver(self, max_steps=MAX_STEPS, loop_kind="lax"):
        return ReversibleSolver(
            damp=REV_DAMP,
            atol=ATOL,
            rtol=RTOL,
            max_steps=max_steps,
            loop_kind=loop_kind,
        )

    def test_scalar_fixed_point(self):
        sol = Reversible(solver=self._solver())((linear, (0.5, 1.0)), jnp.array(0.0))
        assert jnp.allclose(sol.value, 2.0, atol=1e-4)

    def test_vector_fixed_point(self):
        A = jnp.array([[0.0, 0.25], [0.25, 0.0]])
        b = jnp.array([1.0, 2.0])
        sol = Reversible(solver=self._solver())((affine, (A, b)), jnp.zeros(2))
        expected = jnp.linalg.solve(jnp.eye(2) - A, b)
        assert jnp.allclose(sol.value, expected, atol=1e-4)

    def test_pytree_fixed_point(self):
        x0 = {"a": jnp.array(0.0), "b": jnp.array(0.0)}
        sol = Reversible(
            solver=ReversibleSolver(
                damp=REV_DAMP,
                atol=ATOL,
                rtol=RTOL,
                max_steps=MAX_STEPS,
                loop_kind="lax",
            )
        )(pytree_f, x0)
        assert jnp.allclose(sol.value["a"], 2.0, atol=1e-4)
        assert jnp.allclose(sol.value["b"], 4.0, atol=1e-4)


# ── TestConvergence ───────────────────────────────────────────────────────


class TestConvergence:
    def test_atol_only(self):
        sol = BPTT(
            solver=Picard(
                atol=1e-4, rtol=0.0, max_steps=MAX_STEPS, loop_kind="checkpointed"
            )
        )((linear, (0.5, 1.0)), jnp.array(0.0))
        assert sol.stats.abs_err <= 1e-4

    def test_rtol_only(self):
        sol = BPTT(
            solver=Picard(
                atol=0.0, rtol=1e-4, max_steps=MAX_STEPS, loop_kind="checkpointed"
            )
        )((linear, (0.5, 1.0)), jnp.array(0.0))
        assert sol.stats.rel_err <= 1e-4

    def test_max_steps_is_hard_limit(self):
        sol = BPTT(
            solver=Picard(atol=1e-15, rtol=0.0, max_steps=5, loop_kind="checkpointed")
        )((linear, (0.5, 1.0)), jnp.array(0.0))
        assert sol.stats.steps <= 5

    def test_atol_rtol_both_zero_runs_max_steps(self):
        """With both tols disabled, loop runs until max_steps."""
        max_s = 7
        sol = BPTT(
            solver=Picard(atol=0.0, rtol=0.0, max_steps=max_s, loop_kind="checkpointed")
        )((linear, (0.5, 1.0)), jnp.array(0.0))
        assert sol.stats.steps == max_s


# ── TestAux ───────────────────────────────────────────────────────────────


class TestAux:
    @pytest.mark.parametrize("make_adj", ALL_ADJOINTS, ids=ALL_ADJOINT_IDS)
    def test_aux_tracks_last_iterate(self, make_adj):
        def aux_update(step, aux, x, fx, state):
            return x

        sol = make_adj()(
            (linear, (0.5, 1.0)),
            jnp.array(0.0),
            aux_update=aux_update,
            aux_init=jnp.array(0.0),
        )
        # aux should hold last iterate, which is close to x*=2.0
        assert jnp.isfinite(sol.aux)
        assert sol.aux > 1.0  # converging toward 2.0

    @pytest.mark.parametrize("make_adj", ALL_ADJOINTS, ids=ALL_ADJOINT_IDS)
    def test_aux_counts_steps(self, make_adj):
        def aux_update(step, aux, x, fx, state):
            return aux + 1

        sol = make_adj()(
            (linear, (0.5, 1.0)),
            jnp.array(0.0),
            aux_update=aux_update,
            aux_init=jnp.array(0),
        )
        assert jnp.array_equal(sol.aux, sol.stats.steps)

    @pytest.mark.parametrize("make_adj", ALL_ADJOINTS, ids=ALL_ADJOINT_IDS)
    def test_aux_pytree(self, make_adj):
        def aux_update(step, aux, x, fx, state):
            return {"last_x": x, "count": aux["count"] + 1}

        aux_init = {"last_x": jnp.array(0.0), "count": jnp.array(0)}
        sol = make_adj()(
            (linear, (0.5, 1.0)),
            jnp.array(0.0),
            aux_update=aux_update,
            aux_init=aux_init,
        )
        assert jnp.isfinite(sol.aux["last_x"])
        assert sol.aux["count"] > 0

    @pytest.mark.parametrize("make_adj", ALL_ADJOINTS, ids=ALL_ADJOINT_IDS)
    def test_no_aux_is_none(self, make_adj):
        sol = make_adj()((linear, (0.5, 1.0)), jnp.array(0.0))
        assert sol.aux is None

    @pytest.mark.parametrize("make_adj", ALL_ADJOINTS, ids=ALL_ADJOINT_IDS)
    def test_aux_provided_is_not_none(self, make_adj):
        sol = make_adj()(
            (linear, (0.5, 1.0)),
            jnp.array(0.0),
            aux_update=lambda step, aux, x, fx, state: x,
            aux_init=jnp.array(0.0),
        )
        assert sol.aux is not None


# ── TestSolution ──────────────────────────────────────────────────────────


class TestSolution:
    def test_solution_fields(self):
        sol = BPTT(solver=P_BWD)((linear, (0.5, 1.0)), jnp.array(0.0))
        assert hasattr(sol, "value")
        assert hasattr(sol, "result")
        assert hasattr(sol, "stats")
        assert hasattr(sol, "aux")
        assert hasattr(sol.stats, "steps")
        assert hasattr(sol.stats, "abs_err")
        assert hasattr(sol.stats, "rel_err")

    def test_result_converged(self):
        sol = BPTT(
            solver=Picard(
                atol=1e-6, rtol=0.0, max_steps=MAX_STEPS, loop_kind="checkpointed"
            )
        )((linear, (0.5, 1.0)), jnp.array(0.0))
        assert int(sol.result) == Result.CONVERGED

    def test_result_max_steps(self):
        sol = BPTT(
            solver=Picard(atol=1e-15, rtol=0.0, max_steps=3, loop_kind="checkpointed")
        )((linear, (0.5, 1.0)), jnp.array(0.0))
        assert int(sol.result) == Result.MAX_STEPS

    def test_nan_stops_iteration(self):
        """NaN guard stops iteration when x becomes non-finite.

        With atol=rtol=0, no convergence criterion fires — only _tree_allfinite
        can stop the loop. x*10 overflows float32 at ~step 39, making x=Inf,
        which triggers the guard.
        """
        solver = Picard(atol=0.0, rtol=0.0, max_steps=200, loop_kind="lax")
        sol, _ = solver._solve(lambda x: x * 10.0, x0=jnp.array(1.0))
        assert not jnp.isfinite(sol.value)
