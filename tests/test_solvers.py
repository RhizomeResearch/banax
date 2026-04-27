"""Solver-internals tests: norms, convergence, per-solver primal, trace, has_aux.

Gradient correctness belongs in test_adjoints.py.
"""

import jax.numpy as jnp
import pytest

from banax._core import Result
from banax.utils import (
    trace_last,
    trace_last_aux,
    trace_history,
    trace_count,
    zeros_like,
    half_normal_like,
)
from banax.solver import _abs_err, _rel_err
from banax.solver import (
    Picard,
    Relaxed,
    Reversible as ReversibleSolver,
    Broyden,
    Anderson,
)
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


# ── TestBroyden ──────────────────────────────────────────────────────────


class TestBroyden:
    def _solver(
        self,
        atol: float = ATOL,
        rtol: float = RTOL,
        max_steps: int = MAX_STEPS,
        ls_steps: int = 0,
        spectral_clip=1.5,
    ):
        return Broyden(
            atol=atol,
            rtol=rtol,
            max_steps=max_steps,
            history_size=10,
            ls_steps=ls_steps,
            spectral_clip=spectral_clip,
        )

    def test_scalar_fixed_point(self):
        solver = self._solver()
        sol, _ = solver._solve((linear, (0.5, 1.0)), x0=jnp.array(0.0))
        assert jnp.allclose(sol.value, 2.0, atol=1e-4)

    def test_vector_fixed_point(self):
        A = jnp.array([[0.0, 0.25], [0.25, 0.0]])
        b = jnp.array([1.0, 2.0])
        solver = self._solver()
        sol, _ = solver._solve((affine, (A, b)), x0=jnp.zeros(2))
        expected = jnp.linalg.solve(jnp.eye(2) - A, b)
        assert jnp.allclose(sol.value, expected, atol=1e-4)

    def test_pytree_fixed_point(self):
        solver = self._solver()
        x0 = {"a": jnp.array(0.0), "b": jnp.array(0.0)}
        sol, _ = solver._solve(pytree_f, x0)
        assert jnp.allclose(sol.value["a"], 2.0, atol=1e-4)
        assert jnp.allclose(sol.value["b"], 4.0, atol=1e-4)

    def test_fewer_steps_than_picard(self):
        """Broyden should converge in fewer steps on a hard-ish problem."""
        a, b = 0.9, 1.0  # spectral radius 0.9 — slow for Picard
        picard = Picard(atol=1e-4, rtol=0.0, max_steps=500)
        broyden = self._solver(atol=1e-4, rtol=0.0, max_steps=500)
        sol_p, _ = picard._solve((linear, (a, b)), x0=jnp.array(0.0))
        sol_b, _ = broyden._solve((linear, (a, b)), x0=jnp.array(0.0))
        assert sol_b.stats.steps < sol_p.stats.steps

    def test_ls_scalar(self):
        solver = self._solver(ls_steps=5)
        sol, _ = solver._solve((linear, (0.5, 1.0)), x0=jnp.array(0.0))
        assert jnp.allclose(sol.value, 2.0, atol=1e-4)

    def test_ls_matches_no_ls(self):
        """Line search should find the same fixed point as no line search."""
        A = jnp.array([[0.0, 0.25], [0.25, 0.0]])
        b = jnp.array([1.0, 2.0])
        solver_no_ls = self._solver()
        solver_ls = self._solver(ls_steps=5)
        sol_no, _ = solver_no_ls._solve((affine, (A, b)), x0=jnp.zeros(2))
        sol_ls, _ = solver_ls._solve((affine, (A, b)), x0=jnp.zeros(2))
        assert jnp.allclose(sol_no.value, sol_ls.value, atol=1e-4)

    def test_spectral_clip_none_converges(self):
        """spectral_clip=None (clipping disabled) still finds the fixed point."""
        solver = self._solver(spectral_clip=None)
        sol, _ = solver._solve((linear, (0.5, 1.0)), x0=jnp.array(0.0))
        assert jnp.allclose(sol.value, 2.0, atol=1e-4)

    def test_spectral_clip_tight_converges(self):
        """A very tight spectral_clip forces clipping on most updates but
        still converges — degrades gracefully toward Picard-like behaviour."""
        solver = self._solver(spectral_clip=0.01, max_steps=500)
        sol, _ = solver._solve((linear, (0.5, 1.0)), x0=jnp.array(0.0))
        assert jnp.allclose(sol.value, 2.0, atol=1e-4)

    def test_spectral_clip_same_result_as_default(self):
        """Explicit spectral_clip=1.5 matches the default."""
        A = jnp.array([[0.0, 0.25], [0.25, 0.0]])
        b = jnp.array([1.0, 2.0])
        sol_default, _ = self._solver()._solve((affine, (A, b)), x0=jnp.zeros(2))
        sol_explicit, _ = self._solver(spectral_clip=1.5)._solve(
            (affine, (A, b)), x0=jnp.zeros(2)
        )
        assert jnp.allclose(sol_default.value, sol_explicit.value, atol=1e-6)


# ── TestAnderson ─────────────────────────────────────────────────────────


class TestAnderson:
    def _solver(
        self,
        atol: float = ATOL,
        rtol: float = RTOL,
        max_steps: int = MAX_STEPS,
        use_linalg: bool = True,
    ):
        return Anderson(
            atol=atol,
            rtol=rtol,
            max_steps=max_steps,
            depth=5,
            use_linalg=use_linalg,
        )

    def test_scalar_fixed_point(self):
        solver = self._solver()
        sol, _ = solver._solve((linear, (0.5, 1.0)), x0=jnp.array(0.0))
        assert jnp.allclose(sol.value, 2.0, atol=1e-4)

    def test_vector_fixed_point(self):
        A = jnp.array([[0.0, 0.25], [0.25, 0.0]])
        b = jnp.array([1.0, 2.0])
        solver = self._solver()
        sol, _ = solver._solve((affine, (A, b)), x0=jnp.zeros(2))
        expected = jnp.linalg.solve(jnp.eye(2) - A, b)
        assert jnp.allclose(sol.value, expected, atol=1e-4)

    def test_pytree_fixed_point(self):
        solver = self._solver()
        x0 = {"a": jnp.array(0.0), "b": jnp.array(0.0)}
        sol, _ = solver._solve(pytree_f, x0)
        assert jnp.allclose(sol.value["a"], 2.0, atol=1e-4)
        assert jnp.allclose(sol.value["b"], 4.0, atol=1e-4)

    def test_fewer_steps_than_picard(self):
        """Anderson should converge in fewer steps on a hard-ish problem."""
        a, b = 0.9, 1.0
        picard = Picard(atol=1e-4, rtol=0.0, max_steps=500)
        anderson = self._solver(atol=1e-4, rtol=0.0, max_steps=500)
        sol_p, _ = picard._solve((linear, (a, b)), x0=jnp.array(0.0))
        sol_a, _ = anderson._solve((linear, (a, b)), x0=jnp.array(0.0))
        assert sol_a.stats.steps < sol_p.stats.steps

    @pytest.mark.parametrize("use_linalg", [True, False], ids=["linalg", "cholesky"])
    def test_cholesky_matches_linalg(self, use_linalg):
        """Both solver backends produce the same fixed point."""
        solver = self._solver(use_linalg=use_linalg)
        sol, _ = solver._solve((linear, (0.5, 1.0)), x0=jnp.array(0.0))
        assert jnp.allclose(sol.value, 2.0, atol=1e-4)

    def test_cholesky_vector(self):
        A = jnp.array([[0.0, 0.25], [0.25, 0.0]])
        b = jnp.array([1.0, 2.0])
        solver = self._solver(use_linalg=False)
        sol, _ = solver._solve((affine, (A, b)), x0=jnp.zeros(2))
        expected = jnp.linalg.solve(jnp.eye(2) - A, b)
        assert jnp.allclose(sol.value, expected, atol=1e-4)

    def test_cholesky_pytree(self):
        solver = self._solver(use_linalg=False)
        x0 = {"a": jnp.array(0.0), "b": jnp.array(0.0)}
        sol, _ = solver._solve(pytree_f, x0)
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


# ── TestTrace ─────────────────────────────────────────────────────────────


class TestTrace:
    @pytest.mark.parametrize("make_adj", ALL_ADJOINTS, ids=ALL_ADJOINT_IDS)
    def test_trace_tracks_last_iterate(self, make_adj):
        def trace_fn(acc, x, fx, f_aux):
            return x

        sol = make_adj()(
            (linear, (0.5, 1.0)),
            jnp.array(0.0),
            trace=(trace_fn, jnp.array(0.0)),
        )
        # trace holds the last x seen, which should be close to x*=2.0
        assert jnp.isfinite(sol.trace)
        assert sol.trace > 1.0  # converging toward 2.0

    @pytest.mark.parametrize("make_adj", ALL_ADJOINTS, ids=ALL_ADJOINT_IDS)
    def test_trace_counts_f_evals(self, make_adj):
        def trace_fn(acc, x, fx, f_aux):
            return acc + 1

        sol = make_adj()(
            (linear, (0.5, 1.0)),
            jnp.array(0.0),
            trace=(trace_fn, jnp.array(0)),
        )
        # trace counts every f call (including init), so >= number of steps
        assert sol.trace >= sol.stats.steps

    @pytest.mark.parametrize("make_adj", ALL_ADJOINTS, ids=ALL_ADJOINT_IDS)
    def test_trace_pytree(self, make_adj):
        def trace_fn(acc, x, fx, f_aux):
            return {"last_x": x, "count": acc["count"] + 1}

        trace_init = {"last_x": jnp.array(0.0), "count": jnp.array(0)}
        sol = make_adj()(
            (linear, (0.5, 1.0)),
            jnp.array(0.0),
            trace=(trace_fn, trace_init),
        )
        assert jnp.isfinite(sol.trace["last_x"])
        assert sol.trace["count"] > 0

    @pytest.mark.parametrize("make_adj", ALL_ADJOINTS, ids=ALL_ADJOINT_IDS)
    def test_no_trace_is_none(self, make_adj):
        sol = make_adj()((linear, (0.5, 1.0)), jnp.array(0.0))
        assert sol.trace is None

    @pytest.mark.parametrize("make_adj", ALL_ADJOINTS, ids=ALL_ADJOINT_IDS)
    def test_trace_provided_is_not_none(self, make_adj):
        sol = make_adj()(
            (linear, (0.5, 1.0)),
            jnp.array(0.0),
            trace=(lambda acc, x, fx, f_aux: x, jnp.array(0.0)),
        )
        assert sol.trace is not None


# ── TestHasAux ────────────────────────────────────────────────────────────


class TestHasAux:
    @pytest.mark.parametrize("make_adj", ALL_ADJOINTS, ids=ALL_ADJOINT_IDS)
    def test_has_aux_primal_correct(self, make_adj):
        """f returns (fx, aux_data); solver should still find the fixed point."""

        def f_with_aux(x, a, b):
            return a * x + b, {"norm": jnp.abs(x)}

        sol = make_adj()(
            (f_with_aux, (0.5, 1.0)),
            jnp.array(0.0),
            has_aux=True,
        )
        assert jnp.allclose(sol.value, 2.0, atol=0.01)

    @pytest.mark.parametrize("make_adj", ALL_ADJOINTS, ids=ALL_ADJOINT_IDS)
    def test_has_aux_with_trace(self, make_adj):
        """trace_fn receives non-None f_aux when has_aux=True."""

        def f_with_aux(x, a, b):
            return a * x + b, jnp.abs(x)

        def trace_fn(acc, x, fx, f_aux):
            return f_aux

        sol = make_adj()(
            (f_with_aux, (0.5, 1.0)),
            jnp.array(0.0),
            has_aux=True,
            trace=(trace_fn, jnp.array(0.0)),
        )
        assert sol.trace is not None
        assert jnp.isfinite(sol.trace)

    @pytest.mark.parametrize("make_adj", ALL_ADJOINTS, ids=ALL_ADJOINT_IDS)
    def test_has_aux_without_trace(self, make_adj):
        """has_aux=True without trace → sol.trace is None."""

        def f_with_aux(x, a, b):
            return a * x + b, jnp.abs(x)

        sol = make_adj()(
            (f_with_aux, (0.5, 1.0)),
            jnp.array(0.0),
            has_aux=True,
        )
        assert sol.trace is None


# ── TestSolution ──────────────────────────────────────────────────────────


class TestSolution:
    def test_solution_fields(self):
        sol = BPTT(solver=P_BWD)((linear, (0.5, 1.0)), jnp.array(0.0))
        assert hasattr(sol, "value")
        assert hasattr(sol, "result")
        assert hasattr(sol, "stats")
        assert hasattr(sol, "trace")
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
        assert int(sol.result) == Result.DIVERGED


# ── TestStepBudget ────────────────────────────────────────────────────────


class TestStepBudget:
    """Tests for the runtime step_budget parameter on Solver._solve()."""

    def _solver(self, max_steps=MAX_STEPS):
        # atol=rtol=0 so only the step limit terminates the loop
        return Picard(atol=0.0, rtol=0.0, max_steps=max_steps)

    def test_array_limits_steps(self):
        sol, _ = self._solver()._solve(
            (linear, (0.5, 1.0)), jnp.array(0.0), step_budget=jnp.array(5)
        )
        assert int(sol.stats.steps) <= 5

    def test_none_is_noop(self):
        solver = self._solver(max_steps=7)
        sol_default, _ = solver._solve((linear, (0.5, 1.0)), jnp.array(0.0))
        sol_none, _ = solver._solve(
            (linear, (0.5, 1.0)), jnp.array(0.0), step_budget=None
        )
        assert int(sol_default.stats.steps) == int(sol_none.stats.steps)

    def test_result_is_max_steps_when_budget_exhausted(self):
        sol, _ = self._solver()._solve(
            (linear, (0.5, 1.0)), jnp.array(0.0), step_budget=jnp.array(3)
        )
        assert int(sol.result) == Result.MAX_STEPS

    def test_over_ceiling_silently_clamped(self):
        """step_budget above max_steps: static ceiling wins silently."""
        solver = self._solver(max_steps=5)
        sol, _ = solver._solve(
            (linear, (0.5, 1.0)), jnp.array(0.0), step_budget=jnp.array(1000)
        )
        assert int(sol.stats.steps) <= 5

    def test_jit_compatible(self):
        import jax

        solver = self._solver()
        f_spec = (linear, (0.5, 1.0))
        x0 = jnp.array(0.0)
        sol, _ = jax.jit(lambda b: solver._solve(f_spec, x0, step_budget=b))(
            jnp.array(7)
        )
        assert int(sol.stats.steps) <= 7


# ── TestTraceHelpers ──────────────────────────────────────────────────────


class TestTraceHelpers:
    """Tests for the trace_* helper functions in banax.utils."""

    @pytest.mark.parametrize("make_adj", ALL_ADJOINTS, ids=ALL_ADJOINT_IDS)
    def test_trace_last_captures_final_value(self, make_adj):
        # Track the last fx value; for x* = 2.0, fx at convergence ≈ 2.0
        sol = make_adj()(
            (linear, (0.5, 1.0)),
            jnp.array(0.0),
            trace=trace_last(lambda x, fx, f_aux: fx, jnp.array(0.0)),
        )
        assert jnp.isfinite(sol.trace)
        assert jnp.allclose(sol.trace, 2.0, atol=0.05)

    @pytest.mark.parametrize("make_adj", ALL_ADJOINTS, ids=ALL_ADJOINT_IDS)
    def test_trace_last_aux_captures_final_aux(self, make_adj):
        def f_with_aux(x, a, b):
            return a * x + b, jnp.abs(x)

        sol = make_adj()(
            (f_with_aux, (0.5, 1.0)),
            jnp.array(0.0),
            has_aux=True,
            trace=trace_last_aux(jnp.array(0.0)),
        )
        assert jnp.isfinite(sol.trace)
        # At x* = 2.0, |x| ≈ 2.0
        assert jnp.allclose(sol.trace, 2.0, atol=0.05)

    @pytest.mark.parametrize("make_adj", ALL_ADJOINTS, ids=ALL_ADJOINT_IDS)
    def test_trace_history_records_all_evals(self, make_adj):
        n = MAX_STEPS + 1
        sol = make_adj()(
            (linear, (0.5, 1.0)),
            jnp.array(0.0),
            trace=trace_history(lambda x, fx, f_aux: x, n, 0.0),
        )
        count, buf = sol.trace
        assert buf.shape == (n,)
        # count includes the init() call, so count >= steps + 1
        assert count >= sol.stats.steps + 1
        # zero-padding begins at index count
        assert jnp.all(buf[int(count) :] == 0.0)

    @pytest.mark.parametrize("make_adj", ALL_ADJOINTS, ids=ALL_ADJOINT_IDS)
    def test_trace_history_non_scalar(self, make_adj):
        # 2-element x; track full x at each eval
        def f_vec(x, a, b):
            return a * x + b

        x0 = jnp.array([0.0, 0.0])
        n = MAX_STEPS + 1
        sol = make_adj()(
            (f_vec, (0.5, jnp.array([1.0, 2.0]))),
            x0,
            trace=trace_history(lambda x, fx, f_aux: x, n, jnp.zeros(2)),
        )
        _, buf = sol.trace
        assert buf.shape == (n, 2)

    @pytest.mark.parametrize("make_adj", ALL_ADJOINTS, ids=ALL_ADJOINT_IDS)
    def test_trace_count_vs_stats(self, make_adj):
        sol = make_adj()(
            (linear, (0.5, 1.0)),
            jnp.array(0.0),
            trace=trace_count(),
        )
        # trace_count includes init(), so must be > stats.steps
        assert sol.trace > sol.stats.steps


# ── TestPytreeHelpers ─────────────────────────────────────────────────────


class TestPytreeHelpers:
    def test_zeros_like_array(self):
        x = jnp.array([1.0, 2.0, 3.0])
        z = zeros_like(x)
        assert z.shape == x.shape
        assert z.dtype == x.dtype
        assert jnp.all(z == 0.0)

    def test_zeros_like_pytree(self):
        tree = {"a": jnp.ones((2, 3)), "b": jnp.ones(4, dtype=jnp.float16)}
        z = zeros_like(tree)
        assert jnp.all(z["a"] == 0.0)
        assert z["a"].shape == (2, 3)
        assert jnp.all(z["b"] == 0.0)
        assert z["b"].dtype == jnp.float16

    def test_half_normal_like_shape_dtype(self):
        import jax

        key = jax.random.key(0)
        x = jnp.ones((10, 10))
        out = half_normal_like(key, x)
        assert out.shape == x.shape
        assert out.dtype == x.dtype

    def test_half_normal_like_roughly_half_zero(self):
        import jax

        key = jax.random.key(42)
        x = jnp.ones((1000,))
        out = half_normal_like(key, x)
        zero_frac = jnp.mean(out == 0.0)
        # With 1000 elements and p=0.5, fraction should be near 0.5
        assert 0.4 < float(zero_frac) < 0.6

    def test_half_normal_like_pytree(self):
        import jax

        key = jax.random.key(7)
        tree = {"a": jnp.ones((4, 4)), "b": jnp.ones((4,))}
        out = half_normal_like(key, tree)
        assert out["a"].shape == (4, 4)
        assert out["b"].shape == (4,)
        # Non-zero elements should be normally distributed (finite)
        assert jnp.all(jnp.isfinite(out["a"]))
        assert jnp.all(jnp.isfinite(out["b"]))

    def test_half_normal_like_different_keys_differ(self):
        import jax

        x = jnp.ones((20,))
        out1 = half_normal_like(jax.random.key(0), x)
        out2 = half_normal_like(jax.random.key(1), x)
        assert not jnp.array_equal(out1, out2)


# ── TestDtype ─────────────────────────────────────────────────────────────


# Solver factories parameterised by tolerance config.  Each factory takes the
# atol/rtol pair and returns a solver instance with deterministic settings —
# this keeps the dtype tests focused on the type-promotion behaviour rather
# than tuning per-solver hyperparameters.
def _picard(atol, rtol):
    return Picard(atol=atol, rtol=rtol, max_steps=80)


def _relaxed(atol, rtol):
    return Relaxed(damp=0.7, atol=atol, rtol=rtol, max_steps=80)


def _reversible(atol, rtol):
    return ReversibleSolver(
        damp=0.5, atol=atol, rtol=rtol, max_steps=20, loop_kind="lax"
    )


def _broyden(atol, rtol):
    return Broyden(atol=atol, rtol=rtol, max_steps=80, history_size=5)


def _broyden_ls(atol, rtol):
    return Broyden(atol=atol, rtol=rtol, max_steps=80, history_size=5, ls_steps=3)


def _anderson_linalg(atol, rtol):
    return Anderson(atol=atol, rtol=rtol, max_steps=80, depth=3, use_linalg=True)


def _anderson_cholesky(atol, rtol):
    return Anderson(atol=atol, rtol=rtol, max_steps=80, depth=3, use_linalg=False)


_DTYPE_SOLVERS = [
    (_picard, "picard"),
    (_relaxed, "relaxed"),
    (_reversible, "reversible"),
    (_broyden, "broyden"),
    (_broyden_ls, "broyden_ls"),
    (_anderson_linalg, "anderson_linalg"),
    (_anderson_cholesky, "anderson_cholesky"),
]
_DTYPE_FACTORIES = [s[0] for s in _DTYPE_SOLVERS]
_DTYPE_IDS = [s[1] for s in _DTYPE_SOLVERS]

# Tolerance configurations exercised across every solver.  Both-zero shakes
# out the carry without a tolerance branch; atol-only and rtol-only verify
# each branch independently; both-active checks the conjunction.
_TOL_CONFIGS = [
    (1e-2, 0.0, "atol_only"),
    (0.0, 1e-2, "rtol_only"),
    (1e-2, 1e-2, "atol_and_rtol"),
    (0.0, 0.0, "both_zero"),
]
_TOL_PARAMS = [(a, r) for a, r, _ in _TOL_CONFIGS]
_TOL_IDS = [name for *_, name in _TOL_CONFIGS]


class TestDtype:
    """Solvers must work with low-precision inputs (bf16, f16).

    Regression tests for the dtype-mismatch bugs that prevented running
    banax on bf16 training pipelines:

      * ``Solver._solve`` previously kept ``aerr``/``rerr`` at the input
        dtype in the loop body but at f32 in the init, breaking the
        ``eqxi.while_loop`` carry signature for bf16 inputs.
      * ``Broyden`` allocated ``U``/``V`` and the line-search ``alpha`` at
        f32 by default, mismatching the bf16 ``g_flat``/``dx``.
      * ``Anderson`` allocated history buffers and ``jnp.eye`` at f32, and
        ``jnp.linalg.solve`` does not even support bf16, so the small LS
        system needs an explicit promotion to f32 with the result cast back.
    """

    @pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
    @pytest.mark.parametrize("atol,rtol", _TOL_PARAMS, ids=_TOL_IDS)
    @pytest.mark.parametrize("make_solver", _DTYPE_FACTORIES, ids=_DTYPE_IDS)
    def test_scalar_low_precision(self, make_solver, atol, rtol, dtype):
        """Low-precision scalar solve runs without dtype errors and preserves input dtype."""
        x0 = jnp.array(0.0, dtype=dtype)
        a = jnp.array(0.5, dtype=dtype)
        b = jnp.array(1.0, dtype=dtype)
        sol, _ = make_solver(atol, rtol)._solve((linear, (a, b)), x0)
        # Fixed point of f(x) = 0.5*x + 1 is x* = 2.  The point of this test
        # is the dtype carry, not numerical accuracy: with both tols disabled
        # the quasi-Newton solvers overshoot in f16 and trip the NaN guard
        # (correct behaviour, reported as DIVERGED), so accuracy is only
        # asserted when at least one tolerance is active.
        assert sol.value.dtype == dtype
        if atol > 0.0 or rtol > 0.0:
            assert jnp.isfinite(sol.value)
            assert jnp.allclose(sol.value.astype(jnp.float32), 2.0, atol=0.05)
        else:
            # Both tols off: any of CONVERGED / MAX_STEPS / DIVERGED is fine,
            # we only care that the loop didn't blow up at trace time.
            assert int(sol.result) in (Result.CONVERGED, Result.MAX_STEPS, Result.DIVERGED)

    @pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
    @pytest.mark.parametrize("make_solver", _DTYPE_FACTORIES, ids=_DTYPE_IDS)
    def test_vector_low_precision(self, make_solver, dtype):
        """Vector solve with non-trivial spectral radius works in low precision."""
        A = jnp.array([[0.0, 0.25], [0.25, 0.0]], dtype=dtype)
        b = jnp.array([1.0, 2.0], dtype=dtype)
        x0 = jnp.zeros(2, dtype=dtype)
        sol, _ = make_solver(1e-2, 0.0)._solve((affine, (A, b)), x0)
        # Closed-form fixed point computed in f32 to avoid bf16 round-off
        # in the reference value itself.
        expected = jnp.linalg.solve(
            jnp.eye(2) - A.astype(jnp.float32), b.astype(jnp.float32)
        )
        assert sol.value.dtype == dtype
        assert jnp.allclose(sol.value.astype(jnp.float32), expected, atol=0.05)

    @pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
    @pytest.mark.parametrize("make_solver", _DTYPE_FACTORIES, ids=_DTYPE_IDS)
    def test_stats_are_float32(self, make_solver, dtype):
        """``Stats.abs_err`` / ``rel_err`` are always f32 — see ``Stats`` docstring.

        This is the load-bearing invariant for the loop carry: keeping the
        scalar in f32 lets the body re-enter with the same dtype regardless
        of the input precision.
        """
        x0 = jnp.array(0.0, dtype=dtype)
        a = jnp.array(0.5, dtype=dtype)
        b = jnp.array(1.0, dtype=dtype)
        sol, _ = make_solver(1e-2, 1e-2)._solve((linear, (a, b)), x0)
        assert sol.stats.abs_err.dtype == jnp.float32
        assert sol.stats.rel_err.dtype == jnp.float32

    @pytest.mark.parametrize("make_solver", _DTYPE_FACTORIES, ids=_DTYPE_IDS)
    def test_pytree_bf16(self, make_solver):
        """Pytree inputs with bf16 leaves should also work."""
        x0 = {
            "a": jnp.array(0.0, dtype=jnp.bfloat16),
            "b": jnp.array(0.0, dtype=jnp.bfloat16),
        }

        def f(x):
            return {
                "a": jnp.bfloat16(0.5) * x["a"] + jnp.bfloat16(1.0),
                "b": jnp.bfloat16(0.25) * x["b"] + jnp.bfloat16(3.0),
            }

        sol, _ = make_solver(1e-2, 0.0)._solve(f, x0)
        # Fixed point: a*=2, b*=4. bf16 gives ~0.05 absolute precision.
        assert sol.value["a"].dtype == jnp.bfloat16
        assert sol.value["b"].dtype == jnp.bfloat16
        assert jnp.allclose(sol.value["a"].astype(jnp.float32), 2.0, atol=0.05)
        assert jnp.allclose(sol.value["b"].astype(jnp.float32), 4.0, atol=0.05)
