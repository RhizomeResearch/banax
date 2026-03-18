from abc import abstractmethod

import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.internal as eqxi
from equinox.internal import ω

from typing import Callable
from jaxtyping import Array, Bool, PyTree

from banax._core import (
    T,
    Fn,
    Error,
    Step,
    EquinoxLoopKind,
    FSpec,
    Result,
    Stats,
    Solution,
    _normalize_f_spec,
)


def _sq_norm(x: T) -> Error:
    return jax.tree.reduce_associative(
        jnp.add, jax.tree.map(lambda _x: jnp.sum(_x**2), x)
    )


def _abs_err(x: T, fx: T) -> Error:
    return jnp.sqrt(_sq_norm((fx**ω - x**ω).ω))


def _rel_err(x: T, fx: T, eps: float = 1e-8) -> Error:
    return jnp.sqrt(_sq_norm((fx**ω - x**ω).ω)) / (jnp.sqrt(_sq_norm(fx)) + eps)


def _tree_allfinite(tree: T) -> Bool[Array, ""]:
    bools = jax.tree.map(lambda x: jnp.all(jnp.isfinite(x)), tree)
    return jax.tree.reduce_associative(jnp.logical_and, bools)


class Solver[Z, S](eqx.Module):
    """Abstract base class for fixed-point solvers.

    All solvers share the following keyword arguments:

    Args:
        atol: Absolute tolerance.
            Iteration stops when ``‖f(x) − x‖ ≤ atol``.
            Set to ``0.0`` (default) to disable.
        rtol: Relative tolerance.
            Iteration stops when ``‖f(x) − x‖ / ‖f(x)‖ ≤ rtol``.
            Default ``1e-5``.
            Set to ``0.0`` to disable.
        max_steps: Hard cap on the number of iterations.
            The solver always stops after at most ``max_steps`` applications of ``f``.
        loop_kind: Controls how the loop is compiled.
            ``"lax"`` (default) uses ``jax.lax.while_loop`` —
            minimal memory, not differentiable through the loop.
            ``"bounded"`` and ``"checkpointed"`` are differentiable
            and required for :class:`~banax.adjoint.BPTT`.

    Convergence criterion: the loop terminates
        when either active tolerance is satisfied,
        or when ``x`` becomes non-finite (NaN/Inf guard).
    """

    atol: float = eqx.field(static=True, default=0.0, kw_only=True)
    rtol: float = eqx.field(static=True, default=1e-5, kw_only=True)
    max_steps: int = eqx.field(static=True, default=50, kw_only=True)
    loop_kind: EquinoxLoopKind = eqx.field(static=True, default="lax", kw_only=True)

    @abstractmethod
    def init(self, f: Fn[Z], x0: Z) -> S: ...

    @abstractmethod
    def step(self, f: Fn[Z], x: Z, state: S) -> tuple[Z, Z, S]: ...

    def _solve(
        self,
        f_spec: FSpec[Z],
        x0: Z,
        *,
        aux_update: Callable[..., PyTree] | None = None,
        aux_init: PyTree | None = None,
    ) -> tuple[Solution, S]:
        type Carry = tuple[tuple[Z, Step, Error, Error, PyTree | None], S]

        f, f_args, f_kwargs = _normalize_f_spec(f_spec)

        def _f(_x):
            return f(_x, *f_args, **f_kwargs)

        def _loop_cond(_carry: Carry) -> Bool[Array, ""]:
            (x, _, aerr, rerr, _), _ = _carry
            acont = aerr > self.atol if self.atol > 0.0 else jnp.array(True)
            rcont = rerr > self.rtol if self.rtol > 0.0 else jnp.array(True)
            return jnp.logical_and(jnp.logical_and(acont, rcont), _tree_allfinite(x))

        def _loop_body(_carry: Carry) -> Carry:
            (x, step, aerr, rerr, aux), solver_state = _carry
            x_next, fx_next, solver_state_next = self.step(_f, x, solver_state)
            aux_next = (
                aux_update(step, aux, x_next, fx_next, solver_state_next)
                if aux_update is not None
                else None
            )

            aerr = _abs_err(x_next, fx_next) if self.atol > 0.0 else jnp.array(0.0)
            rerr = _rel_err(x_next, fx_next) if self.rtol > 0.0 else jnp.array(0.0)
            step += 1

            return (x_next, step, aerr, rerr, aux_next), solver_state_next

        solver_state_init = self.init(_f, x0)
        step = jnp.array(0)
        aerr = jnp.array(2 * self.atol)
        rerr = jnp.array(2 * self.rtol)

        carry = ((x0, step, aerr, rerr, aux_init), solver_state_init)

        carry = eqxi.while_loop(
            _loop_cond, _loop_body, carry, max_steps=self.max_steps, kind=self.loop_kind
        )

        (x, step, aerr, rerr, aux), solver_state = carry
        converged = jnp.logical_not(_loop_cond(carry))
        result_code = jnp.where(converged, Result.CONVERGED, Result.MAX_STEPS)
        stats = Stats(steps=step, abs_err=aerr, rel_err=rerr)
        solution = Solution(value=x, result=result_code, stats=stats, aux=aux)

        return solution, solver_state


class Picard[Z: T](Solver[Z, Z]):
    """Picard (simple) fixed-point iteration.

    Applies the update ``x_{n+1} = f(x_n)`` at each step.
    Convergence is guaranteed by the Banach fixed-point theorem
        when ``f`` is contractive (Lipschitz constant < 1).
    """

    type State = Z

    def init(self, f: Fn[Z], x0: Z) -> State:
        return f(x0)

    def step(self, f: Fn[Z], x: Z, state: State) -> tuple[Z, Z, State]:
        x_next = state
        fx_next = f(x_next)
        return x_next, fx_next, fx_next


class Relaxed[Z: T](Solver[Z, Z]):
    """Damped (under-relaxed) fixed-point iteration.

    Applies the update ``x_{n+1} = (1 − β) x_n + β f(x_n)`` at each step,
        where β = ``damp`` ∈ (0, 1].

    When ``damp=1.0`` this reduces to :class:`Picard`.
    For β ∈ (0, 1), the effective spectral radius of the iteration operator
        is shrunk by a factor of β,
        which can stabilise convergence
        when the Jacobian of ``f`` at the fixed point
        has spectral radius close to (but still below) 1.
    In practice β ∈ (0.5, 1) gives a good speed/stability tradeoff.

    Args:
        damp: Damping factor β ∈ (0, 1]. Default ``0.8``.
    """

    type State = Z
    damp: float = eqx.field(static=True, default=0.8, kw_only=True)

    def init(self, f: Fn[Z], x0: Z) -> State:
        return f(x0)

    def step(self, f: Fn[Z], x: Z, state: State) -> tuple[Z, Z, State]:
        f_x = state
        x_next = ((1 - self.damp) * x**ω + self.damp * f_x**ω).ω
        fx_next = f(x_next)
        return x_next, fx_next, fx_next


class Reversible[Z: T](Solver[Z, tuple[Z, Z]]):
    """Reversible DEQ solver.

    From McCallum et al. (2025):
        *Reversible Deep Equilibrium Models*
        doi:10.48550/arXiv.2509.12917

    Find the fixed point of ``z ↦ f(z)``
        through the paired-iterate update::

        y_{n+1} ← (1 − β) y_n + β f(z_n)
        z_{n+1} ← (1 − β) z_n + β f(y_{n+1}).

    If ½ ≤ β < 1 and ``f`` has a contractive Lipschitz constant,
        then it has a unique fixed point z*, z* = f(z*),
        and lim_n y_n = lim_n z_n = z*.

    The update rule is invertible::

        z_n = (z_{n+1} − β f(y_{n+1})) / (1 − β)
        y_n = (y_{n+1} − β f(z_n))     / (1 − β).

    During the backward pass,
        this allows the :class:`~banax.adjoint.Reversible` adjoint
        to reconstruct the iteration trajectory from (y*, z*)
        without storing intermediate iterates,
        achieving O(1) memory in ``max_steps``.

    .. note::
        This solver is designed to be used
            with the :class:`~banax.adjoint.Reversible` adjoint.
        The backward reconstruction amplifies errors by 1/(1−β) per step;
            for numerical stability, the damping factor β
            should be kept well below 1 (in practice, β ∈ [0.5, 0.9)).

    Args:
        damp: Damping factor β ∈ (0, 1). Default ``0.8``.
            Must be strictly less than 1
            (the backward pass divides by 1−β).
    """

    type State = tuple[Z, Z]
    damp: float = eqx.field(static=True, default=0.8, kw_only=True)

    def init(self, f: Fn[Z], x0: Z) -> State:
        return x0, f(x0)

    def step(self, f: Fn[Z], x: Z, state: State) -> tuple[Z, Z, State]:
        w, f_x = state

        def _body(_f_carry, _a):
            b = ((1 - self.damp) * _a**ω + self.damp * _f_carry**ω).ω
            return f(b), b

        wx = jax.tree.map(lambda a, b: jnp.stack([a, b]), w, x)
        fx_next, wx_next = jax.lax.scan(_body, f_x, wx)
        w_next = (wx_next**ω)[0].ω
        x_next = (wx_next**ω)[1].ω

        return x_next, fx_next, (w_next, fx_next)
