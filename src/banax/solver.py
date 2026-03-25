from abc import abstractmethod
from functools import partial

import jax
import jax.flatten_util
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
        step_budget: Step | None = None,
    ) -> tuple[Solution, S]:
        """Run the fixed-point loop.

        Args:
            step_budget: Optional runtime cap on the number of iterations.
                Pass a JAX array (e.g. ``jnp.array(n)``) as a JIT argument
                so it is traced as an abstract value — varying it between calls
                does not trigger recompilation.  Must be ``<= max_steps``;
                values above are silently clamped by the static ceiling.
                ``None`` (default) leaves the behaviour unchanged.
        """
        type Carry = tuple[tuple[Z, Step, Error, Error, PyTree | None], S]

        f, f_args, f_kwargs = _normalize_f_spec(f_spec)

        def _f(_x):
            return f(_x, *f_args, **f_kwargs)

        def _above_tols(_aerr, _rerr) -> Bool[Array, ""] | bool:
            """True when all active tolerance criteria are unsatisfied."""
            if self.atol > 0.0 and self.rtol > 0.0:
                return jnp.logical_and(_aerr > self.atol, _rerr > self.rtol)
            elif self.atol > 0.0:
                return _aerr > self.atol
            elif self.rtol > 0.0:
                return _rerr > self.rtol
            else:
                return True

        def _loop_cond(_carry: Carry) -> Bool[Array, ""]:
            (x, step, aerr, rerr, _), _ = _carry
            running = jnp.logical_and(_above_tols(aerr, rerr), _tree_allfinite(x))
            if step_budget is None:
                return running
            return jnp.logical_and(running, step < step_budget)

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
        result_code = jnp.where(
            jnp.logical_not(_tree_allfinite(x)),
            Result.DIVERGED,
            jnp.where(_above_tols(aerr, rerr), Result.MAX_STEPS, Result.CONVERGED),
        )
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


class Broyden[Z: T](Solver[Z, tuple]):
    """Limited-memory Broyden's method for fixed-point iteration.

    Maintains a low-rank approximation to the inverse Jacobian of g(x) = f(x) - x.
    Converges faster than Picard on many problems,
    especially when the Jacobian spectral radius is close to 1.

    Args:
        history_size: Number of rank-1 updates to retain. Default ``10``.
        ls_steps: Maximum number of Armijo backtracking halvings per step.
            ``0`` (default) disables line search entirely.
            When enabled, the step size ``alpha`` is halved
            until ``‖g(x + alpha·dx)‖² ≤ (1 − 2e−4·alpha)·‖g(x)‖²``
            or ``ls_steps`` halvings are exhausted.

            .. note::
                Line search uses an inner ``jax.lax.while_loop``,
                which is not differentiable.
                ``ls_steps > 0`` is therefore incompatible with
                :class:`~banax.adjoint.BPTT` and
                :class:`~banax.adjoint.Reversible` adjoints.
    """

    type State = tuple
    history_size: int = eqx.field(static=True, default=10, kw_only=True)
    ls_steps: int = eqx.field(static=True, default=0, kw_only=True)

    def init(self, f: Fn[Z], x0: Z) -> State:
        fx0 = f(x0)
        x_flat, _ = jax.flatten_util.ravel_pytree(x0)
        n = x_flat.shape[0]
        fx_flat, _ = jax.flatten_util.ravel_pytree(fx0)
        g_flat = fx_flat - x_flat
        U = jnp.zeros((self.history_size, n))
        V = jnp.zeros((self.history_size, n))
        return fx0, g_flat, U, V, jnp.array(0)

    def step(self, f: Fn[Z], x: Z, state: State) -> tuple[Z, Z, State]:
        fx_prev, g_flat, U, V, idx = state

        x_flat, unflatten = jax.flatten_util.ravel_pytree(x)

        # Broyden direction: dx = -J^{-1} g = g - U^T (V g)
        Vg = V @ g_flat  # [history_size]
        dx = g_flat - U.T @ Vg  # [n]

        if self.ls_steps > 0:
            # Armijo backtracking: halve alpha until
            # ‖g(x + alpha·dx)‖² ≤ (1 − 2e−4·alpha)·‖g‖²
            g_sq_curr = g_flat @ g_flat

            def _ls_cond(_carry):
                _alpha, _x_next_flat, _fx_next, _g_sq_next = _carry
                return _g_sq_next > (1 - 2e-4 * _alpha) * g_sq_curr

            def _ls_body(_carry):
                _alpha, _, _, _ = _carry
                _alpha_new = _alpha * 0.5
                _x_trial_flat = x_flat + _alpha_new * dx
                _fx_trial = f(unflatten(_x_trial_flat))
                _fx_trial_flat, _ = jax.flatten_util.ravel_pytree(_fx_trial)
                _g_trial_flat = _fx_trial_flat - _x_trial_flat
                return (
                    _alpha_new,
                    _x_trial_flat,
                    _fx_trial,
                    _g_trial_flat @ _g_trial_flat,
                )

            x_next_flat_init = x_flat + dx
            fx_next_init = f(unflatten(x_next_flat_init))
            fx_next_flat_init, _ = jax.flatten_util.ravel_pytree(fx_next_init)
            g_next_flat_init = fx_next_flat_init - x_next_flat_init

            init_carry = (
                jnp.array(1.0),
                x_next_flat_init,
                fx_next_init,
                g_next_flat_init @ g_next_flat_init,
            )
            alpha, x_next_flat, fx_next, _ = eqxi.while_loop(
                _ls_cond, _ls_body, init_carry, max_steps=self.ls_steps, kind="lax"
            )
            dx = alpha * dx
            x_next = unflatten(x_next_flat)
        else:
            x_next_flat = x_flat + dx
            x_next = unflatten(x_next_flat)
            fx_next = f(x_next)

        # Good Broyden (Type-I) rank-1 update
        fx_next_flat, _ = jax.flatten_util.ravel_pytree(fx_next)
        g_next_flat = fx_next_flat - x_next_flat
        dg = g_next_flat - g_flat

        # J^{-1} dg = -dg + U^T (V dg)
        Vdg = V @ dg
        Jinv_dg = U.T @ Vdg - dg

        # v^T = dx^T J^{-1} = -dx^T + (U dx)^T V
        Udx = U @ dx  # [history_size]
        vT = -dx + Udx @ V  # [n]

        numerator = dx - Jinv_dg
        denom = vT @ dg + 1e-12

        # Update U, V ring buffer at position idx % history_size
        slot = idx % self.history_size
        u_new = numerator / denom
        U = U.at[slot].set(u_new)
        V = V.at[slot].set(vT)

        return x_next, fx_next, (fx_next, g_next_flat, U, V, idx + 1)


def _cholesky_solve(A, b, n):
    """Solve A x = b for small SPD A without ``jnp.linalg``.

    Uses ``jax.lax.fori_loop`` so the compiled graph is O(1) in ``n``
    and contains no linalg kernels (ONNX-exportable).
    """

    # Cholesky factorization: A = L L^T
    def _chol_inner(_i, _j, _L):
        mask = (jnp.arange(n) < _j).astype(A.dtype)
        s = _L[_i] * mask @ _L[_j]
        return _L.at[_i, _j].set((A[_i, _j] - s) / _L[_j, _j])

    def _chol_outer(_i, _L):
        _L = jax.lax.fori_loop(0, _i, partial(_chol_inner, _i), _L)
        mask = (jnp.arange(n) < _i).astype(A.dtype)
        s = _L[_i] * mask @ _L[_i]
        return _L.at[_i, _i].set(jnp.sqrt(jnp.maximum(A[_i, _i] - s, 1e-30)))

    L = jax.lax.fori_loop(0, n, _chol_outer, jnp.zeros_like(A))

    # Forward substitution: L y = b
    def _fwd(_i, _y):
        mask = (jnp.arange(n) < _i).astype(b.dtype)
        s = L[_i] * mask @ _y
        return _y.at[_i].set((b[_i] - s) / L[_i, _i])

    y = jax.lax.fori_loop(0, n, _fwd, jnp.zeros_like(b))

    # Back substitution: L^T x = y
    def _bwd(_ki, _x):
        i = n - 1 - _ki
        mask = (jnp.arange(n) > i).astype(b.dtype)
        s = L[:, i] * mask @ _x
        return _x.at[i].set((y[i] - s) / L[i, i])

    return jax.lax.fori_loop(0, n, _bwd, jnp.zeros_like(b))


class Anderson[Z: T](Solver[Z, tuple]):
    """Anderson acceleration for fixed-point iteration.

    Maintains a history of iterates and residuals,
    solving a small least-squares problem in each step
    to find optimal mixing coefficients.

    Args:
        depth: Number of history entries for mixing. Default ``5``.
        damp: Damping factor beta in (0, 1]. Default ``1.0``.
        ridge: Regularization for the normal equations. Default ``1e-6``.
        use_linalg: If ``True`` (default),
            use ``jnp.linalg.solve`` for the normal equations.
            If ``False``, use a hand-rolled Cholesky decomposition
            that requires no linear algebra backend,
            suitable for embedded or restricted hardware targets.
    """

    type State = tuple
    depth: int = eqx.field(static=True, default=5, kw_only=True)
    damp: float = eqx.field(static=True, default=1.0, kw_only=True)
    ridge: float = eqx.field(static=True, default=1e-6, kw_only=True)
    use_linalg: bool = eqx.field(static=True, default=True, kw_only=True)

    def init(self, f: Fn[Z], x0: Z) -> State:
        fx0 = f(x0)
        x_flat, _ = jax.flatten_util.ravel_pytree(x0)
        n = x_flat.shape[0]
        fx_flat, _ = jax.flatten_util.ravel_pytree(fx0)
        g_flat = fx_flat - x_flat
        X_hist = jnp.zeros((self.depth, n))
        G_hist = jnp.zeros((self.depth, n))
        X_hist = X_hist.at[0].set(x_flat)
        G_hist = G_hist.at[0].set(g_flat)
        return fx0, g_flat, X_hist, G_hist, jnp.array(1)

    def step(self, f: Fn[Z], x: Z, state: State) -> tuple[Z, Z, State]:
        fx_prev, g_flat, X_hist, G_hist, idx = state

        x_flat, unflatten = jax.flatten_util.ravel_pytree(x)
        m = jnp.minimum(idx, self.depth)
        cur = idx % self.depth

        # Build difference matrices via gather + broadcast
        rolled = (cur - 1 - jnp.arange(self.depth)) % self.depth
        DG = g_flat[None, :] - G_hist[rolled]  # [depth, n]
        DX = x_flat[None, :] - X_hist[rolled]  # [depth, n]

        # Mask out invalid entries (beyond current history)
        indices = jnp.arange(self.depth)
        mask = (indices < m).astype(x_flat.dtype)
        DG = DG * mask[:, None]
        DX = DX * mask[:, None]

        # Solve normal equations: (DG^T DG + ridge * I) gamma = DG^T g
        GTG = DG @ DG.T + self.ridge * jnp.eye(self.depth)
        GTg = DG @ g_flat
        if self.use_linalg:
            gamma = jnp.linalg.solve(GTG, GTg)
        else:
            gamma = _cholesky_solve(GTG, GTg, self.depth)
        gamma = gamma * mask

        # Anderson update: x_next = x + beta * g - (DX + beta * DG) @ gamma
        x_next_flat = x_flat + self.damp * g_flat - (DX + self.damp * DG).T @ gamma

        x_next = unflatten(x_next_flat)
        fx_next = f(x_next)

        # Update ring buffer
        fx_next_flat, _ = jax.flatten_util.ravel_pytree(fx_next)
        g_next_flat = fx_next_flat - x_next_flat
        X_hist = X_hist.at[cur].set(x_next_flat)
        G_hist = G_hist.at[cur].set(g_next_flat)

        return x_next, fx_next, (fx_next, g_next_flat, X_hist, G_hist, idx + 1)
