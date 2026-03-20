from abc import abstractmethod

import jax
import jax.flatten_util
import jax.numpy as jnp
import equinox as eqx
import equinox.internal as eqxi
from equinox.internal import ω

from typing import Callable
from jaxtyping import PyTree

from banax._core import T, FSpec, Solution, _normalize_f_spec
from banax.solver import Solver, Relaxed, Broyden, Reversible as ReversibleSolver


def _apply_pullback(f, f_args, f_kwargs, x):
    def _apply(f_spec, _x):
        _f, _fa, _fkw = f_spec
        return _f(_x, *_fa, **_fkw)

    _, pull = eqx.filter_vjp(_apply, (f, f_args, f_kwargs), x)
    return pull


class Adjoint(eqx.Module):
    """Abstract base class for adjoint / differentiation methods.

    An ``Adjoint`` wraps a :class:`~banax.solver.Solver`
        and exposes a single ``__call__`` interface
        that finds the fixed point of ``f``,
        and attaches gradients
        according to the chosen differentiation strategy.

    Calling an adjoint::

        sol = adjoint(f_spec, x0)
        sol = adjoint(f_spec, x0, aux_update=fn, aux_init=init)

    Args:
        f_spec: The fixed-point function;
            see :obj:`~banax._core.FSpec`.
        x0: Initial iterate.
            Must have the same pytree structure and array shapes
            as the output of ``f``.
        aux_update: Optional callable ``(step, aux, x, fx, state) → aux``
            called at each iteration to accumulate auxiliary state.
        aux_init: Initial value for ``aux``;
            required when ``aux_update`` is provided.

    Returns:
        A :class:`~banax._core.Solution`
            whose ``value`` field carries gradients.
    """

    def __call__(
        self,
        f_spec: FSpec[T],
        x0: T,
        *,
        aux_update: Callable[..., PyTree] | None = None,
        aux_init: PyTree | None = None,
    ) -> Solution:
        f, f_args, f_kwargs = _normalize_f_spec(f_spec)
        out_struct = jax.eval_shape(lambda: f(x0, *f_args, **f_kwargs))
        in_struct = jax.eval_shape(lambda: x0)
        if eqx.tree_equal(in_struct, out_struct) is not True:
            raise ValueError("f_spec must produce output with same structure as x0")
        return self._loop(f_spec, x0, aux_update=aux_update, aux_init=aux_init)

    @abstractmethod
    def _loop(
        self,
        f_spec: FSpec[T],
        x0: T,
        *,
        aux_update: Callable[..., PyTree] | None = None,
        aux_init: PyTree | None = None,
    ) -> Solution: ...


class BPTT(Adjoint):
    """Backpropagation through time (BPTT) adjoint.

    Directly differentiates through the unrolled fixed-point loop
        using reverse-mode automatic differentiation.
        Gradients are exact.

    The solver's ``loop_kind`` must be differentiable
        (``"bounded"`` or ``"checkpointed"``, not ``"lax"``).
    For large ``max_steps``,
        ``"checkpointed"`` is strongly recommended:
        it applies gradient checkpointing
        to reduce peak memory from O(max_steps) to O(log max_steps)
        at the cost of additional recomputation during the backward pass.

    Args:
        solver: A :class:`~banax.solver.Solver`
            with ``loop_kind`` in ``{"bounded", "checkpointed"}``.
    """

    solver: Solver

    def __init__(self, solver: Solver):
        if solver.loop_kind not in ["bounded", "checkpointed"]:
            raise TypeError(
                f"BPTT adjoint requires a solver with a differentiable loop_kind"
                f" ('bounded', 'checkpointed'), got {solver.loop_kind!r}"
            )
        self.solver = solver

    def _loop(
        self,
        f_spec: FSpec[T],
        x0: T,
        *,
        aux_update: Callable[..., PyTree] | None = None,
        aux_init: PyTree | None = None,
    ) -> Solution:
        solution, _ = self.solver._solve(
            f_spec, x0, aux_update=aux_update, aux_init=aux_init
        )
        return solution


@eqx.filter_custom_vjp
def _implicit(grad_arg, x0, solver, b_solver, aux_update=None, aux_init=None):
    del b_solver
    solution, _ = solver._solve(
        grad_arg, x0=x0, aux_update=aux_update, aux_init=aux_init
    )
    return solution


@_implicit.def_fwd
def _implicit_fwd(
    perturbed, grad_arg, x0, solver, b_solver, aux_update=None, aux_init=None
):
    del perturbed, b_solver

    solution, _ = solver._solve(
        grad_arg, x0=x0, aux_update=aux_update, aux_init=aux_init
    )

    return solution, solution.value


@_implicit.def_bwd
def _implicit_bwd(
    residuals,
    gradients,
    perturbed,
    grad_arg,
    x0,
    solver,
    b_solver,
    aux_update,
    aux_init,
):
    del perturbed, x0, solver, aux_update, aux_init
    f, f_args, f_kwargs = grad_arg
    x_star = residuals
    grad_x = gradients.value

    pull = _apply_pullback(f, f_args, f_kwargs, x_star)

    # Backward fixed-point map: λ ↦ grad_x + J_f(x*)^T λ
    # Converges to solution of (I - J_f^T) λ = grad_x
    def bwd_f(lam):
        _, lam_x = pull(lam)
        return (grad_x**ω + lam_x**ω).ω

    lam_sol, _ = b_solver._solve(bwd_f, x0=grad_x)
    lam_star = lam_sol.value

    grads, _ = pull(lam_star)
    return grads


class Implicit(Adjoint):
    """Implicit Function Theorem (IFT) adjoint.

    From Bai, S., Kolter, J. Z., & Koltun, V. (2019):
        *Deep Equilibrium Models*
        doi:10.48550/arXiv.1909.01377

    At the fixed point x* = f(x*),
        the IFT gives the gradient of any loss
        L(x*) with respect to the parameters of f as::

        dL/dθ = (∂f/∂θ)^T · λ*,
        where (I − J_f^T) λ* = ∇_{x*} L.

    The linear system ``(I − J_f^T) λ* = ∇_{x*} L``
        is solved with a second fixed-point solver ``b_solver``
        via the iteration ``λ ↦ ∇_{x*} L + J_f(x*)^T λ``.

    Memory cost is O(1) in ``max_steps``
        (only x* is stored, not the trajectory).
    Gradients are exact when ``b_solver`` converges.

    Args:
        solver: Forward fixed-point solver.
        b_solver: Backward solver for the IFT linear system.
            Must converge for the given ``f``
            (e.g., the spectral radius of J_f at x* must be < 1).
    """

    solver: Solver
    b_solver: Solver

    def _loop(
        self,
        f_spec: FSpec[T],
        x0: T,
        *,
        aux_update: Callable[..., PyTree] | None = None,
        aux_init: PyTree | None = None,
    ) -> Solution:
        grad_arg = _normalize_f_spec(f_spec)
        return _implicit(grad_arg, x0, self.solver, self.b_solver, aux_update, aux_init)


@eqx.filter_custom_vjp
def _jfb(grad_arg, x0, solver, aux_update=None, aux_init=None):
    solution, _ = solver._solve(
        grad_arg, x0=x0, aux_update=aux_update, aux_init=aux_init
    )
    return solution


@_jfb.def_fwd
def _jfb_fwd(perturbed, grad_arg, x0, solver, aux_update=None, aux_init=None):
    del perturbed

    solution, _ = solver._solve(
        grad_arg, x0=x0, aux_update=aux_update, aux_init=aux_init
    )

    return solution, solution.value


@_jfb.def_bwd
def _jfb_bwd(
    residuals, gradients, perturbed, grad_arg, x0, solver, aux_update, aux_init
):
    del perturbed, x0, solver, aux_update, aux_init
    f, f_args, f_kwargs = grad_arg
    x_star = residuals
    grad_x = gradients.value

    pull = _apply_pullback(f, f_args, f_kwargs, x_star)
    grads, _ = pull(grad_x)

    return grads


class JFB(Adjoint):
    """Jacobian-Free Backpropagation (JFB) adjoint.

    From Fung et al. (2021):
        *JFB: Jacobian-Free Backpropagation for Implicit Networks*
        doi:10.48550/arXiv.2103.12803

    JFB approximates the IFT gradient
        by replacing the full inverse ``(I − J_f^T)^{-1}`` with the identity,
        giving the biased estimate::

        dL/dθ ≈ (∂f/∂θ)^T · ∇_{x*} L.

    This requires only a single VJP through ``f`` per backward pass,
        making it the cheapest available gradient.
    The bias can be significant
        when the spectral radius of J_f is close to 1.

    Args:
        solver: Forward fixed-point solver.
    """

    solver: Solver

    def _loop(
        self,
        f_spec: FSpec[T],
        x0: T,
        *,
        aux_update: Callable[..., PyTree] | None = None,
        aux_init: PyTree | None = None,
    ) -> Solution:
        grad_arg = _normalize_f_spec(f_spec)
        return _jfb(grad_arg, x0, self.solver, aux_update, aux_init)


@eqx.filter_custom_vjp
def _unroll_phantom(grad_arg, x0, solver, b_solver, aux_update=None, aux_init=None):
    del b_solver
    solution, _ = solver._solve(
        grad_arg, x0=x0, aux_update=aux_update, aux_init=aux_init
    )
    return solution


@_unroll_phantom.def_fwd
def _unroll_phantom_fwd(
    perturbed, grad_arg, x0, solver, b_solver, aux_update=None, aux_init=None
):
    del perturbed, b_solver

    solution, _ = solver._solve(
        grad_arg, x0=x0, aux_update=aux_update, aux_init=aux_init
    )

    return solution, solution.value


@_unroll_phantom.def_bwd
def _unroll_phantom_bwd(
    residuals,
    gradients,
    perturbed,
    grad_arg,
    x0,
    solver,
    b_solver,
    aux_update,
    aux_init,
):
    del perturbed, x0, solver, aux_update, aux_init
    f, f_args, f_kwargs = grad_arg
    x_star = residuals
    grad_x = gradients.value

    def phantom_fn(_f, _f_args, _f_kwargs):
        sol, _ = b_solver._solve(
            (_f, _f_args, _f_kwargs), x0=jax.lax.stop_gradient(x_star)
        )
        return sol.value

    _, pull = eqx.filter_vjp(phantom_fn, f, f_args, f_kwargs)

    return pull(grad_x)


class UnrollPhantom(Adjoint):
    """Unrolling-based Phantom Gradient adjoint.

    From Geng et al. (2022):
        *On Training Implicit Models*
        doi:10.48550/arXiv.2111.05177

    The backward pass re-runs ``b_solver`` starting from ``x*``
        and differentiates through its unrolled trajectory.
    As the number of ``b_solver`` steps increases,
        the gradient converges toward the exact BPTT gradient.
    With a single step, this approximates JFB;
        with many steps it approaches the true gradient
        at the cost of additional VJPs.

    ``b_solver`` must have a differentiable ``loop_kind``
        (``"bounded"`` or ``"checkpointed"``).

    Args:
        solver: Forward fixed-point solver (any ``loop_kind``).
        b_solver: Backward unrolling solver;
            a :class:`~banax.solver.Relaxed` instance
            with ``loop_kind="checkpointed"`` is recommended.
    """

    solver: Solver
    b_solver: Relaxed

    def _loop(
        self,
        f_spec: FSpec[T],
        x0: T,
        *,
        aux_update: Callable[..., PyTree] | None = None,
        aux_init: PyTree | None = None,
    ) -> Solution:
        grad_arg = _normalize_f_spec(f_spec)
        return _unroll_phantom(
            grad_arg, x0, self.solver, self.b_solver, aux_update, aux_init
        )


@eqx.filter_custom_vjp
def _neumann(grad_arg, x0, solver, b_solver, aux_update=None, aux_init=None):
    del b_solver
    solution, _ = solver._solve(
        grad_arg, x0=x0, aux_update=aux_update, aux_init=aux_init
    )
    return solution


@_neumann.def_fwd
def _neumann_fwd(
    perturbed, grad_arg, x0, solver, b_solver, aux_update=None, aux_init=None
):
    del perturbed, b_solver
    solution, _ = solver._solve(
        grad_arg, x0=x0, aux_update=aux_update, aux_init=aux_init
    )

    return solution, (solution.value, solution.stats.steps)


@_neumann.def_bwd
def _neumann_bwd(
    residuals,
    gradients,
    perturbed,
    grad_arg,
    x0,
    solver,
    b_solver,
    aux_update,
    aux_init,
):
    del perturbed, x0, solver, aux_update, aux_init
    x_star, n_steps = residuals
    grad_x = gradients.value
    f, f_args, f_kwargs = grad_arg

    damp = b_solver.damp
    neumann_steps = b_solver.max_steps

    def _relaxed(f_spec, _x):
        _f, _fa, _fkw = f_spec
        return (1 - damp) * _x + damp * _f(_x, *_fa, **_fkw)

    _, pull = eqx.filter_vjp(_relaxed, (f, f_args, f_kwargs), x_star)

    def _neumann_step(_, g_hat):
        _, g_x = pull(g_hat)
        return grad_x + g_x

    k = jnp.maximum(1, jnp.minimum(neumann_steps, n_steps))
    g_hat = jax.lax.fori_loop(0, k - 1, _neumann_step, grad_x)

    grads, _ = pull(g_hat)
    return grads


class NeumannPhantom(Adjoint):
    """Neumann-series Phantom Gradient adjoint.

    From Geng et al. (2022):
        *On Training Implicit Models*
        doi:10.48550/arXiv.2111.05177

    Approximates ``(I − J_f^T)^{-1}`` via the truncated Neumann series::

        (I − J_f^T)^{-1} ≈ I + J_f^T + (J_f^T)² + … + (J_f^T)^{K−1}.

    The number of terms ``K`` is ``min(b_solver.max_steps, n_steps)``,
        where ``n_steps`` is the number of forward iterations.
    Increasing ``K`` reduces gradient bias
        at the cost of additional VJPs.

    Unlike :class:`UnrollPhantom`,
        this does not require a differentiable ``loop_kind`` in ``b_solver``:
        it is computed via ``jax.lax.fori_loop``.

    Args:
        solver: Forward fixed-point solver (any ``loop_kind``).
        b_solver: A :class:`~banax.solver.Relaxed` instance whose
            ``max_steps`` controls the number of Neumann terms and whose
            ``damp`` factor β is used in the relaxed backward map
            ``g ↦ ∇_{x*} L + β J_f^T g``.
    """

    solver: Solver
    b_solver: Relaxed

    def _loop(
        self,
        f_spec: FSpec[T],
        x0: T,
        *,
        aux_update: Callable[..., PyTree] | None = None,
        aux_init: PyTree | None = None,
    ) -> Solution:
        grad_arg = _normalize_f_spec(f_spec)
        return _neumann(grad_arg, x0, self.solver, self.b_solver, aux_update, aux_init)


@eqx.filter_custom_vjp
def _rev(grad_arg, x0, solver, aux_update=None, aux_init=None):
    solution, _ = solver._solve(
        grad_arg, x0=x0, aux_update=aux_update, aux_init=aux_init
    )
    return solution


@_rev.def_fwd
def _rev_fwd(perturbed, grad_arg, x0, solver, aux_update=None, aux_init=None):
    del perturbed

    solution, solver_state = solver._solve(
        grad_arg, x0=x0, aux_update=aux_update, aux_init=aux_init
    )

    w_star, _ = solver_state

    return solution, (solution.value, w_star, solution.stats.steps)


@_rev.def_bwd
def _rev_bwd(
    residuals, gradients, perturbed, grad_arg, x0, solver, aux_update, aux_init
):
    del perturbed, x0, aux_update, aux_init
    f, f_args, f_kwargs = grad_arg
    damp = solver.damp

    def _loop_cond(_carry):
        step, *_ = _carry
        return step > 0

    def _loop_body(_carry):
        (
            step,
            x_cur,
            w_cur,
            g_x,
            g_w,
            g_f,
            g_args,
            g_kwargs,
        ) = _carry

        # Reconstruct the previous iterates by inverting the paired update.
        x_prev = ((x_cur**ω - damp * f(w_cur, *f_args, **f_kwargs) ** ω) / (1 - damp)).ω
        w_prev = (
            (w_cur**ω - damp * f(x_prev, *f_args, **f_kwargs) ** ω) / (1 - damp)
        ).ω

        pull_w_cur = _apply_pullback(f, f_args, f_kwargs, w_cur)
        (dgrad_f_w_cur, dgrad_args_w_cur, dgrad_kwargs_w_cur), dgrad_w_cur = pull_w_cur(
            g_x
        )

        g_w = (g_w**ω + damp * dgrad_w_cur**ω).ω

        pull_x_prev = _apply_pullback(f, f_args, f_kwargs, x_prev)
        (dgrad_f_x_prev, dgrad_args_x_prev, dgrad_kwargs_x_prev), dgrad_x_prev = (
            pull_x_prev(g_w)
        )

        g_w_prev = ((1 - damp) * g_w**ω).ω
        g_x_prev = ((1 - damp) * g_x**ω + damp * dgrad_x_prev**ω).ω

        def _add_relaxed_dgrads(_grad, _dgrad_w_cur, _dgrad_x_prev):
            return eqx.apply_updates(
                _grad, (damp * (_dgrad_w_cur**ω + _dgrad_x_prev**ω)).ω
            )

        g_f = _add_relaxed_dgrads(g_f, dgrad_f_w_cur, dgrad_f_x_prev)
        g_args = _add_relaxed_dgrads(g_args, dgrad_args_w_cur, dgrad_args_x_prev)
        g_kwargs = _add_relaxed_dgrads(
            g_kwargs, dgrad_kwargs_w_cur, dgrad_kwargs_x_prev
        )

        step -= 1

        return step, x_prev, w_prev, g_x_prev, g_w_prev, g_f, g_args, g_kwargs

    def _filter_zeros_like(_z: PyTree) -> PyTree:
        diff = eqx.filter(_z, eqx.is_inexact_array)
        return jax.tree.map(jnp.zeros_like, diff)

    x_star, w_star, n_steps = residuals
    grad_x = gradients.value

    g_w = jax.tree.map(jnp.zeros_like, w_star)

    g_f_init = _filter_zeros_like(f)
    g_args_init = _filter_zeros_like(f_args)
    g_kwargs_init = _filter_zeros_like(f_kwargs)

    init_carry = (
        n_steps,
        x_star,
        w_star,
        grad_x,
        g_w,
        g_f_init,
        g_args_init,
        g_kwargs_init,
    )
    final_carry = eqxi.while_loop(_loop_cond, _loop_body, init_carry, kind="lax")

    *_, grad_f, grad_args, grad_kwargs = final_carry
    return grad_f, grad_args, grad_kwargs


class Reversible(Adjoint):
    """Reversible DEQ adjoint.

    From McCallum et al. (2025):
        *Reversible Deep Equilibrium Models*
        doi:10.48550/arXiv.2509.12917

    The backward pass reconstructs the forward iteration trajectory in reverse
        by inverting the :class:`~banax.solver.Reversible` paired-iterate update rule.
    Only the final pair (y*, z*) and the step count need to be stored:
        no intermediate iterates are retained,
        giving O(1) memory cost in ``max_steps``.

    .. note::
        This adjoint must be used with a :class:`~banax.solver.Reversible` solver.
        Using any other solver raises :exc:`TypeError`.

    Args:
        solver: A :class:`~banax.solver.Reversible` instance.
    """

    solver: ReversibleSolver

    def __init__(self, solver: ReversibleSolver):
        if not isinstance(solver, ReversibleSolver):
            raise TypeError(
                f"Reversible adjoint requires a Reversible solver, got {type(solver).__name__}"
            )
        self.solver = solver

    def _loop(
        self,
        f_spec: FSpec[T],
        x0: T,
        *,
        aux_update: Callable[..., PyTree] | None = None,
        aux_init: PyTree | None = None,
    ) -> Solution:
        grad_arg = _normalize_f_spec(f_spec)
        return _rev(grad_arg, x0, self.solver, aux_update, aux_init)


@eqx.filter_custom_vjp
def _gdeq(grad_arg, x0, solver, aux_update=None, aux_init=None):
    solution, _ = solver._solve(
        grad_arg, x0=x0, aux_update=aux_update, aux_init=aux_init
    )
    return solution


@_gdeq.def_fwd
def _gdeq_fwd(perturbed, grad_arg, x0, solver, aux_update=None, aux_init=None):
    del perturbed

    solution, solver_state = solver._solve(
        grad_arg, x0=x0, aux_update=aux_update, aux_init=aux_init
    )
    _, _, U, V, idx = solver_state  # Broyden state: (fx, g_flat, U, V, idx)
    return solution, (solution.value, U, V, idx)


@_gdeq.def_bwd
def _gdeq_bwd(
    residuals, gradients, perturbed, grad_arg, x0, solver, aux_update, aux_init
):
    del perturbed, x0, solver, aux_update, aux_init
    f, f_args, f_kwargs = grad_arg
    x_star, U, V, idx = residuals
    grad_x = gradients.value

    history_size = U.shape[0]
    valid_mask = (jnp.arange(history_size) < jnp.minimum(idx, history_size)).astype(
        U.dtype
    )

    grad_flat, unflatten = jax.flatten_util.ravel_pytree(grad_x)
    g_tilde_flat = grad_flat - V.T @ (valid_mask * (U @ grad_flat))  # B^T · grad
    g_tilde = unflatten(g_tilde_flat)

    pull = _apply_pullback(f, f_args, f_kwargs, x_star)
    grads, _ = pull(g_tilde)
    return grads


class GDEQ(Adjoint):
    """GDEQ adjoint: JFB with Broyden inverse-Jacobian preconditioning.

    From Nguyen et al. (2023):
        *Efficient Training of Deep Equilibrium Models*
        10.48550/arXiv.2304.11663

    Approximates the IFT gradient more accurately than JFB
        by preconditioning the outgoing adjoint with ``-B^T``,
        where ``B = I - U^T V`` is the Broyden limited-memory
        inverse Jacobian from the forward solve.

    Requires a :class:`~banax.solver.Broyden` forward solver
        (the ``U``, ``V`` factors are read from its state).

    Args:
        solver: A :class:`~banax.solver.Broyden` instance.
    """

    solver: Broyden

    def __init__(self, solver: Broyden):
        if not isinstance(solver, Broyden):
            raise TypeError(
                f"GDEQ adjoint requires a Broyden solver, got {type(solver).__name__}"
            )
        self.solver = solver

    def _loop(
        self,
        f_spec: FSpec[T],
        x0: T,
        *,
        aux_update: Callable[..., PyTree] | None = None,
        aux_init: PyTree | None = None,
    ) -> Solution:
        grad_arg = _normalize_f_spec(f_spec)
        return _gdeq(grad_arg, x0, self.solver, aux_update, aux_init)
