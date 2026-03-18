"""Core types shared across banax.

Public type aliases and data classes used by solvers, adjoints, and
regularization utilities.  Import these directly if you need to annotate
types in downstream code:

    from banax._core import FSpec, Solution, Result, Stats
"""

from typing import Callable, Literal
from jaxtyping import Array, ArrayLike, Float, Int, PyTree, Shaped

import equinox as eqx

# ── Type aliases ──────────────────────────────────────────────────────────

type T = PyTree[Shaped[ArrayLike, "?*x"]]
"""Any JAX-compatible pytree of arrays; the state type for fixed-point iterations."""

type Fn[Z] = Callable[[Z], Z]
"""A fixed-point map: a callable from Z to Z whose fixed point we seek."""

type Error = Float[Array, ""]
"""Scalar float array representing a convergence error metric."""

type Step = Int[Array, ""]
"""Scalar integer array representing an iteration count."""

type DifferentiableLoopKind = Literal["bounded", "checkpointed"]
"""Loop kinds that support automatic differentiation through the iteration.

- ``"bounded"``: unrolls the loop up to ``max_steps`` steps; fully differentiable
  but uses O(max_steps) memory.
- ``"checkpointed"``: like ``"bounded"`` but applies gradient checkpointing,
  trading recomputation for memory (O(log max_steps) memory).
"""

type EquinoxLoopKind = DifferentiableLoopKind | Literal["lax"]
"""All supported loop kinds.

- ``"lax"``: uses ``jax.lax.while_loop``; minimal memory, not differentiable
  through the loop.  Suitable for forward-only evaluation or custom-VJP adjoints.
- ``"bounded"``, ``"checkpointed"``: see :obj:`DifferentiableLoopKind`.
"""

type FSpec[Z] = (
    Callable[..., Z]
    | tuple[Callable[..., Z], tuple]
    | tuple[Callable[..., Z], tuple, dict]
)
"""Flexible function specification accepted by solvers and adjoints.

Three forms are valid:

- ``f`` — bare callable; invoked as ``f(x)``.
- ``(f, args)`` — called as ``f(x, *args)``.
- ``(f, args, kwargs)`` — called as ``f(x, *args, **kwargs)``.

This convention appears consistently across :mod:`banax.solver`,
:mod:`banax.adjoint`, and :mod:`banax.regularization`.
"""


# ── Result codes ─────────────────────────────────────────────────────────


class Result:
    """Integer result codes returned in :attr:`Solution.result`.

    Attributes:
        CONVERGED: ``0`` — the solver met the tolerance criterion before
            reaching ``max_steps``.
        MAX_STEPS: ``1`` — the solver exhausted ``max_steps`` without
            satisfying the tolerance.  The value may still be a useful
            approximation.
    """

    CONVERGED = 0
    MAX_STEPS = 1


# ── Data classes ─────────────────────────────────────────────────────────


class Stats(eqx.Module):
    """Convergence statistics attached to every :class:`Solution`.

    Attributes:
        steps: Number of fixed-point iterations executed.
        abs_err: Final absolute error ``‖f(x) − x‖`` at the last iterate.
            Zero if ``atol=0.0`` (criterion disabled).
        rel_err: Final relative error ``‖f(x) − x‖ / ‖f(x)‖`` at the last
            iterate.  Zero if ``rtol=0.0`` (criterion disabled).
    """

    steps: Step
    abs_err: Error
    rel_err: Error


class Solution(eqx.Module):
    """Return value of every :class:`~banax.adjoint.Adjoint`.

    Attributes:
        value: The fixed-point iterate ``x*`` (or the best approximation
            reached within ``max_steps``).  Carries gradients when used
            inside a differentiable context.
        result: :attr:`Result.CONVERGED` or :attr:`Result.MAX_STEPS` as an
            ``Int[Array, ""]``.
        stats: Convergence statistics; see :class:`Stats`.
        aux: Accumulated auxiliary state from ``aux_update``, or ``None``
            when no ``aux_update`` was provided.
    """

    value: T
    result: Step  # Result.CONVERGED or Result.MAX_STEPS as Int[Array, ""]
    stats: Stats
    aux: PyTree | None


def _normalize_f_spec(f_spec):
    if callable(f_spec):
        return f_spec, (), {}
    f, *rest = f_spec
    f_args = rest[0] if len(rest) > 0 else ()
    f_kwargs = rest[1] if len(rest) > 1 else {}
    return f, f_args, f_kwargs
