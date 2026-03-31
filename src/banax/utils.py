"""Utility helpers for banax.

Includes:

- ``trace_*`` helpers — return ``(trace_fn, trace_init)`` tuples for the
  ``trace=`` argument of any :class:`~banax.adjoint.Adjoint`.
- ``half_normal_like`` — generate a random PyTree with ~50 % zeros.
- ``zeros_like`` — generate a PyTree with zero values.

Example usage::

    import jax
    import jax.numpy as jnp
    from banax.utils import trace_history, trace_last_aux, trace_count, half_normal

    # Collect the absolute residual at every f evaluation
    sol = adjoint(
        f_spec, x0,
        trace=trace_history(lambda x, fx, f_aux: jnp.abs(fx - x).mean(), 50, 0.0),
    )
    count, residuals = sol.trace

    # Capture the final aux returned by f
    sol = adjoint(
        f_spec, x0, has_aux=True,
        trace=trace_last_aux(jnp.zeros(hidden_dim)),
    )
    final_hidden = sol.trace

    # Random initialisation with ~half sparsity
    key = jax.random.PRNGKey(0)
    x0 = half_normal_like(key, template_pytree)
"""

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from banax import T


def trace_last(select_fn, init_value):
    """Capture the last value of ``select_fn(x, fx, f_aux)`` across all evaluations.

    Args:
        select_fn: ``(x, fx, f_aux) -> value`` extracts the quantity to keep.
            Must return the same PyTree structure and shape on every call.
        init_value: A value with the same structure, shape, and dtype
            as ``select_fn``'s output.
            Required because it enters the solver's ``while_loop`` carry,
            which has a statically fixed structure in JAX.

    Returns:
        A ``(trace_fn, trace_init)`` pair for ``trace=``.
        After solving, ``sol.trace`` is the value from the *last* ``f`` evaluation.
    """
    return (lambda acc, x, fx, f_aux: select_fn(x, fx, f_aux), init_value)


def trace_last_aux(init_aux):
    """Capture the last ``f_aux`` returned by ``f``.

    Convenience wrapper around :func:`trace_last` for ``has_aux=True`` functions.

    Args:
        init_aux: A value with the same PyTree structure, shape, and dtype as ``f_aux``.
        Required for the static JAX carry structure.

    Returns:
        A ``(trace_fn, trace_init)`` pair for ``trace=``.
        After solving, ``sol.trace`` is the ``f_aux`` from the final evaluation.
    """
    return trace_last(lambda x, fx, f_aux: f_aux, init_aux)


def trace_history(select_fn, n_evals, init_value):
    """Accumulate ``select_fn(x, fx, f_aux)`` into a buffer at every ``f`` evaluation.

    Args:
        select_fn: ``(x, fx, f_aux) -> value`` — extracts the quantity to record.
            Must return consistent shape and dtype matching ``init_value``.
        n_evals: Buffer length.  Set to at least ``solver.max_steps + 1``
            (one extra slot for the ``init()`` call).
            For some solvers, e.g., Broyden with line search,
            budget additional slots for the sub-steps.
            Out-of-bounds writes are silently dropped by JAX's scatter semantics.
        init_value: A value (scalar or array)
            with the shape and dtype of ``select_fn``'s output,
            used to infer the buffer shape.

    Returns:
        A ``(trace_fn, trace_init)`` pair for ``trace=``.
        After solving, ``sol.trace`` is ``(count, buffer)`` where:

        - ``count``: number of ``f`` evaluations performed (scalar int array).
        - ``buffer``: array of shape ``(n_evals, *value_shape)``;
          ``buffer[i]`` is the value at evaluation ``i``.  Entries at indices
          ``>= count`` are zero-padded.  Outside JIT, use ``buffer[:int(count)]``
          to trim.
    """
    init_value = jnp.asarray(init_value)
    init_buf = jnp.zeros((n_evals,) + init_value.shape, dtype=init_value.dtype)

    def trace_fn(acc, x, fx, f_aux):
        idx, buf = acc
        val = select_fn(x, fx, f_aux)
        return (idx + 1, buf.at[idx].set(val))

    return (trace_fn, (jnp.array(0), init_buf))


def trace_count():
    """Count the total number of ``f`` evaluations inside the solver.

    Unlike ``sol.stats.steps``, which counts loop iterations,
    this counts all function evaluations,
    including the initial ``f(x0)`` call in ``init()``.

    Returns:
        A ``(trace_fn, trace_init)`` pair for ``trace=``.
        After solving, ``sol.trace`` is a scalar int array with the total count.
    """
    return (lambda acc, x, fx, f_aux: acc + 1, jnp.array(0))


# ── PyTree helpers ────────────────────────────────────────────────────────


def zeros_like(x: T):
    """Return a PyTree of zeros with the same structure, shapes, and dtypes.

    Like ``jnp.zeros_like``, but works on arbitrary pytrees of arrays.

    Args:
        x: Any JAX-compatible PyTree of floating-point arrays.

    Returns:
        A PyTree with the same structure as ``x``, with every leaf
        replaced by an array of zeros of matching shape and dtype.
    """
    return jax.tree.map(jnp.zeros_like, x)


def half_normal_like(key: PRNGKeyArray, x: T):
    """Return a random PyTree shaped like ``x`` with ~half the elements zero.

    Each leaf is masked independently: a Bernoulli(p=0.5) mask selects
    which elements receive an i.i.d. standard-normal value; the rest are zero.
    Works on arbitrary pytrees, not just arrays.

    Args:
        x: Any JAX-compatible PyTree of floating-point arrays,
            used only to determine structure, shapes, and dtypes.
        key: A JAX PRNG key.

    Returns:
        A PyTree with the same structure, shapes, and dtypes as ``x``.
    """
    leaves, treedef = jax.tree.flatten(x)
    keys = jax.random.split(key, 2 * len(leaves))
    new_leaves = []
    for i, leaf in enumerate(leaves):
        mask = jax.random.bernoulli(keys[2 * i], shape=leaf.shape)
        noise = jax.random.normal(keys[2 * i + 1], shape=leaf.shape, dtype=leaf.dtype)
        new_leaves.append(noise * mask.astype(noise.dtype))
    return treedef.unflatten(new_leaves)
