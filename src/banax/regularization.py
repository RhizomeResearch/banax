"""Jacobian regularization utilities for DEQ training.

These functions compute penalty terms
    that encourage the Jacobian of ``f`` at the fixed point
    to have small spectral or Frobenius norm,
    which promotes stability and faster convergence
    of the fixed-point iteration.

All functions accept an :obj:`~banax._core.FSpec
    and the fixed-point ``x_star``,
    and return a scalar loss term
    to be added to the training objective.
"""

import jax
import jax.numpy as jnp
import equinox as eqx

from jaxtyping import PRNGKeyArray
from banax._core import T, FSpec, _normalize_f_spec


def _randn_like(key, x):
    leaves, treedef = jax.tree.flatten(x)
    keys = jax.random.split(key, len(leaves))
    return treedef.unflatten(
        [
            jax.random.normal(k, leaf.shape, dtype=leaf.dtype)
            for k, leaf in zip(keys, leaves)
        ]
    )


def _norm(x):
    return jnp.sqrt(sum(jnp.sum(leaf**2) for leaf in jax.tree.leaves(x)))


def _size(x):
    return sum(leaf.size for leaf in jax.tree.leaves(x))


def jacobian_spectral_norm(
    f_spec: FSpec,
    x_star: T,
    eps: float = 1e-6,
    n_steps: int = 4,
    *,
    key: PRNGKeyArray,
) -> jax.Array:
    """Estimate the spectral norm of the Jacobian of ``f`` at ``x_star``.

    Uses ``n_steps`` iterations of power iteration on the Gram matrix J^T J.

    Args:
        f_spec: The fixed-point function; see :obj:`~banax._core.FSpec`.
        x_star: The fixed point at which to evaluate the Jacobian.
        eps: Small constant to prevent division by zero in normalisation.
        n_steps: Number of power iteration steps.
            More steps give a more accurate estimate
            at the cost of additional JVP/VJP calls.
        key: JAX PRNG key for the initial random vector.

    Returns:
        Scalar estimate of ``‖J_f(x*)‖_2``.
    """
    f, f_args, f_kwargs = _normalize_f_spec(f_spec)

    def _f(_x):
        return f(_x, *f_args, **f_kwargs)

    _, push = jax.linearize(_f, x_star)
    _, pull = jax.vjp(_f, x_star)

    def _normalize(_v):
        return jax.tree.map(lambda leaf: leaf / jnp.maximum(_norm(_v), eps), _v)

    def _j_gram(_v):
        return pull(push(_v))[0]

    v = _normalize(_randn_like(key, x_star))

    v_final = jax.lax.fori_loop(0, n_steps, lambda _, _u: _normalize(_j_gram(_u)), v)
    v_final = jax.lax.stop_gradient(v_final)

    return _norm(push(v_final))


def denoising_energy(
    f_spec: FSpec,
    x_star: T,
    sigma: float = 1.0,
    *,
    key: PRNGKeyArray,
) -> jax.Array:
    """Denoising Jacobian regularization loss.

    From Perschewski and Stober (2025):
        *Efficient Deep Equilibrium Models [...]
        doi:10.1016/j.procs.2025.07.136

    Computes ``‖f(x* + ε) − x*‖² / d``
        where ``ε ~ N(0, σ²I)``
        and ``d`` is the dimensionality of ``x_star``.
    This approximates the expected squared deviation of ``f``
        from the fixed point under Gaussian noise,
        which is related to the Frobenius norm of J_f for small σ.

    Args:
        f_spec: The fixed-point function; see :obj:`~banax._core.FSpec`.
        x_star: The fixed point.
        sigma: Standard deviation of the noise perturbation.
        key: JAX PRNG key.

    Returns:
        Scalar regularization loss.
    """
    f, f_args, f_kwargs = _normalize_f_spec(f_spec)
    x_star = jax.lax.stop_gradient(x_star)
    noise = jax.tree.map(lambda leaf: leaf * sigma, _randn_like(key, x_star))
    x_noisy = jax.tree.map(jnp.add, x_star, noise)
    delta = jax.tree.map(jnp.subtract, f(x_noisy, *f_args, **f_kwargs), x_star)
    return sum(jnp.sum(leaf**2) for leaf in jax.tree.leaves(delta)) / _size(x_star)


def hutchinson_jacobian_frobenius(
    f_spec: FSpec,
    x_star: T,
    n_steps: int = 4,
    *,
    key: PRNGKeyArray,
) -> jax.Array:
    """Estimate the scaled squared Frobenius norm of the Jacobian via Hutchinson's estimator.

    Computes an unbiased estimate of ``tr(J^T J) / d``
        where ``J := (∂f/∂x)(x_star)``
        and ``d`` is the dimensionality of ``x_star``,
        using ``n_steps`` random probe vectors.

    The estimator is ``(1/n_steps) Σ_i ‖J^T εᵢ‖² / d``,
        where each ``εᵢ`` is an i.i.d. standard Gaussian vector.
    Increasing ``n_steps`` reduces variance
        at the cost of ``n_steps`` additional VJP calls.

    Args:
        f_spec: The fixed-point function; see :obj:`~banax._core.FSpec`.
        x_star: The fixed point at which to evaluate the Jacobian.
        n_steps: Number of random probe vectors.
            More steps reduce variance but increase computation linearly.
        key: JAX PRNG key.

    Returns:
        Scalar estimate of ``tr(J^T J) / d``.
    """
    f, f_args, f_kwargs = _normalize_f_spec(f_spec)

    def _f(_x):
        return f(_x, *f_args, **f_kwargs)

    x_star = jax.lax.stop_gradient(x_star)
    _, pull = eqx.filter_vjp(_f, x_star)
    d = _size(x_star)

    def _body(n, acc):
        eps = _randn_like(jax.random.fold_in(key, n), x_star)
        return acc + sum(jnp.sum(leaf**2) for leaf in jax.tree.leaves(pull(eps)[0])) / d

    reg = jax.lax.fori_loop(0, n_steps, _body, 0.0)
    return reg / n_steps
