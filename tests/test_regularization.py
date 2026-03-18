import jax
import jax.numpy as jnp
import pytest

from banax.regularization import (
    jacobian_spectral_norm,
    denoising_energy,
    hutchinson_jacobian_frobenius,
)

KEY = jax.random.PRNGKey(0)
X = jnp.array(1.5)  # scalar test point
A = jnp.array(0.3)  # contraction factor


def scalar_contraction(x, a, b):
    return a * x + b


_FSPEC_FORMS = [
    ("bare", lambda: lambda z: A * z + 1.0),
    ("tuple", lambda: (scalar_contraction, (A, jnp.array(1.0)))),
    ("kwargs", lambda: (scalar_contraction, (), {"a": A, "b": jnp.array(1.0)})),
]
_FSPEC_IDS = ["bare", "tuple", "kwargs"]


# ── TestJacobianSpectralNorm ──────────────────────────────────────────────


class TestJacobianSpectralNorm:
    @pytest.mark.parametrize("make_f", [f for _, f in _FSPEC_FORMS], ids=_FSPEC_IDS)
    def test_fspec_forms(self, make_f):
        result = jacobian_spectral_norm(make_f(), X, key=KEY)
        assert jnp.isfinite(result)

    def test_value_close_to_a(self):
        result = jacobian_spectral_norm(
            (scalar_contraction, (A, jnp.array(1.0))), X, key=KEY, n_steps=20
        )
        assert jnp.allclose(result, jnp.abs(A), atol=1e-4)

    def test_returns_scalar(self):
        result = jacobian_spectral_norm(lambda z: 0.5 * z, X, key=KEY)
        assert result.shape == ()

    def test_nonnegative(self):
        result = jacobian_spectral_norm(lambda z: 0.5 * z, X, key=KEY)
        assert result >= 0.0

    def test_grad_flows(self):
        def reg(a):
            return jacobian_spectral_norm(
                (scalar_contraction, (a, jnp.array(1.0))), X, key=KEY
            )

        grad = jax.grad(reg)(A)
        assert jnp.isfinite(grad)


# ── TestDenoisingEnergy ───────────────────────────────────────────────────


class TestDenoisingEnergy:
    @pytest.mark.parametrize("make_f", [f for _, f in _FSPEC_FORMS], ids=_FSPEC_IDS)
    def test_fspec_forms(self, make_f):
        result = denoising_energy(make_f(), X, key=KEY)
        assert jnp.isfinite(result)

    def test_nonnegative(self):
        result = denoising_energy(lambda z: 0.5 * z, X, key=KEY)
        assert result >= 0.0

    def test_returns_scalar(self):
        result = denoising_energy(lambda z: 0.5 * z, X, key=KEY)
        assert result.shape == ()

    def test_sigma_zero_gives_zero_energy(self):
        # With sigma=0, noise=0, delta = f(x_star) - x_star.
        # That equals 0 only when x_star is a true fixed point.
        # f(z) = az+b with a=0.5, b=1 has fixed point x*=2.
        x_fp = jnp.array(2.0)
        result = denoising_energy(
            (scalar_contraction, (jnp.array(0.5), jnp.array(1.0))),
            x_fp,
            sigma=0.0,
            key=KEY,
        )
        assert jnp.allclose(result, 0.0, atol=1e-5)

    def test_grad_flows(self):
        def reg(a):
            return denoising_energy(
                (scalar_contraction, (a, jnp.array(1.0))), X, key=KEY
            )

        grad = jax.grad(reg)(A)
        assert jnp.isfinite(grad)


# ── TestHutchinsonJacobianFrobenius ───────────────────────────────────────


class TestHutchinsonJacobianFrobenius:
    @pytest.mark.parametrize("make_f", [f for _, f in _FSPEC_FORMS], ids=_FSPEC_IDS)
    def test_fspec_forms(self, make_f):
        result = hutchinson_jacobian_frobenius(make_f(), X, n_steps=10, key=KEY)
        assert jnp.isfinite(result)

    def test_value_close_to_a_squared(self):
        result = hutchinson_jacobian_frobenius(
            (scalar_contraction, (A, jnp.array(1.0))), X, n_steps=100, key=KEY
        )
        assert jnp.allclose(result, A**2, atol=0.05)

    def test_returns_scalar(self):
        result = hutchinson_jacobian_frobenius(
            lambda z: 0.5 * z, X, n_steps=10, key=KEY
        )
        assert result.shape == ()

    def test_nonnegative(self):
        result = hutchinson_jacobian_frobenius(
            lambda z: 0.5 * z, X, n_steps=10, key=KEY
        )
        assert result >= 0.0

    def test_grad_flows(self):
        def reg(a):
            return hutchinson_jacobian_frobenius(
                (scalar_contraction, (a, jnp.array(1.0))), X, n_steps=10, key=KEY
            )

        grad = jax.grad(reg)(A)
        assert jnp.isfinite(grad)
