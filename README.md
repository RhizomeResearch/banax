# banax

Deep equilibrium models in JAX/Equinox.

A deep equilibrium model (DEQ) replaces a deep network with the fixed point of a
contractive function `f`: instead of unrolling layers, it solves `f(x) = x` and
differentiates through the solution. **banax** provides the solvers that find those
fixed points and the adjoint methods that differentiate through them.

## Installation

```bash
uv add banax
# or, using the legacy `pip`
pip install banax
```

## Library layout

```
banax/
  solver.py         — iterative fixed-point solvers (Picard, Relaxed, Reversible)
  adjoint.py        — adjoint / differentiation methods (BPTT, JFB, Implicit, …)
  regularization.py — Jacobian regularization utilities
  _core.py          — shared types (T, FSpec, …)
```

The main entry point is an `Adjoint`. It wraps a `Solver` and exposes a single
`()` method that finds the fixed point and handles gradients.

## Basic usage

```python
import jax.numpy as jnp
from banax.solver import Picard
from banax.adjoint import BPTT

def f(x, W, b):
    return jnp.tanh(W @ x + b)

solver = Picard(rtol=1e-5, max_steps=50, loop_kind="checkpointed")
adjoint = BPTT(solver=solver)

x0 = jnp.zeros(64)
sol = adjoint((f, (W, b)), x0)
x_star = sol.value          # fixed point; carries gradients
steps  = sol.stats.steps    # number of iterations taken
```

Calling the adjoint returns a `Solution` object. The fixed point `sol.value`
carries gradients — use it in a loss and call `jax.grad` normally.

## f_spec

Functions are passed as an **`FSpec`**: a bare callable, or a tuple bundling the
callable with its extra arguments.

```python
# bare callable — f takes only x
adjoint(f, x0)

# tuple with positional args — f(x, *args)
adjoint((f, (W, b)), x0)

# tuple with positional and keyword args — f(x, *args, **kwargs)
adjoint((f, (W,), {"bias": b}), x0)
```

This convention appears consistently across solvers, adjoints, and regularization
functions.

## Solvers

All solvers inherit from `Solver` and share the same keyword arguments:

| argument | default | meaning |
|---|---|---|
| `atol` | `0.0` | absolute tolerance (disabled if `0.0`) |
| `rtol` | `1e-5` | relative tolerance (disabled if `0.0`) |
| `max_steps` | `50` | iteration cap |
| `loop_kind` | `"lax"` | `"lax"` / `"bounded"` / `"checkpointed"` |
| `damp` | `0.8` | damping factor β (`Relaxed` and `Reversible` only) |

`loop_kind` controls how equinox unrolls the iteration. `"lax"` uses
`jax.lax.while_loop` (not differentiable through the loop). `"bounded"` and
`"checkpointed"` are differentiable — required when using `BPTT`. `"checkpointed"`
trades memory for recomputation.

```python
from banax.solver import Picard, Relaxed, Reversible as ReversibleSolver

Picard(atol=1e-5, max_steps=50)
Relaxed(damp=0.8, atol=1e-5, rtol=0.0, max_steps=50)
ReversibleSolver(damp=0.8, atol=1e-5, max_steps=20)
```

`Relaxed` applies damping: `x ← (1 − β) x + β f(x)` where `damp=β`. `Reversible`
uses a two-sequence scheme that reconstructs the iteration trajectory during the
backward pass without storing all intermediate iterates.

## Adjoint methods

Adjoint methods control how gradients flow through the fixed-point equation.
They all wrap a `Solver` and expose the same `()` interface.

| class | gradient method | notes |
|---|---|---|
| `BPTT` | backprop through the unrolled iterations | exact; solver needs `loop_kind="bounded"` or `"checkpointed"` |
| `Implicit` | implicit function theorem (IFT) | exact; requires a second `b_solver` for the backward linear system |
| `JFB` | Jacobian-free backprop | biased; cheap; one VJP per step |
| `UnrollPhantom` | unrolled phantom gradient | interpolates between JFB and BPTT |
| `NeumannPhantom` | Neumann-series phantom gradient | similar to UnrollPhantom via Neumann expansion |
| `Reversible` | reversible adjoint | exact; O(1) memory; pairs with `ReversibleSolver` |

```python
from banax.adjoint import BPTT, Implicit, JFB
from banax.solver import Picard

solver = Picard(atol=1e-5, max_steps=50, loop_kind="checkpointed")
b_solver = Picard(rtol=1e-8, max_steps=50, loop_kind="checkpointed")

# Exact gradient via backprop
sol = BPTT(solver=solver)((f, (W, b)), x0)

# Exact gradient via IFT
sol = Implicit(solver=solver, b_solver=b_solver)((f, (W, b)), x0)

# Cheap biased gradient
sol = JFB(solver=Picard(atol=1e-6, max_steps=100))((f, (W, b)), x0)

x_star = sol.value
```

### Auxiliary state

Pass `aux_update` and `aux_init` to accumulate state across iterations.
The result is returned in `sol.aux`.

```python
max_steps = 50

# aux is a (max_steps, 64) array; each iteration writes x into the next row
def record_iterates(step, aux, x, fx, state):
    return aux.at[step].set(x)

sol = adjoint(
    (f, (W, b)), x0,
    aux_update=record_iterates,
    aux_init=jnp.zeros((max_steps, 64)),
)

# sol.aux[n_steps:] are the unused rows (still zeros)
trajectory = sol.aux[:sol.stats.steps]
```

## Regularization

Three Jacobian regularizers for penalizing the spectral or Frobenius norm of
`df/dx` at the fixed point, all accepting an `FSpec`:

```python
from banax.regularization import (
    jacobian_spectral_norm,
    denoising_energy,
    hutchinson_jacobian_frobenius,
)
import jax

key = jax.random.key(0)

# Spectral norm via power iteration
sigma = jacobian_spectral_norm((f, (W, b)), x_star, key=key, n_steps=5)

# Denoising energy (Perschewski & Stober 2025)
energy = denoising_energy((f, (W, b)), x_star, sigma=0.1, key=key)

# Scaled Frobenius norm via Hutchinson estimator
frob = hutchinson_jacobian_frobenius((f, (W, b)), x_star, n_steps=10, key=key)
```

Add any of these as a penalty term to your training loss.

## License

MIT. See [LICENSE.md](LICENSE.md).
