# Banax

Deep equilibrium models in JAX/Equinox.

A deep equilibrium model (DEQ) replaces a deep network
with the fixed point of a contractive function `f`:
instead of unrolling layers,
it solves `f(x) = x` and differentiates through the solution.
**Banax** provides the solvers that find those fixed points,
the adjoint methods that differentiate through them,
as well as utilities to train DEQ models
such as Jacobian regularization loss terms.

## Installation

```bash
uv add banax
# or, using the legacy `pip`
pip install banax
```

## Library layout

```
banax/
  solver.py         — iterative fixed-point solvers (Picard, Relaxed, Reversible, Broyden, Anderson)
  adjoint.py        — adjoint / differentiation methods (BPTT, JFB, Implicit, …)
  regularization.py — Jacobian regularization utilities
  utils.py          — trace helpers and PyTree utilities
  _core.py          — shared types (T, FSpec, …)
```

The main entry point is an `Adjoint`.
It wraps a `Solver` and exposes a single `__call__()` method
that finds the fixed point and handles gradients.

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

Calling the adjoint returns a `Solution` object.
The fixed point `sol.value` carries gradients:
use it in a loss and call `jax.grad` normally.

## f_spec

Functions are passed as an **`FSpec`**: a bare callable,
or a tuple bundling the callable with its extra arguments.

```python
# bare callable — f takes only x
adjoint(f, x0)

# tuple with positional args — f(x, *args)
adjoint((f, (W, b)), x0)

# tuple with positional and keyword args — f(x, *args, **kwargs)
adjoint((f, (W,), {"bias": b}), x0)
```

This convention appears consistently
across solvers, adjoints, and regularization functions.

## Solvers

| class | method | notes |
|---|---|---|
| `Picard` | standard fixed-point iteration | simplest; converges when spectral radius < 1 |
| `Relaxed` | damped iteration: `x ← (1−β)x + βf(x)` | `damp=β`; widens convergence basin |
| `Reversible` | two-sequence reversible scheme | O(1) memory backward pass; pairs with `Reversible` adjoint |
| `Broyden` | limited-memory quasi-Newton | rank-1 inverse Jacobian updates; optional Armijo line search |
| `Anderson` | Anderson acceleration | least-squares mixing over recent iterates; fastest near fixed point |

All solvers inherit from `Solver` and share the same keyword arguments:

| argument | default | meaning |
|---|---|---|
| `atol` | `0.0` | absolute tolerance (disabled if `0.0`) |
| `rtol` | `1e-5` | relative tolerance (disabled if `0.0`) |
| `max_steps` | `50` | iteration cap |
| `loop_kind` | `"lax"` | `"lax"` / `"bounded"` / `"checkpointed"` |
| `damp` | `0.8` | damping factor β (`Relaxed`, `Reversible`, and `Anderson`) |
| `history_size` | `10` | rank-1 update history (`Broyden` only) |
| `ls_steps` | `0` | Armijo backtracking halvings per step; `0` disables line search (`Broyden` only) |
| `depth` | `5` | mixing history length (`Anderson` only) |
| `ridge` | `1e-6` | normal-equation regularization (`Anderson` only) |
| `use_linalg` | `True` | if `False`, use hand-rolled Cholesky instead of `jnp.linalg.solve` (`Anderson` only) |

`loop_kind` controls how equinox unrolls the iteration. `"lax"` uses
`jax.lax.while_loop` (not differentiable through the loop). `"bounded"` and
`"checkpointed"` are differentiable — required when using `BPTT`. `"checkpointed"`
trades memory for recomputation.

```python
from banax.solver import Picard, Relaxed, Reversible as ReversibleSolver, Broyden, Anderson

Picard(atol=1e-5, max_steps=50)
Relaxed(damp=0.8, atol=1e-5, rtol=0.0, max_steps=50)
ReversibleSolver(damp=0.8, atol=1e-5, max_steps=20)
Broyden(history_size=10, atol=1e-5, max_steps=50)           # ls_steps=0: no line search
Broyden(history_size=10, ls_steps=5, atol=1e-5, max_steps=50)  # Armijo backtracking
Anderson(depth=5, damp=1.0, ridge=1e-6, atol=1e-5, max_steps=50)
```

`Relaxed` applies damping: `x ← (1 − β) x + β f(x)` where `damp=β`.
`Reversible` uses a two-sequence scheme
that reconstructs the iteration trajectory during the backward pass
without storing all intermediate iterates.

`Broyden` uses a limited-memory quasi-Newton update on the residual `g(x) = f(x) - x`,
maintaining a low-rank inverse Jacobian approximation.
Setting `ls_steps > 0` enables Armijo backtracking line search,
up to `ls_steps` step-size halvings per iteration;
this is incompatible with `BPTT` and `Reversible` adjoints.
`Anderson` acceleration solves a small least-squares problem over recent iterates
to find optimal mixing coefficients.
Both converge faster than Picard
when the Jacobian spectral radius is close to 1.

## Adjoint methods

Adjoint methods control how gradients flow through the fixed-point equation.
They all wrap a `Solver` and expose the same `__call__()` interface.

| class | gradient method | notes |
|---|---|---|
| `BPTT` | backprop through the unrolled iterations | exact; solver needs `loop_kind="bounded"` or `"checkpointed"` |
| `Implicit` | implicit function theorem (IFT) | exact; requires a second `b_solver` for the backward linear system |
| `JFB` | Jacobian-free backprop | biased; cheap; one VJP per step |
| `GDEQ` | JFB with Broyden preconditioning | less biased than JFB; requires a `Broyden` solver |
| `UnrollPhantom` | unrolled phantom gradient | interpolates between JFB and BPTT |
| `NeumannPhantom` | Neumann-series phantom gradient | similar to UnrollPhantom via Neumann expansion |
| `Reversible` | reversible adjoint | exact; O(1) memory; pairs with `ReversibleSolver` |

```python
from banax.adjoint import BPTT, Implicit, JFB, GDEQ
from banax.solver import Picard, Broyden

solver = Picard(atol=1e-5, max_steps=50, loop_kind="checkpointed")
b_solver = Picard(rtol=1e-8, max_steps=50, loop_kind="checkpointed")

# Exact gradient via backprop
sol = BPTT(solver=solver)((f, (W, b)), x0)

# Exact gradient via IFT
sol = Implicit(solver=solver, b_solver=b_solver)((f, (W, b)), x0)

# Cheap biased gradient (JFB)
sol = JFB(solver=Picard(atol=1e-6, max_steps=100))((f, (W, b)), x0)

# Better-conditioned biased gradient using Broyden's inverse-Jacobian factors
sol = GDEQ(solver=Broyden(atol=1e-6, max_steps=100))((f, (W, b)), x0)

x_star = sol.value
```

### Dynamic step budget

`max_steps` is a static field baked into the compiled JAX trace.
Changing it between calls triggers a recompile.

For strategies that vary the iteration depth at runtime,
such as progressive deepening, randomized step counts, curriculum schedules,
pass `step_budget` to the adjoint call instead:

```python
solver = Picard(atol=1e-6, max_steps=100)   # max_steps: compile-time ceiling
adjoint = JFB(solver=solver)

sol = adjoint((f, (W, b)), x0, step_budget=jnp.array(10))
sol = adjoint((f, (W, b)), x0, step_budget=jnp.array(50))
```

To avoid recompilation when varying the budget,
pass it as a JAX array inside a JIT-compiled function
so JAX traces it as an abstract value:

```python
@eqx.filter_jit
def train_step(model, x0, budget):
    sol = adjoint((model, ()), x0, step_budget=budget)
    return loss(sol.value)

train_step(model, x0, jnp.array(10))  # compiles once
train_step(model, x0, jnp.array(50))  # reuses compiled code
```

`step_budget` only accepts a JAX array (or `None`);
passing a plain Python `int` is a type error.
`max_steps` remains the hard ceiling:
a `step_budget` larger than `max_steps`
is silently clamped to `max_steps`.

### Function auxiliary output

If `f` returns a `(fx, f_aux)` pair alongside the fixed-point iterate,
pass `has_aux=True`:

```python
def f(x, W, b):
    pre = W @ x + b
    return jnp.tanh(pre), {"pre_activations": pre}   # (fx, f_aux)

sol = adjoint((f, (W, b)), x0, has_aux=True)
x_star = sol.value   # fixed point; f_aux is discarded unless trace is also provided
```

### Tracing f evaluations

Pass `trace=(trace_fn, trace_init)`
to fold over every `f` evaluation inside the solver,
including those inside `init()` and any line-search sub-steps.
The result is returned in `sol.trace`.

```python
# Count total f evaluations (init + every step)
def count_evals(acc, x, fx, f_aux):
    return acc + 1

sol = adjoint(
    (f, (W, b)), x0,
    trace=(count_evals, jnp.array(0)),
)
print(sol.trace)   # >= sol.stats.steps (init also calls f)
```

The trace function signature is `(acc, x, fx, f_aux) -> acc`.
`f_aux` is `None` unless `has_aux=True` is also passed.
`trace_init` must be a JAX value with the same PyTree structure and shapes
as the output of `trace_fn` — it enters the solver's `while_loop` carry,
which has a statically fixed structure.

### Trace helpers

`banax.utils` provides ready-made trace specs for common patterns:

```python
from banax import trace_last, trace_last_aux, trace_history, trace_count

# Last value of any projection over (x, fx, f_aux)
sol = adjoint((f, (W, b)), x0,
    trace=trace_last(lambda x, fx, f_aux: fx, jnp.zeros(64)))
sol.trace   # fx at the final evaluation

# Last f_aux (shorthand for the above when has_aux=True)
def f(x, W, b):
    pre = W @ x + b
    return jnp.tanh(pre), {"pre_activations": pre}

sol = adjoint((f, (W, b)), x0, has_aux=True,
    trace=trace_last_aux({"pre_activations": jnp.zeros(64)}))
sol.trace["pre_activations"]   # pre-activations at the final iterate

# History buffer: collect a scalar at every evaluation
sol = adjoint((f, (W, b)), x0,
    trace=trace_history(lambda x, fx, f_aux: jnp.linalg.norm(fx - x),
                        n_evals=solver.max_steps + 1,
                        init_value=0.0))
count, residuals = sol.trace   # residuals[i] is the value at evaluation i

# Count total f evaluations
sol = adjoint((f, (W, b)), x0, trace=trace_count())
sol.trace   # scalar int: total calls including init() and line-search sub-steps
```

`trace_history` returns `(count, buffer)` in `sol.trace`.
`buffer` has shape `(n_evals, *value_shape)`;
entries at indices `>= count` are zero-padded.
Set `n_evals` to at least `solver.max_steps + 1`.
For `Broyden` with `ls_steps > 0`,
budget additional slots for line-search sub-steps.

Some solvers may pass additional keyword arguments to the trace function
(e.g. `Broyden` with `ls_steps > 0` passes `tag="line_search"`
at line-search call sites).
If you use such a solver, accept `**kwargs` in your trace function.


# PyTree utilities

`banax.utils` also provides PyTree-aware analogues of common JAX array functions:

```python
from banax import zeros_like, half_normal_like
import jax

# Zero-valued PyTree with the same structure, shapes, and dtypes
x0 = zeros_like(model_state)

# Random PyTree with ~half the elements zero, the rest i.i.d. standard normal
key = jax.random.key(0)
x0 = half_normal_like(key, model_state)
```

`zeros_like` and `half_normal_like` accept any JAX PyTree,
not just plain arrays.

## Regularization

Three Jacobian regularizers for penalizing
the spectral or Frobenius norm of `df/dx` at the fixed point,
all accepting an `FSpec`:

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

## Acknowledgements

**Banax** was inspired by and learned from several excellent projects:

- [**torchdeq**](https://github.com/locuslab/torchdeq) — a comprehensive DEQ library for PyTorch that shaped many of the solver and adjoint interfaces here
- [**revdeq**](https://github.com/sammccallum/revdeq) — the reversible DEQ adjoint that motivated the `Reversible` solver/adjoint pair
- [**optimistix**](https://github.com/patrick-kidger/optimistix) — a JAX-based nonlinear solvers library whose clean design influenced the solver API

Grateful to the [**JAX**](https://github.com/google/jax)
and [**Equinox**](https://github.com/patrick-kidger/equinox) teams
for the foundations that make this library possible.

## License

MIT. See [LICENSE.md](LICENSE.md).
