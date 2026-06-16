# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "jax>=0.9.2",
# ]
# ///
import jax
import jax.numpy as jnp
from jax import Array


# Some function to differentiate
def f(xy: Array, c: float) -> Array:
  x, y = xy
  exp = jnp.exp(x * x)
  f1 = jnp.cos(y) * exp + c
  f2 = jnp.sin(y) * exp + c
  return jnp.array([f1, f2])


xy = jnp.array([1.0, -jnp.pi / 4])
c = 1.0  # constant parameter
print(f(xy, c))
# [ 2.9221153  -0.92211545]

# Gradient (w.r.t. xy)
f1 = lambda xy, c: f(xy, c)[0]
print(jax.grad(f1)(xy, c))
# [ 3.844231   1.9221154]
f2 = lambda xy, c: f(xy, c)[1]
print(jax.grad(f2)(xy, c))
# [-3.844231   1.9221154]

# Jacobian matrix (w.r.t. xy)
dfdx, dfdy = jax.jacobian(f)(xy, c).T
print(dfdx)
# [ 3.844231   -3.844231]
print(dfdy)
# [-1.9221154   1.9221154]
