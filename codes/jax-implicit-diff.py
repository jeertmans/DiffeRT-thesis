# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "jax>=0.9.2",
# ]
# ///
from functools import partial

import jax
from jax import Array


# Return path length
def L(T: Array, theta: Array) -> Array: ...


# Return optimal path T_opt
@partial(
  jax.custom_vjp,
  nondiff_argnames=("static_params",),
)
def f(
  theta: Array, *static_params
) -> Array: ...


# Compute optimal path and any intermediate
# values needed for the backward pass
def f_fwd(
  theta: Array, *static_params
) -> Array:
  T_opt = f(theta, *static_params)
  return T_opt, (T_opt, theta)


# Compute backward pass using intermediate
# values from the forward pass
def f_bwd(
  *static_params,
  fwd_res: tuple[Array, Array],
  cotangent: Array,
) -> tuple[Array]:
  # Convergence should not depend on static
  # parameters, so we can ignore them here
  del static_params
  # Output of the forward pass
  T_opt, theta = fwd_res

  # Gradient w.r.t. T
  def grad_T(T: Array, theta: Array) -> Array:
    return jax.grad(L)(T, theta)

  # Hessian A at (T_opt, theta)
  v = cotangent
  A = jax.hessian(L)(T_opt, theta)
  # Hessian is a positive semi-definite matrix
  u = jax.scipy.linalg.solve(
    A, -v, assume_a="pos"
  )
  # v^T dT*/dtheta = u^T ∂(grad_T L)/∂theta
  _, vjp_fun_theta = jax.vjp(
    lambda theta: grad_T(T_opt, theta), theta
  )
  # Compute gradient w.r.t. theta
  return vjp_fun_theta(u)


# Register custom VJP for f
f.defvjp(f_fwd, f_bwd)
