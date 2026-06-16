# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "jax>=0.9.2",
# ]
# ///
import jax

x = ...
a = ...
b = ...

if x < 0:
  x = a
else:
  x = b

x = (
  jax.lax.cond(
    x < 0,
    lambda: a,
    lambda: b,
  )
)


def f(i):
  pass


c = 0.0
for i in range(
  10
):
  c += f(i)


c = jax.lax.scan(
  lambda i, c: (
    c + f(i)
  ),
  0.0,  # Carry
  length=10,
)
