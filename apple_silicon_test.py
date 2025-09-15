import jax 
import jax.numpy as jnp


jax.config.update("jax_platform_name", "METAL")
print(jax.default_backend())
print(jax.devices())


x = jnp.ones((100,100))
y = jnp.dot(x, x.T)

print(y.device)