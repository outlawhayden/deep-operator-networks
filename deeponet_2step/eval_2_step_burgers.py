import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import matplotlib.pyplot as plt


# === Reload model ===
# You must define the architecture the same way as during training
u_dim = 2        # same as training
y_dim = 2        # (x, t)
num_branches = 40
branch_arch = [u_dim, 40, 40, num_branches]
trunk_arch  = [y_dim, 40, 40, num_branches]

class Linear(eqx.Module):
    weight: jax.Array
    bias: jax.Array
    def __init__(self, in_size, out_size, key, initializer=jax.nn.initializers.he_normal()):
        wkey, bkey = jax.random.split(key)
        self.weight = initializer(wkey, (out_size, in_size), dtype=jnp.float32)
        self.bias = jnp.zeros((out_size,), dtype=jnp.float32)
    def __call__(self, x):
        return self.weight @ x + self.bias

class MLP(eqx.Module):
    layers: list
    activations: list
    def __init__(self, architecture, key, activation=jax.nn.relu, initializer=jax.nn.initializers.he_normal()):
        keys = jax.random.split(key, len(architecture)-1)
        self.layers = [Linear(architecture[i], architecture[i+1], keys[i], initializer=initializer) 
                       for i in range(len(architecture)-1)]
        self.activations = [activation]*(len(self.layers)-1) + [eqx.nn.Identity()]
    def __call__(self, x):
        for layer, act in zip(self.layers, self.activations):
            x = act(layer(x))
        return x

class DeepONet(eqx.Module):
    branch: MLP
    trunk: MLP
    def __init__(self, branch_arch, trunk_arch, key, num_branches, activation=jax.nn.relu, initializer=jax.nn.initializers.he_normal()):
        bkey, tkey = jax.random.split(key)
        self.branch = MLP(branch_arch, bkey, activation, initializer=initializer)
        self.trunk = MLP(trunk_arch, tkey, activation, initializer=initializer)
    def __call__(self, x):
        u, y = x
        branch_out = self.branch(jnp.atleast_1d(u))
        trunk_out = self.trunk(jnp.atleast_1d(y))
        return jnp.inner(branch_out, trunk_out)


# Dummy init with same shape as training
key = jax.random.key(0)
model = DeepONet(branch_arch, trunk_arch, key, num_branches)
# Load trained weights
model = eqx.tree_deserialise_leaves("deeponet_burgers_model.eqx", model)


# === Load dataset ===
dataset = np.load("/Users/haydenoutlaw/Documents/Research/opnet/deep-operator-networks/data/burgers_dataset.npz", allow_pickle=True)
t_grid = jnp.array(dataset["t"])
x_grid = jnp.array(dataset["x"])
data   = dataset["samples"]

# Pick one example
example = data[0]
u_params = jnp.array(example["params"])    # branch input
true_sol = jnp.array(example["solution"])  # shape (nt, nx)

# Build meshgrid of (x,t) pairs for evaluation
xx, tt = jnp.meshgrid(x_grid, t_grid, indexing="ij")
y_pairs = jnp.stack([xx.ravel(), tt.ravel()], axis=1)   # (nx*nt, 2)

# Repeat u for each (x,t)
u_repeat = jnp.tile(u_params[None, :], (y_pairs.shape[0], 1))

# Run model
pred = jax.vmap(model)((u_repeat, y_pairs))
pred = pred.reshape(len(t_grid), len(x_grid))  # reshape to (nt, nx)

# === Compare prediction vs truth ===
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.title("True solution")
plt.pcolormesh(x_grid, t_grid, true_sol.T, shading="auto")
plt.colorbar()

plt.subplot(1,2,2)
plt.title("Predicted solution - 2 Step QR DeepONet ")
plt.pcolormesh(x_grid, t_grid, pred, shading="auto")
plt.colorbar()

plt.tight_layout()

plt.savefig("2_step_burger_soln.png", dpi = 300)

plt.show()

plt.figure(figsize = (8,5))
plt.title("Burger's 2 Step DeepONet Error (pred - true)")
plt.pcolormesh(x_grid, t_grid, pred - true_sol.T, shading = "auto")
plt.colorbar()
plt.tight_layout()

plt.savefig("2_step_burger_error.png", dpi = 300)

plt.show()