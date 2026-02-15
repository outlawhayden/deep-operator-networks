
import os
import yaml
from pathlib import Path
config_path = Path("config.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

gpu_idx = config.get("gpu_idx", 0)
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

import jax.numpy as jnp
import numpy as np
import jax
import optax
from jaxopt import LBFGS
import scipy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import equinox as eqx
import scipy.linalg
from jax.nn.initializers import he_normal
from matplotlib import cbook, cm
from matplotlib.colors import LightSource
import time




seed = config.get('seed', 42)
num_bases = config.get('num_bases', 10)
num_trunk_epochs = config.get('num_trunk_epochs', 6000)
lr = config.get('lr', 1e-3)
num_LBFGS_epochs = config.get('num_LBFGS_epochs', 1000)
eps = config.get('eps', 1e-8)

# Dataset paths
dataset_path = Path(config.get('dataset_path', '/Users/haydenoutlaw/Documents/Research/opnet/deep-operator-networks/data/burgers_dataset.npz'))
save_path = Path(config.get('save_path', 'factors.npz'))

# Network architectures
trunk_arch = config.get('trunk_arch', [2, 40, 40, 10])
branch_arch = config.get('branch_arch', [2, 60, 60, 10])
assert trunk_arch[-1] == branch_arch[-1]

# Training parameters
test_size = config.get('test_size', 0.2)
log_frequency = config.get('log_frequency', 10)
lbfgs_log_frequency = config.get('lbfgs_log_frequency', 100)

# Set random seeds
np.random.seed(seed)
key = jax.random.key(seed)

print("\nconfiguring backend...")

jax.config.update("jax_platform_name", "gpu")


print("backend selected:\n", jax.default_backend())
print("active devices:\n", jax.devices())
print("--------------------\n")

## EQUINOX CLASS DEFINITIONS
# Linear Model (Wx + b)
class Linear(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __init__(self, in_size, out_size, key, initializer = he_normal()):
        wkey, bkey = jax.random.split(key)
        self.weight = initializer(wkey, (out_size, in_size), dtype=jnp.float32)
        self.bias = jnp.zeros((out_size,), dtype=jnp.float32)

    def __call__(self, x):
        return self.weight @ x + self.bias


# Basic MultiLayer Perceptron Primitive
class MLP(eqx.Module):
    layers: list
    activations: list

    def __init__(self, architecture, key, activation = jax.nn.relu, initializer = he_normal()):
        """
        architecture: List[in, hidden1, hidden2, ..., out]
        key: random key
        activation: jax.nn act fxn
        """
        keys = jax.random.split(key, len(architecture) - 1)
        self.layers = [
            Linear(architecture[i], architecture[i+1], keys[i], initializer = initializer) for i in range(len(architecture) - 1)]
        self.activations = [activation] * (len(self.layers) - 1) + [eqx.nn.Identity()] # no activation on last layer

    def __call__(self, x):
        for layer, act in zip(self.layers, self.activations):
            x = act(layer(x))
        return x

# ## loading dataset

# load external dataset
dataset = np.load(dataset_path, allow_pickle = True)
t_grid = jnp.array(dataset['t'])
x_grid = jnp.array(dataset['x'])

data = dataset['samples']
u = np.array([i['params'] for i in data])
s = np.array([i['solution'] for i in data]) # s is shape (n,t,x)

n_samp = len(data)
train_indices, test_indices = train_test_split(np.arange(n_samp), test_size = test_size, random_state = seed)
u_train, u_test = jnp.array(u[train_indices]), jnp.array(u[test_indices])
s_train, s_test = jnp.array(s[train_indices]), jnp.array(s[test_indices])

output_tr = jnp.transpose(s_train, axes=(1,2,0)) 
output_test = jnp.transpose(s_test, axes = (1,2,0))


tt, xx = jnp.meshgrid(t_grid, x_grid, indexing="ij")  # tt,xx are (T,X)
tx_grid = jnp.stack([tt.reshape(-1), xx.reshape(-1)], axis=1)  # (T*X,2)


u_dim = 2 # in this case, two 
y_dim = 2 # (t,x)

# define architectures for component networks

key, subkey_t, subkey_b = jax.random.split(key, num=3)
trunk_model = MLP(trunk_arch, key = subkey_t)
branch_model = MLP(branch_arch, key=subkey_b)


model = trunk_model, branch_model
best_model = trunk_model, branch_model


def loss_fn(model, inputs, labels):
    trunk_model, branch_model = model
    T_MAT = jax.vmap(trunk_model)(tx_grid)
    B_MAT = jax.vmap(branch_model)(inputs)
    pred_y = T_MAT @ B_MAT.T
    # Reshape pred_y to match output_tr's 3D shape
    pred_y_3d = pred_y.reshape(len(t_grid), len(x_grid), -1)  # (T, X, N)
    return jnp.mean((labels - pred_y_3d) ** 2)  # mse

opt = optax.adam(lr)
opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

loss_hist = []
min_loss_hist = [np.inf]
# standard eqx training step
@eqx.filter_jit
def train_step(model, opt_state):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, u_train, output_tr)
    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

s_time = time.time()
start_time = time.time()
for step in range(num_trunk_epochs):
    model, opt_state, loss = train_step(model, opt_state)
    loss_hist.append(float(loss)) # store as list
    if float(loss) < min_loss_hist[-1]:
        min_loss_hist.append(float(loss))
        best_model = model
    else:
        min_loss_hist.append(min_loss_hist[-1])
    if step % log_frequency == 0: # new carriage print, avoid tqdm for memory
        end_time = time.time()
        print(f"\rAdam step {step}: loss={float(loss):.3e}, {end_time-start_time:.2f}s", end="", flush=True)
        start_time = time.time()
e_time = time.time()
print(f"\nfinal adam loss: {loss:.3e}, total time: {e_time-s_time:.2f}s\n")

model = best_model ## save out best version

# opt_lbfgs = optax.lbfgs()
# opt_state_lbfgs = opt_lbfgs.init(eqx.filter(model, eqx.is_inexact_array))

# @eqx.filter_jit
# def train_step_lbfgs(model, opt_state):
#     loss, grads = eqx.filter_value_and_grad(loss_fn)(model, u_train, output_tr)
#     updates, opt_state = opt_lbfgs.update(grads, opt_state, model)
#     model = eqx.apply_updates(model, updates)
#     return model, opt_state, loss


# for step in range(num_LBFGS_epochs):
#     model, opt_state, loss = train_step_lbfgs(model, opt_state_lbfgs)
#     loss_hist.append(float(loss))
#     test_loss = loss_fn(model, u_test, output_test)
#     test_loss_hist.append(float(test_loss))
#     if step % lbfgs_log_frequency == 0:
#         print(
#             f"\r{' ' * 80}\rLBFGS step {step}: loss={float(loss):.3e}, test loss={float(test_loss):.3e}", 
#             end="",
#             flush=True
#         )

plt.plot(loss_hist)
plt.plot(min_loss_hist[1:])

plt.title(f"Model Training History, MSE")
plt.yscale('log')
plt.savefig('classic_training_history.png')
plt.close()

# ## save out models factors and data

# export data to .npz arrays
model_path = Path(config.get("model_path", "classic_model.eqx"))
meta_path  = Path(config.get("meta_path", "classic_model_meta.npz"))

# Save the Equinox pytree (trunk_model, branch_model)
eqx.tree_serialise_leaves(model_path, model)

# Save lightweight metadata you may want later
np.savez_compressed(
    meta_path,
    t_grid=np.asarray(t_grid),
    x_grid=np.asarray(x_grid),
    trunk_arch=np.asarray(trunk_arch),
    branch_arch=np.asarray(branch_arch),
    seed=np.asarray(seed),
)

print("\nexported model to:", model_path)
print("exported metadata to:", meta_path)

u_rand, s_rand = u_test[20], s_test[20] # s is true values, u is input parameters

def compute_THb_grid(trunk_model, branch_model, u, t_grid, x_grid):
    """
    trunk_model: maps (t,x) -> R^K
    branch_model: maps u -> R^K  
    u: input parameters (2,)
    returns: (T, X) prediction grid
    """
    # Create all (t,x) points
    tt, xx = jnp.meshgrid(t_grid, x_grid, indexing='ij')
    tx_points = jnp.stack([tt.flatten(), xx.flatten()], axis=1)  # (T*X, 2)
    
    # Compute trunk outputs
    T_flat = jax.vmap(trunk_model)(tx_points)  # (T*X, K)
    
    # Compute branch output for this u
    b = branch_model(u)  # (K,)
    
    # Compute prediction: T_flat @ b = (T*X,)
    pred_flat = T_flat @ b
    
    # Reshape to grid
    pred_grid = pred_flat.reshape(len(t_grid), len(x_grid))
    
    return pred_grid

# Example usage
trunk_model_final, branch_model_final = model
pred_grid = compute_THb_grid(trunk_model_final, branch_model_final, u_rand, t_grid, x_grid)

plt.figure(figsize=(7, 4))
plt.imshow(
    pred_grid,
    extent=[x_grid[0], x_grid[-1], t_grid[-1], t_grid[0]],
    aspect="auto",
    origin="upper"
)
plt.colorbar(label=r"$T(t,x)^* b_{\mathrm{NN}}(t)$")
plt.xlabel("x")
plt.ylabel("t")
plt.title("Grid evaluation of $T^*(t,x)b_{NN}$")
plt.tight_layout()
plt.savefig('classic_prediction_grid.png')
plt.close()

plt.figure(figsize=(7, 4))
plt.imshow(
    s_rand,
    extent=[x_grid[0], x_grid[-1], t_grid[-1], t_grid[0]],
    aspect="auto",
    origin="upper"
)
plt.colorbar(label=r"$s$")
plt.xlabel("x")
plt.ylabel("t")
plt.title("Grid evaluation of $s$")
plt.tight_layout()
plt.savefig('classic_true_solution_grid.png')
plt.close()

plt.figure(figsize=(7, 4))
plt.imshow(
    pred_grid- s_rand,
    extent=[x_grid[0], x_grid[-1], t_grid[-1], t_grid[0]],
    aspect="auto",
    origin="upper"
)
plt.colorbar(label=r"$Q(t,x)^* b_{\mathrm{NN}}(t) - s(x,t)$")
plt.xlabel("x")
plt.ylabel("t")
plt.title("Evaluation Error ($Q^*(t,x)b_{NN}(t) - s(x,t)$)")
plt.tight_layout()
plt.savefig('classic_error_grid.png')
plt.close()



trunk_model_final, branch_model_final = model

# Compute test loss using the same loss function
test_mse = loss_fn(model, u_test, output_test)

print("\n==============================")
print(f"Average Test MSE: {float(test_mse):.6e}")
print("==============================\n")