
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


master_start_time = time.time()

seed = config.get('seed', 42)
num_bases = config.get('num_bases', 10)
num_trunk_epochs = config.get('num_trunk_epochs', 6000)
lr_start = config.get('lr_start', 1e-3)
lr_transition_steps = config.get('lr_transition_steps', 1000)
lr_decay_rate = config.get('lr_decay_rate', 0.01)
lr_end = config.get('lr_end', 1e-5)
num_LBFGS_epochs = config.get('num_LBFGS_epochs', 1000)
eps = config.get('eps', 1e-8)

# Dataset paths
dataset_path = Path(config.get('dataset_path', '/Users/haydenoutlaw/Documents/Research/opnet/deep-operator-networks/data/burgers_dataset.npz'))
save_path = Path(config.get('save_path', 'factors.npz'))

# Network architectures
trunk_arch = config.get('trunk_arch', [2, 40, 40, 10])
branch_arch = config.get('branch_arch', [3, 60, 60, 10]) 
branch_arch[0] = branch_arch[0] + 1 ## modify branch to be time dependent
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

# u_train: (N, 2),  t_grid: (T,)
# branch_inputs_train: (N, T, 3)
u_broadcast = jnp.broadcast_to(u_train[:, None, :], (len(train_indices), len(t_grid), 2))
t_broadcast = jnp.broadcast_to(t_grid[None, :, None], (len(train_indices), len(t_grid), 1))
branch_inputs_train = jnp.concatenate([u_broadcast, t_broadcast], axis=-1) ## concatenating u and t vectors in new axis

# same for test
u_broadcast_test = jnp.broadcast_to(u_test[:, None, :], (len(test_indices), len(t_grid), 2))
t_broadcast_test = jnp.broadcast_to(t_grid[None, :, None], (len(test_indices), len(t_grid), 1))
branch_inputs_test = jnp.concatenate([u_broadcast_test, t_broadcast_test], axis=-1)


u_dim = 2 # in this case, two 
y_dim = 2 # (t,x)

# define architectures for component networks

key, subkey_t, subkey_b = jax.random.split(key, num=3)
trunk_model = MLP(trunk_arch, key = subkey_t)
branch_model = MLP(branch_arch, key=subkey_b)


model = trunk_model, branch_model
best_model = trunk_model, branch_model


def loss_fn(model, branch_inputs, labels):
    trunk_model, branch_model = model
    T_MAT = jax.vmap(trunk_model)(tx_grid)
    T_3d  = T_MAT.reshape(len(t_grid), len(x_grid), num_bases)
    B_MAT = jax.vmap(jax.vmap(branch_model))(branch_inputs)
    pred  = jnp.einsum('txk,ntk->ntx', T_3d, B_MAT)
    diff  = pred - labels
    return jnp.mean(jnp.linalg.norm(diff.reshape(diff.shape[0], -1), axis=-1) /
                    (jnp.linalg.norm(labels.reshape(labels.shape[0], -1), axis=-1) + eps))


lr_schedule = optax.schedules.exponential_decay(
    init_value=lr_start,
    transition_steps=lr_transition_steps,
    decay_rate=lr_decay_rate,
    end_value=lr_end if lr_end is not None else None
)

opt = optax.adam(lr_schedule)
opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

loss_hist = []
min_loss_hist = [np.inf]
# standard eqx training step
@eqx.filter_jit
def train_step(model, opt_state, branch_inputs, labels):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, branch_inputs, labels)
    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

s_time = time.time()
start_time = time.time()
for step in range(num_trunk_epochs):
    model, opt_state, loss = train_step(model, opt_state, branch_inputs_train, s_train)
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

opt_lbfgs = optax.lbfgs()
opt_state_lbfgs = opt_lbfgs.init(eqx.filter(model, eqx.is_inexact_array))

@eqx.filter_jit
def train_step_lbfgs(model, opt_state, branch_inputs, labels):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, branch_inputs, labels)
    updates, opt_state = opt_lbfgs.update(grads, opt_state, model, value=loss, grad=grads, value_fn=lambda m: loss_fn(m, branch_inputs, labels))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

s_time_lbfgs = time.time()
start_time = time.time()
for step in range(num_LBFGS_epochs):
    model, opt_state_lbfgs, loss = train_step_lbfgs(model, opt_state_lbfgs, branch_inputs_train, s_train)
    if float(loss) < min_loss_hist[-1]:
        min_loss_hist.append(float(loss))
        best_model = model
    else:
        min_loss_hist.append(min_loss_hist[-1])
    if step % lbfgs_log_frequency == 0:
        end_time = time.time()
        print(f"\rLBFGS step {step}: loss={float(loss):.3e}, {end_time-start_time:.2f}s", end="", flush=True)
        start_time = time.time()
e_time_lbfgs = time.time()
print(f"\nfinal LBFGS loss: {loss:.3e}, total time: {e_time_lbfgs-s_time_lbfgs:.2f}s\n")

plt.plot(loss_hist)
plt.plot(min_loss_hist[1:])

plt.title(f"Model Training History, Rel L2, Time Dep Classic")
plt.yscale('log')
plt.savefig('classic_td_training_history.png')
plt.close()

# ## save out models factors and data

# export data to .npz arrays
model_path = Path(config.get("model_path", "classic_td_model.eqx"))
meta_path  = Path(config.get("meta_path", "classic_td_model_meta.npz"))

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
    tt, xx = jnp.meshgrid(t_grid, x_grid, indexing='ij')
    tx_points = jnp.stack([tt.flatten(), xx.flatten()], axis=1)   # (T*X, 2)
    T_flat = jax.vmap(trunk_model)(tx_points)                      # (T*X, K)
    T_3d   = T_flat.reshape(len(t_grid), len(x_grid), -1)          # (T, X, K)

    # branch input for each timestep: (T, 3)
    u_rep  = jnp.broadcast_to(u[None, :], (len(t_grid), len(u)))
    t_col  = t_grid[:, None]
    branch_inputs = jnp.concatenate([u_rep, t_col], axis=-1)       # (T, 3)
    B_t    = jax.vmap(branch_model)(branch_inputs)                  # (T, K)

    pred   = jnp.einsum('txk,tk->tx', T_3d, B_t)                   # (T, X)
    return pred
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
plt.title("Grid evaluation of $T^*(t,x)b_{NN}(u,t)$, Time Dep Classic")
plt.tight_layout()
plt.savefig('classic_td_prediction_grid.png')
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
plt.title("Grid evaluation of $s$, Time Dep Classic")
plt.tight_layout()
plt.savefig('classic_td_true_solution_grid.png')
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
plt.title("Evaluation Error ($T^*(t,x)b_{NN}(u,t) - s(x,t)$), Time Dep Classic")
plt.tight_layout()
plt.savefig('classic_td_error_grid.png')
plt.close()



trunk_model_final, branch_model_final = model

T_MAT = jax.vmap(trunk_model)(tx_grid)   # (T*X, K)


ncols = 4
nrows = int(np.ceil(num_bases / ncols))

T, X = np.meshgrid(t_grid, x_grid, indexing="ij")

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(5 * ncols, 5 * nrows),
    constrained_layout=True
)

axes = axes.flatten()

for k, ax in enumerate(axes):
    if k < num_bases:
        T_k = T_MAT[:, k].reshape(len(t_grid), len(x_grid))
        # 2D top-down plot
        im = ax.contourf(T, X, T_k, levels=100, cmap='viridis')

        ax.set_title(f"T[{k}]")
        ax.set_xlabel("t")
        ax.set_ylabel("x")
    else:
        ax.axis("off")

fig.suptitle(
    f"Trunk Bases Across Time, Time Dep Classic",
    fontsize=16
)

# attach a single colorbar for the whole figure
cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.75)

plt.savefig("classic_td_bases.png")
plt.close()



def predict_grid_for_u(trunk_model, branch_model, u, tx_grid, t_grid, x_grid):
    T_MAT = jax.vmap(trunk_model)(tx_grid)                      # (T*X, K)
    T_3d  = T_MAT.reshape(len(t_grid), len(x_grid), -1)         # (T, X, K)
    u_rep = jnp.broadcast_to(u[None, :], (len(t_grid), len(u)))
    t_col = t_grid[:, None]
    branch_inputs = jnp.concatenate([u_rep, t_col], axis=-1)    # (T, 3)
    B_t   = jax.vmap(branch_model)(branch_inputs)               # (T, K)
    return jnp.einsum('txk,tk->tx', T_3d, B_t)                  # (T, X)


def rel_l2_error_vs_time_classic(model, u_test, s_test, tx_grid, t_grid, x_grid, eps=1e-12):
    """
    Mean over test samples of ( ||pred-s||_2 / ||s||_2 ) computed at each time.
    Returns rel_curve of shape (T,).
    """
    trunk_model, branch_model = model

    # (N_test, T, X)
    preds = jax.vmap(
        lambda u: predict_grid_for_u(trunk_model, branch_model, u, tx_grid, t_grid, x_grid)
    )(u_test)

    diff = preds - s_test  # (N, T, X)

    # L2 over space (X) -> (N, T)
    num = jnp.linalg.norm(diff, axis=-1)
    den = jnp.linalg.norm(s_test, axis=-1)

    rel = num / (den + eps)          # (N, T)
    rel_curve = jnp.linalg.norm(diff, axis=(0, 2)) / (jnp.linalg.norm(s_test, axis=(0, 2)) + eps)

    return rel_curve


def test_rel_l2_classic(model, u_test, s_test, tx_grid, t_grid, x_grid):
    trunk_model, branch_model = model
    preds = jax.vmap(lambda u: predict_grid_for_u(trunk_model, branch_model, u, tx_grid, t_grid, x_grid))(u_test)
    diff  = preds - s_test
    return jnp.mean(jnp.linalg.norm(diff.reshape(diff.shape[0], -1), axis=-1) /
                    (jnp.linalg.norm(s_test.reshape(s_test.shape[0], -1), axis=-1) + eps))

final_test_rel_l2 = test_rel_l2_classic(model, u_test, s_test, tx_grid, t_grid, x_grid)

print("\n==============================")
print(f"Average Test Rel L2 (T(t,x) @ b(u,t)): {float(final_test_rel_l2):.6e}")
print("==============================\n")


# ---- compute + plot the curve ----
rel_curve = rel_l2_error_vs_time_classic(model, u_test, s_test, tx_grid, t_grid, x_grid)

plt.figure()
plt.plot(np.asarray(t_grid), np.asarray(jax.device_get(rel_curve)))
plt.xlabel("t")
plt.ylabel("mean relative L2 error over x")
plt.title("Test relative L2 error vs time, Time Dep Classic")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("classic_td_rel_l2_vs_time.png")
plt.close()

print("saved: classic_td_rel_l2_vs_time.png")



master_end_time = time.time()

print("Total Time (sec):", master_end_time - master_start_time)