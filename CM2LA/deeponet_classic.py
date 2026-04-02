import os
import yaml
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
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

# ── Hyperparameters ────────────────────────────────────────────────────────────
seed = config.get('seed', 42)
num_bases = config.get('num_bases', 10)
num_trunk_epochs = config.get('num_trunk_epochs', 6000)
lr_start = config.get('lr_start', 1e-3)
lr_transition_steps = config.get('lr_transition_steps', 1000)
lr_decay_rate = config.get('lr_decay_rate', 0.01)
lr_end = config.get('lr_end', 1e-5)
num_LBFGS_epochs = config.get('num_LBFGS_epochs', 1000)
eps = config.get('eps', 1e-8)

# ── Paths ──────────────────────────────────────────────────────────────────────
dataset_path = Path(config.get('dataset_path', '/Users/haydenoutlaw/Documents/Research/opnet/deep-operator-networks/data/burgers_dataset.npz'))
save_path = Path(config.get('save_path', 'factors.npz'))

# ── Network Architectures ──────────────────────────────────────────────────────
# trunk: maps (t, x) -> R^K
# branch: maps u -> R^K  (time-independent)
trunk_arch = config.get('trunk_arch', [2, 40, 40, 10])
branch_arch = config.get('branch_arch', [2, 60, 60, 10])
assert trunk_arch[-1] == branch_arch[-1], "trunk and branch output dims must match"

# ── Training Parameters ────────────────────────────────────────────────────────
test_size = config.get('test_size', 0.2)
log_frequency = config.get('log_frequency', 10)
lbfgs_log_frequency = config.get('lbfgs_log_frequency', 100)

# ── Random Seeds ───────────────────────────────────────────────────────────────
np.random.seed(seed)
key = jax.random.key(seed)

print("\nconfiguring backend...")

jax.config.update("jax_platform_name", "gpu")

print("backend selected:\n", jax.default_backend())
print("active devices:\n", jax.devices())
print("--------------------\n")


# ── Model Definitions ──────────────────────────────────────────────────────────

class Linear(eqx.Module):
    """Fully connected linear layer: y = Wx + b."""

    weight: jax.Array
    bias: jax.Array

    def __init__(self, in_size, out_size, key, initializer=he_normal()):
        """
        Args:
            in_size: input feature dimension
            out_size: output feature dimension
            key: JAX PRNG key
            initializer: weight initializer (default: He normal)
        """
        wkey, bkey = jax.random.split(key)
        self.weight = initializer(wkey, (out_size, in_size), dtype=jnp.float32)
        self.bias = jnp.zeros((out_size,), dtype=jnp.float32)

    def __call__(self, x):
        return self.weight @ x + self.bias


class MLP(eqx.Module):
    """Fully connected multilayer perceptron with configurable depth and activation."""

    layers: list
    activations: list

    def __init__(self, architecture, key, activation=jax.nn.gelu, initializer=he_normal()):
        """
        Args:
            architecture: list of layer widths [in, h1, h2, ..., out]
            key: JAX PRNG key
            activation: activation function applied after each hidden layer
            initializer: weight initializer (default: He normal)
        """
        keys = jax.random.split(key, len(architecture) - 1)
        self.layers = [
            Linear(architecture[i], architecture[i+1], keys[i], initializer=initializer)
            for i in range(len(architecture) - 1)
        ]
        # no activation on the final layer
        self.activations = [activation] * (len(self.layers) - 1) + [eqx.nn.Identity()]

    def __call__(self, x):
        for layer, act in zip(self.layers, self.activations):
            x = act(layer(x))
        return x


# ── Dataset Loading ────────────────────────────────────────────────────────────
# Expected dataset layout:
#   t: (T,)       time grid
#   x: (X,)       spatial grid
#   samples[i]:   dict with 'params' (u, shape (2,)) and 'solution' (shape (T, X))

dataset = np.load(dataset_path, allow_pickle=True)
t_grid = jnp.array(dataset['t'])
x_grid = jnp.array(dataset['x'])

data = dataset['samples']
u = np.array([i['params'] for i in data])
s = np.array([i['solution'] for i in data])  # (N, T, X)

n_samp = len(data)
train_indices, test_indices = train_test_split(np.arange(n_samp), test_size=test_size, random_state=seed)
u_train, u_test = jnp.array(u[train_indices]), jnp.array(u[test_indices])
s_train, s_test = jnp.array(s[train_indices]), jnp.array(s[test_indices])

# transpose to (T, X, N) for use in loss_fn
output_tr   = jnp.transpose(s_train, axes=(1, 2, 0))
output_test = jnp.transpose(s_test,  axes=(1, 2, 0))


# ── Coordinate Grid ────────────────────────────────────────────────────────────
# Build a flat (T*X, 2) grid of (t, x) query points for the trunk network
tt, xx = jnp.meshgrid(t_grid, x_grid, indexing="ij")  # (T, X) each
tx_grid = jnp.stack([tt.reshape(-1), xx.reshape(-1)], axis=1)  # (T*X, 2)

u_dim = 2  # number of PDE input parameters
y_dim = 2  # query point dimension (t, x)


# ── Model Initialization ───────────────────────────────────────────────────────
key, subkey_t, subkey_b = jax.random.split(key, num=3)
trunk_model  = MLP(trunk_arch,  key=subkey_t)
branch_model = MLP(branch_arch, key=subkey_b)

model      = trunk_model, branch_model
best_model = trunk_model, branch_model


# ── Loss Functions ─────────────────────────────────────────────────────────────

def loss_fn(model, inputs, labels):
    """
    MSE loss over the full (T, X, N) prediction grid.

    The prediction is formed as:
        pred[t, x, n] = sum_k T[t*X+x, k] * B[n, k]   (matrix product T @ B^T)

    Args:
        model: (trunk_model, branch_model) tuple
        inputs: (N, 2) array of u input parameters
        labels: (T, X, N) ground-truth solutions

    Returns:
        scalar MSE
    """
    trunk_model, branch_model = model
    T_MAT    = jax.vmap(trunk_model)(tx_grid)       # (T*X, K)
    B_MAT    = jax.vmap(branch_model)(inputs)        # (N, K)
    pred_y   = T_MAT @ B_MAT.T                       # (T*X, N)
    pred_y_3d = pred_y.reshape(len(t_grid), len(x_grid), -1)  # (T, X, N)
    diff = labels - pred_y_3d
    return jnp.mean(diff ** 2)  # MSE


def l2_fn(model, inputs, labels):
    """
    Mean absolute L2 norm over samples (diagnostic only, not used in training).

    Args:
        model: (trunk_model, branch_model) tuple
        inputs: (N, 2) array of u input parameters
        labels: (T, X, N) ground-truth solutions

    Returns:
        scalar mean L2 error over N
    """
    trunk_model, branch_model = model
    T_MAT    = jax.vmap(trunk_model)(tx_grid)       # (T*X, K)
    B_MAT    = jax.vmap(branch_model)(inputs)        # (N, K)
    pred_y   = T_MAT @ B_MAT.T                       # (T*X, N)
    pred_y_3d = pred_y.reshape(len(t_grid), len(x_grid), -1)  # (T, X, N)
    diff = labels - pred_y_3d
    return jnp.mean(jnp.linalg.norm(diff, axis=(0, 1)))  # absolute mean L2 over N


# ── Adam Optimizer ─────────────────────────────────────────────────────────────
lr_schedule = optax.schedules.exponential_decay(
    init_value=lr_start,
    transition_steps=lr_transition_steps,
    decay_rate=lr_decay_rate,
    end_value=lr_end if lr_end is not None else None
)

opt = optax.adam(lr_schedule)
opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

loss_hist     = []
min_loss_hist = [np.inf]
l2_hist       = []  # absolute L2 trajectory, tracked for diagnostics only


@eqx.filter_jit
def train_step(model, opt_state):
    """Single Adam gradient update step."""
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, u_train, output_tr)
    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


# ── Adam Training Loop ─────────────────────────────────────────────────────────
s_time     = time.time()
start_time = time.time()

for step in range(num_trunk_epochs):
    model, opt_state, loss = train_step(model, opt_state)
    loss_hist.append(float(loss))
    l2_hist.append(float(l2_fn(model, u_train, output_tr)))  # diagnostic, not used in training

    # track best model by minimum training loss
    if float(loss) < min_loss_hist[-1]:
        min_loss_hist.append(float(loss))
        best_model = model
    else:
        min_loss_hist.append(min_loss_hist[-1])

    if step % log_frequency == 0:
        end_time = time.time()
        print(f"\rAdam step {step}: loss={float(loss):.3e}, {end_time-start_time:.2f}s", end="", flush=True)
        start_time = time.time()

e_time = time.time()
print(f"\nfinal adam loss: {loss:.3e}, total time: {e_time-s_time:.2f}s\n")


# ── LBFGS Fine-tuning ──────────────────────────────────────────────────────────
params = eqx.filter(model, eqx.is_inexact_array)
static = jax.tree_util.tree_map(lambda x: None if eqx.is_inexact_array(x) else x, model)

frozen_loss = jax.jit(lambda p: loss_fn(eqx.combine(p, static), u_train, output_tr))
frozen_l2   = jax.jit(lambda p: l2_fn(eqx.combine(p, static), u_train, output_tr))

# warm-up compile
_ = frozen_loss(params)
jax.block_until_ready(_)
print("frozen_loss compiled")

lbfgs_solver = LBFGS(fun=frozen_loss, maxiter=num_LBFGS_epochs, tol=1e-9,
                     history_size=20, implicit_diff=True, stepsize=-1.0)

s_time = time.time()
print(f"LBFGS start loss: {float(frozen_loss(params)):.3e}")
params, lbfgs_state = lbfgs_solver.run(params)
e_time = time.time()
print(f"LBFGS final loss: {float(lbfgs_state.value):.3e}, total time: {e_time-s_time:.2f}s\n")

# record LBFGS endpoint in histories and restore model
loss_hist.append(float(lbfgs_state.value))
min_loss_hist.append(min(min_loss_hist[-1], float(lbfgs_state.value)))
l2_hist.append(float(frozen_l2(params)))
model      = eqx.combine(params, static)
best_model = model

# ── Training Curves ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(loss_hist)
axes[0].plot(min_loss_hist[1:])
axes[0].set_title("Model Training History, MSE")
axes[0].set_yscale('log')
axes[0].set_xlabel("step")
axes[0].set_ylabel("MSE")
axes[1].plot(l2_hist)
axes[1].set_title("Model Training History, Absolute L2")
axes[1].set_yscale('log')
axes[1].set_xlabel("step")
axes[1].set_ylabel("Absolute L2")
plt.tight_layout()
plt.savefig('classic_training_history.png')
plt.close()


# ── Model Export ───────────────────────────────────────────────────────────────
model_path = Path(config.get("model_path", "classic_model.eqx"))
meta_path  = Path(config.get("meta_path",  "classic_model_meta.npz"))

eqx.tree_serialise_leaves(model_path, model)

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


# ── Inference Utilities ────────────────────────────────────────────────────────

def compute_THb_grid(trunk_model, branch_model, u, t_grid, x_grid):
    """
    Evaluate the DeepONet prediction on the full (t, x) grid for a single input u.

    Args:
        trunk_model: trained trunk MLP, maps (t, x) -> R^K
        branch_model: trained branch MLP, maps u -> R^K
        u: input parameters, shape (2,)
        t_grid: time coordinates, shape (T,)
        x_grid: spatial coordinates, shape (X,)

    Returns:
        pred_grid: predicted solution, shape (T, X)
    """
    tt, xx = jnp.meshgrid(t_grid, x_grid, indexing='ij')
    tx_points = jnp.stack([tt.flatten(), xx.flatten()], axis=1)  # (T*X, 2)
    T_flat    = jax.vmap(trunk_model)(tx_points)                  # (T*X, K)
    b         = branch_model(u)                                   # (K,)
    pred_flat = T_flat @ b                                        # (T*X,)
    return pred_flat.reshape(len(t_grid), len(x_grid))            # (T, X)


def predict_grid_for_u(trunk_model, branch_model, u, tx_grid, t_grid, x_grid):
    """
    Evaluate the DeepONet prediction on a pre-built flat tx_grid for a single u.
    Slightly more efficient than compute_THb_grid when tx_grid is already available.

    Args:
        trunk_model: trained trunk MLP
        branch_model: trained branch MLP
        u: input parameters, shape (2,)
        tx_grid: flat query points, shape (T*X, 2)
        t_grid: time coordinates, shape (T,)
        x_grid: spatial coordinates, shape (X,)

    Returns:
        predicted solution, shape (T, X)
    """
    T_MAT     = jax.vmap(trunk_model)(tx_grid)           # (T*X, K)
    b         = branch_model(u)                           # (K,)
    pred_flat = T_MAT @ b                                 # (T*X,)
    return pred_flat.reshape(len(t_grid), len(x_grid))    # (T, X)


def mse_vs_time_classic(model, u_test, s_test, tx_grid, t_grid, x_grid):
    """
    Compute MSE as a function of time, averaged over the test set.

    Args:
        model: (trunk_model, branch_model) tuple
        u_test: test input parameters, shape (N_test, 2)
        s_test: ground-truth test solutions, shape (N_test, T, X)
        tx_grid: flat query grid, shape (T*X, 2)
        t_grid: time coordinates, shape (T,)
        x_grid: spatial coordinates, shape (X,)

    Returns:
        mse_curve: MSE at each time step, shape (T,)
    """
    trunk_model, branch_model = model

    preds = jax.vmap(
        lambda u: predict_grid_for_u(trunk_model, branch_model, u, tx_grid, t_grid, x_grid)
    )(u_test)  # (N, T, X)

    diff = preds - s_test  # (N, T, X)

    # MSE over x, then mean over N -> (T,)
    mse_curve = jnp.mean(diff ** 2, axis=(0, 2))
    return mse_curve


def test_mse_classic(model, u_test, s_test, tx_grid, t_grid, x_grid):
    """
    Compute the mean MSE over the full test set (scalar summary).

    Args:
        model: (trunk_model, branch_model) tuple
        u_test: test input parameters, shape (N_test, 2)
        s_test: ground-truth test solutions, shape (N_test, T, X)
        tx_grid: flat query grid, shape (T*X, 2)
        t_grid: time coordinates, shape (T,)
        x_grid: spatial coordinates, shape (X,)

    Returns:
        scalar MSE
    """
    trunk_model, branch_model = model
    preds = jax.vmap(
        lambda u: predict_grid_for_u(trunk_model, branch_model, u, tx_grid, t_grid, x_grid)
    )(u_test)
    diff = preds - s_test
    return jnp.mean(diff ** 2)


# ── Qualitative Prediction Plots ───────────────────────────────────────────────
# Pick a single test sample for visual inspection
u_rand, s_rand = u_test[20], s_test[20]

trunk_model_final, branch_model_final = model
pred_grid = compute_THb_grid(trunk_model_final, branch_model_final, u_rand, t_grid, x_grid)

# Predicted solution
plt.figure(figsize=(7, 4))
plt.imshow(pred_grid, extent=[x_grid[0], x_grid[-1], t_grid[-1], t_grid[0]], aspect="auto", origin="upper")
plt.colorbar(label=r"$T(t,x)^* b_{\mathrm{NN}}(t)$")
plt.xlabel("x"); plt.ylabel("t")
plt.title("Grid evaluation of $T^*(t,x)b_{NN}$")
plt.tight_layout()
plt.savefig('classic_prediction_grid.png')
plt.close()

# Ground-truth solution
plt.figure(figsize=(7, 4))
plt.imshow(s_rand, extent=[x_grid[0], x_grid[-1], t_grid[-1], t_grid[0]], aspect="auto", origin="upper")
plt.colorbar(label=r"$s$")
plt.xlabel("x"); plt.ylabel("t")
plt.title("Grid evaluation of $s$")
plt.tight_layout()
plt.savefig('classic_true_solution_grid.png')
plt.close()

# Pointwise error
plt.figure(figsize=(7, 4))
plt.imshow(pred_grid - s_rand, extent=[x_grid[0], x_grid[-1], t_grid[-1], t_grid[0]], aspect="auto", origin="upper")
plt.colorbar(label=r"$Q(t,x)^* b_{\mathrm{NN}}(t) - s(x,t)$")
plt.xlabel("x"); plt.ylabel("t")
plt.title("Evaluation Error ($Q^*(t,x)b_{NN}(t) - s(x,t)$)")
plt.tight_layout()
plt.savefig('classic_error_grid.png')
plt.close()


# ── Trunk Basis Visualization ──────────────────────────────────────────────────
# Plot each learned trunk basis function T_k(t, x) as a 2D contour
trunk_model, branch_model = model
T_MAT = jax.vmap(trunk_model)(tx_grid)  # (T*X, K)

ncols = 4
nrows = int(np.ceil(num_bases / ncols))
T, X  = np.meshgrid(t_grid, x_grid, indexing="ij")

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), constrained_layout=True)
axes = axes.flatten()

for k, ax in enumerate(axes):
    if k < num_bases:
        T_k = T_MAT[:, k].reshape(len(t_grid), len(x_grid))
        im  = ax.contourf(T, X, T_k, levels=100, cmap='viridis')
        ax.set_title(f"T[{k}]")
        ax.set_xlabel("t")
        ax.set_ylabel("x")
    else:
        ax.axis("off")

fig.suptitle("Trunk Bases Across Time, Classic", fontsize=16)
cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.75)
plt.savefig("classic_bases.png")
plt.close()


# ── Quantitative Evaluation ────────────────────────────────────────────────────
final_test_mse = test_mse_classic(model, u_test, s_test, tx_grid, t_grid, x_grid)

print("\n==============================")
print(f" Test MSE (T(t,x) @ b(u)): {float(final_test_mse):.6e}")
print("==============================\n")

# MSE as a function of time
mse_curve = mse_vs_time_classic(model, u_test, s_test, tx_grid, t_grid, x_grid)

plt.figure()
plt.plot(np.asarray(t_grid), np.asarray(jax.device_get(mse_curve)))
plt.xlabel("t")
plt.ylabel("Mean MSE over x")
plt.title("Test Mean MSE vs time")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("classic_mse_vs_time.png")
plt.close()

print("saved: classic_mse_vs_time.png")


# ── Total Runtime ──────────────────────────────────────────────────────────────
master_end_time = time.time()
print("Total Time (sec):", master_end_time - master_start_time)