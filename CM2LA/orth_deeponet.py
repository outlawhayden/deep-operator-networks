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
save_path = Path(config.get('save_path', 'orth_qr_factors.npz'))

# ── Network Architectures ──────────────────────────────────────────────────────
# trunk:  maps (t, x) -> R^K
# branch: maps u -> R^K  (time-independent)
# output dims are overridden by num_bases to stay consistent
trunk_arch = config.get('trunk_arch', [2, 40, 40, 10])
branch_arch = config.get('branch_arch', [2, 60, 60, 10])

trunk_arch[-1]  = num_bases
branch_arch[-1] = num_bases

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
            activation: activation function applied after each hidden layer (default: GELU)
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
tt, xx = jnp.meshgrid(t_grid, x_grid, indexing="ij")   # (T, X) each
tx_grid = jnp.stack([tt.reshape(-1), xx.reshape(-1)], axis=1)  # (T*X, 2)

u_dim = 2  # number of PDE input parameters
y_dim = 2  # query point dimension (t, x)


# ── Stage 1 Model Initialization ──────────────────────────────────────────────
# Stage 1 jointly trains the trunk MLP and a global coefficient matrix A_model.
# A_model[k, n] holds the expansion coefficient for basis k and sample n.
# After training, A_model is factored via QR to produce orthonormal trunk bases Q.

key, subkey_t, subkey_A = jax.random.split(key, num=3)
trunk_model = MLP(trunk_arch, key=subkey_t)
A_model = jax.random.normal(subkey_A, (num_bases, output_tr.shape[2]))  # (K, N)

model      = trunk_model, A_model
best_model = model


# ── Stage 1 Loss Functions ─────────────────────────────────────────────────────

def loss_fn(model, inputs, labels):
    """
    MSE loss for stage 1 (trunk + A_model joint training).

    Prediction: pred[t*X+x, n] = sum_k T[t*X+x, k] * A[k, n]  (i.e. T_MAT @ A_model)
    Reshaped to (T, X, N) before comparing to labels.

    Args:
        model: (trunk_model, A_model) tuple
        inputs: flat query grid, shape (T*X, 2)
        labels: ground-truth solutions, shape (T, X, N)

    Returns:
        scalar MSE
    """
    trunk_model, A_model = model
    T_MAT     = jax.vmap(trunk_model)(inputs)                         # (T*X, K)
    pred_y    = T_MAT @ A_model                                       # (T*X, N)
    pred_y_3d = pred_y.reshape(len(t_grid), len(x_grid), -1)          # (T, X, N)
    diff      = labels - pred_y_3d
    return jnp.mean(diff ** 2)


def l2_fn(model, inputs, labels):
    """
    Mean absolute L2 norm over samples (diagnostic only, not used in training).

    Args:
        model: (trunk_model, A_model) tuple
        inputs: flat query grid, shape (T*X, 2)
        labels: ground-truth solutions, shape (T, X, N)

    Returns:
        scalar mean L2 error over N
    """
    trunk_model, A_model = model
    T_MAT     = jax.vmap(trunk_model)(inputs)               # (T*X, K)
    pred_y    = T_MAT @ A_model                             # (T*X, N)
    pred_y_3d = pred_y.reshape(len(t_grid), len(x_grid), -1)
    diff      = labels - pred_y_3d
    return jnp.mean(jnp.linalg.norm(diff.reshape(-1, diff.shape[2]), axis=0))


# ── Stage 1 Adam Optimizer ─────────────────────────────────────────────────────
lr_schedule = optax.schedules.exponential_decay(
    init_value=lr_start,
    transition_steps=lr_transition_steps,
    decay_rate=lr_decay_rate,
    end_value=lr_end if lr_end is not None else None
)

opt       = optax.adam(lr_schedule)
opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

loss_hist     = []
min_loss_hist = [np.inf]
l2_hist       = []  # absolute L2 trajectory, tracked for diagnostics only


@eqx.filter_jit
def train_step(model, opt_state):
    """Single Adam gradient update step for stage 1."""
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, tx_grid, output_tr)
    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


# ── Stage 1 Adam Training Loop ─────────────────────────────────────────────────
s_time     = time.time()
start_time = time.time()

for step in range(num_trunk_epochs):
    model, opt_state, loss = train_step(model, opt_state)
    loss_hist.append(float(loss))
    l2_hist.append(float(l2_fn(model, tx_grid, output_tr)))  # diagnostic, not used in training

    # track best model by minimum training loss
    if float(loss) < min_loss_hist[-1]:
        min_loss_hist.append(float(loss))
        best_model = model
    else:
        min_loss_hist.append(float(min_loss_hist[-1]))

    if step % 10000 == 0:
        end_time = time.time()
        print(f"\rAdam step {step}: loss={float(loss):.3e}, {end_time-start_time:.2f}s", end="", flush=True)
        start_time = time.time()

e_time = time.time()
print(f"\nfinal adam loss: {loss:.3e}, total time: {e_time-s_time:.2f}s\n")


# ── Stage 1 LBFGS Fine-tuning ──────────────────────────────────────────────────
# Freeze the model structure and run jaxopt LBFGS on the parameter leaves only.
params = eqx.filter(model, eqx.is_inexact_array)
static = jax.tree_util.tree_map(lambda x: None if eqx.is_inexact_array(x) else x, model)

frozen_loss = jax.jit(lambda p: loss_fn(eqx.combine(p, static), tx_grid, output_tr))
frozen_l2   = jax.jit(lambda p: l2_fn(eqx.combine(p, static), tx_grid, output_tr))

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


# ── Stage 1 Training Curves ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(loss_hist)
axes[0].plot(min_loss_hist[1:])
axes[0].set_title("Trunk Model Training History, MSE, 2-Step")
axes[0].set_yscale('log')
axes[0].set_xlabel("step")
axes[0].set_ylabel("MSE")
axes[1].plot(l2_hist)
axes[1].set_title("Trunk Model Training History, Absolute L2, 2-Step")
axes[1].set_yscale('log')
axes[1].set_xlabel("step")
axes[1].set_ylabel("Absolute L2")
plt.tight_layout()
plt.savefig('orth_training_history.png')
plt.close()


# ── QR Factorization of Trunk Bases ───────────────────────────────────────────
# Compute a single thin QR of the full trunk matrix T_MAT (T*X x K).
# Q_MAT (T*X x K) gives orthonormal spatial bases; R_MAT (K x K) upper-triangular.
# Unlike the time-dependent variant, no sign correction is needed here.

trunk_model, A_model = model
T_MAT      = jax.vmap(trunk_model)(tx_grid)          # (T*X, K)
T_MAT_host = np.asarray(T_MAT)

Q_MAT, R_MAT = scipy.linalg.qr(T_MAT_host, mode="economic")  # Q: (T*X, K), R: (K, K)

# convert back to JAX arrays for downstream matrix multiplications
Q_MAT = jnp.asarray(Q_MAT)
R_MAT = jnp.asarray(R_MAT)


# ── Stage 2: Branch Network Training ──────────────────────────────────────────
# The branch network learns to predict (R @ A)^T[n, :] from u_n.
# Target: RA_target[n, k] = (R @ A)[k, n]  transposed to (N, K) to match B_MAT.
# This decouples the branch from the trunk after stage 1.

K = A_model.shape[0]  # number of basis functions
N = A_model.shape[1]  # number of training samples

key, subkey_t, subkey_A = jax.random.split(key, num=3)
branch_model      = MLP(branch_arch, key=subkey_t)
best_branch_model = branch_model

# pre-compute the frozen regression target RA_target = (R @ A)^T
RA_model  = (R_MAT @ A_model).T   # (N, K)
RA_target = jnp.asarray(RA_model)  # frozen, not updated during branch training


# ── Stage 2 Loss Functions ─────────────────────────────────────────────────────

def branch_loss_fn(branch_model, RA_target, u_train, eps=1e-8):
    """
    MSE loss for stage 2 branch training.

    The branch maps u -> R^K. The target for each sample n is RA_target[n, :].

    Args:
        branch_model: branch MLP
        RA_target: frozen regression targets, shape (N, K)
        u_train: training input parameters, shape (N, 2)
        eps: unused stability constant (reserved)

    Returns:
        scalar MSE
    """
    B_MAT = jax.vmap(branch_model)(u_train)  # (N, K)
    diff  = B_MAT - RA_target
    return jnp.mean(diff ** 2)


def branch_l2_fn(branch_model, RA_target, u_train):
    """
    Mean absolute L2 norm over samples for branch (diagnostic only).

    Args:
        branch_model: branch MLP
        RA_target: frozen regression targets, shape (N, K)
        u_train: training input parameters, shape (N, 2)

    Returns:
        scalar mean L2 error over N
    """
    B_MAT = jax.vmap(branch_model)(u_train)  # (N, K)
    diff  = B_MAT - RA_target
    return jnp.mean(jnp.linalg.norm(diff, axis=1))


# ── Stage 2 Adam Optimizer ─────────────────────────────────────────────────────
@eqx.filter_jit
def branch_train_step(branch_model, opt_state, RA_target, u_train):
    """Single Adam gradient update step for the branch network."""
    loss, grads = eqx.filter_value_and_grad(branch_loss_fn)(
        branch_model, RA_target, u_train
    )
    updates, opt_state = opt.update(grads, opt_state, branch_model)
    branch_model = eqx.apply_updates(branch_model, updates)
    return branch_model, opt_state, loss


branch_loss_hist     = []
branch_min_loss_hist = [np.inf]
branch_l2_hist       = []  # absolute L2 trajectory, tracked for diagnostics only

# re-initialise optimizer for branch parameters only
opt       = optax.adam(lr_schedule)
opt_state = opt.init(eqx.filter(branch_model, eqx.is_inexact_array))


# ── Stage 2 Adam Training Loop ─────────────────────────────────────────────────
for step in range(num_trunk_epochs):
    branch_model, opt_state, loss = branch_train_step(
        branch_model, opt_state, RA_target, u_train
    )
    branch_loss_hist.append(float(loss))
    branch_l2_hist.append(float(branch_l2_fn(branch_model, RA_target, u_train)))  # diagnostic

    # track best branch model by minimum training loss
    if float(loss) < float(branch_min_loss_hist[-1]):
        branch_min_loss_hist.append(float(loss))
        best_branch_model = branch_model
    else:
        branch_min_loss_hist.append(float(branch_min_loss_hist[-1]))

    if step % 10000 == 0:
        end_time = time.time()
        print(f"\rAdam step {step}: loss={float(loss):.3e}", end="", flush=True)

print(f"\nfinal adam loss: {float(loss):.3e}\n")

branch_model = best_branch_model  # restore best checkpoint


# ── Stage 2 Training Curves ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(branch_loss_hist)
axes[0].plot(branch_min_loss_hist[1:])
axes[0].set_title("Branch Model Training History, MSE, 2-Step")
axes[0].set_yscale('log')
axes[0].set_xlabel("step")
axes[0].set_ylabel("MSE")
axes[1].plot(branch_l2_hist)
axes[1].set_title("Branch Model Training History, Absolute L2, 2-Step")
axes[1].set_yscale('log')
axes[1].set_xlabel("step")
axes[1].set_ylabel("Absolute L2")
plt.tight_layout()
plt.savefig("orth_branch_training_history.png")
plt.close()


# ── Trunk Basis Visualization ──────────────────────────────────────────────────
# Plot each learned trunk basis function T_k(t, x) as a 2D contour
T_MAT = jax.vmap(lambda t: jax.vmap(lambda x: trunk_model(jnp.stack([t, x])))(x_grid))(t_grid)  # (T, X, K)

ncols = 4
nrows = int(np.ceil(num_bases / ncols))
T, X  = np.meshgrid(t_grid, x_grid, indexing="ij")

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), constrained_layout=True)
axes = axes.flatten()

for k, ax in enumerate(axes):
    if k < num_bases:
        T_k = T_MAT[:, :, k]
        im  = ax.contourf(T, X, T_k, levels=100)
        ax.set_title(f"T[{k}]")
        ax.set_xlabel("t")
        ax.set_ylabel("x")
    else:
        ax.axis("off")

fig.suptitle("Trunk Bases Across Time, 2-Step", fontsize=16)
cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.75)
plt.savefig("orth_bases.png")
plt.close()


# ── Model Export ───────────────────────────────────────────────────────────────
trunkA_path = Path(config.get("trunkA_model_path", "orth_trunkA.eqx"))
branch_path = Path(config.get("branch_model_path", "orth_branch.eqx"))
meta_path   = Path(config.get("meta_path",          "orth_meta.npz"))
qr_path     = Path(config.get("qr_path",             "orth_qr_factors.npz"))

# trunk + A_model pytree (stage 1 result)
eqx.tree_serialise_leaves(trunkA_path, model)

# branch model pytree (stage 2 result)
eqx.tree_serialise_leaves(branch_path, branch_model)

# lightweight metadata needed to rebuild model templates on load
np.savez_compressed(
    meta_path,
    t_grid=np.asarray(t_grid),
    x_grid=np.asarray(x_grid),
    trunk_arch=np.asarray(trunk_arch),
    branch_arch=np.asarray(branch_arch),
    num_bases=np.asarray(num_bases),
    seed=np.asarray(seed),
)

# QR factors for downstream analysis / reconstruction
np.savez_compressed(qr_path, Q=np.asarray(Q_MAT), R=np.asarray(R_MAT))

print("\nexported trunk+A model to:", trunkA_path)
print("exported branch model to:", branch_path)
print("exported metadata to:", meta_path)
print("exported QR factors to:", qr_path)


# ── Inference Utilities ────────────────────────────────────────────────────────

def compute_QHb_grid(Q_MAT, branch_model, u, t_grid):
    """
    Evaluate the DeepONet prediction on the full (T, X) grid for a single input u.

    Prediction: pred[t, x] = sum_k Q[t*X+x, k] * b[k]  (i.e. Q_MAT @ b(u), reshaped)

    Args:
        Q_MAT: orthonormal trunk bases, shape (T*X, K)
        branch_model: trained branch MLP
        u: input parameters, shape (u_dim,)
        t_grid: time coordinates, shape (T,)

    Returns:
        pred_grid: predicted solution, shape (T, X)
    """
    b         = branch_model(u)                          # (K,)
    pred_flat = Q_MAT @ b                                # (T*X,)
    return pred_flat.reshape(len(t_grid), len(x_grid))   # (T, X)


def predict_solution_from_Q(Q_MAT, branch_model, u, t_grid, x_grid):
    """
    Alias for compute_QHb_grid; used in batched evaluation helpers below.

    Args:
        Q_MAT: orthonormal trunk bases, shape (T*X, K)
        branch_model: trained branch MLP
        u: input parameters, shape (u_dim,)
        t_grid: time coordinates, shape (T,)
        x_grid: spatial coordinates, shape (X,)

    Returns:
        predicted solution, shape (T, X)
    """
    b         = branch_model(u)                          # (K,)
    pred_flat = Q_MAT @ b                                # (T*X,)
    return pred_flat.reshape(len(t_grid), len(x_grid))   # (T, X)


def test_mse(Q_MAT, branch_model, u_test, s_test, t_grid, x_grid):
    """
    Compute scalar MSE over the full test set.

    Args:
        Q_MAT: orthonormal trunk bases, shape (T*X, K)
        branch_model: trained branch MLP
        u_test: test input parameters, shape (N_test, u_dim)
        s_test: ground-truth test solutions, shape (N_test, T, X)
        t_grid: time coordinates, shape (T,)
        x_grid: spatial coordinates, shape (X,)

    Returns:
        scalar MSE over all (N, T, X)
    """
    preds = jax.vmap(
        lambda u: predict_solution_from_Q(Q_MAT, branch_model, u, t_grid, x_grid)
    )(u_test)
    diff = preds - s_test
    return jnp.mean(diff ** 2)


def mse_vs_time_Q_branch(Q_MAT, branch_model, u_test, s_test, t_grid, x_grid):
    """
    Compute MSE as a function of time, averaged over the test set.

    Args:
        Q_MAT: orthonormal trunk bases, shape (T*X, K)
        branch_model: trained branch MLP
        u_test: test input parameters, shape (N_test, u_dim)
        s_test: ground-truth test solutions, shape (N_test, T, X)
        t_grid: time coordinates, shape (T,)
        x_grid: spatial coordinates, shape (X,)

    Returns:
        mse_curve: MSE at each time step, shape (T,)
    """
    preds = jax.vmap(
        lambda u: predict_solution_from_Q(Q_MAT, branch_model, u, t_grid, x_grid)
    )(u_test)  # (N, T, X)
    diff = preds - s_test
    # MSE over x per (n, t), then mean over N -> (T,)
    return jnp.mean(diff ** 2, axis=(0, 2))


# ── Quantitative Evaluation ────────────────────────────────────────────────────
final_test_mse = test_mse(jnp.asarray(Q_MAT), branch_model, u_test, s_test, t_grid, x_grid)

print("\n==============================")
print(f"Average Test MSE (Q @ b(u)): {float(final_test_mse):.6e}")
print("==============================\n")


# ── Qualitative Prediction Plots ───────────────────────────────────────────────
# Pick a single test sample for visual inspection
u_rand, s_rand = u_test[20], s_test[20]

rand_pred_tx = compute_QHb_grid(Q_MAT, branch_model, u_rand, t_grid)

# Predicted solution
plt.figure(figsize=(7, 4))
plt.imshow(rand_pred_tx, extent=[x_grid[0], x_grid[-1], t_grid[-1], t_grid[0]], aspect="auto", origin="upper")
plt.colorbar(label=r"$Q(t,x)^* b_{\mathrm{NN}}(u)$")
plt.xlabel("x"); plt.ylabel("t")
plt.title("Grid evaluation of $Q^*(t,x)b_{NN}$, 2-Step")
plt.tight_layout()
plt.savefig("orth_prediction_grid.png")
plt.close()

# Ground-truth solution
plt.figure(figsize=(7, 4))
plt.imshow(s_rand, extent=[x_grid[0], x_grid[-1], t_grid[-1], t_grid[0]], aspect="auto", origin="upper")
plt.colorbar(label=r"$s$")
plt.xlabel("x"); plt.ylabel("t")
plt.title("Grid evaluation of $s$")
plt.tight_layout()
plt.savefig("orth_true_solution_grid.png")
plt.close()

# Pointwise error
plt.figure(figsize=(7, 4))
plt.imshow(rand_pred_tx - s_rand, extent=[x_grid[0], x_grid[-1], t_grid[-1], t_grid[0]], aspect="auto", origin="upper")
plt.colorbar(label=r"$Q(t,x)^* b_{\mathrm{NN}}(u) - s(x,t)$")
plt.xlabel("x"); plt.ylabel("t")
plt.title("Evaluation Error ($Q^*(t,x)b_{NN}(u) - s(x,u)$), 2-Step")
plt.tight_layout()
plt.savefig("orth_error_grid.png")
plt.close()


# ── MSE vs Time Plot ───────────────────────────────────────────────────────────
mse_curve = mse_vs_time_Q_branch(jnp.asarray(Q_MAT), branch_model, u_test, s_test, t_grid, x_grid)

plt.figure()
plt.plot(jnp.asarray(t_grid), jax.device_get(mse_curve))
plt.xlabel("t")
plt.ylabel("MSE over x")
plt.title("Test Mean MSE vs time")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("orth_time_error.png")


# ── Total Runtime ──────────────────────────────────────────────────────────────
master_end_time = time.time()
print("Total Time (sec):", master_end_time - master_start_time)