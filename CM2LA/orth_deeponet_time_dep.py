## using envml conda environment

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
save_path = Path(config.get('save_path', 'orth_qr_td_factors.npz'))

# ── Network Architectures ──────────────────────────────────────────────────────
# trunk:  maps (t, x) -> R^K
# branch: maps (u, t) -> R^K  (+1 input for time dependence)
# output dims are overridden by num_bases to stay consistent
trunk_arch = config.get('trunk_arch', [2, 40, 40, 10])
branch_arch = config.get('branch_arch', [2, 60, 60, 10])

trunk_arch[-1]  = num_bases
branch_arch[-1] = num_bases
branch_arch[0]  = branch_arch[0] + 1  # extend branch input dim to include t

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
            activation: activation function applied after each hidden layer (default: leaky ReLU)
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
output_tr = jnp.transpose(s_train, axes=(1, 2, 0))


# ── Coordinate Grid ────────────────────────────────────────────────────────────
# Build a flat (T*X, 2) grid of (t, x) query points for the trunk network
tt, xx = jnp.meshgrid(t_grid, x_grid, indexing="ij")
tx_grid = jnp.concatenate([tt.flatten()[:, None], xx.flatten()[:, None]], axis=1)  # (T*X, 2)

u_dim = 2  # number of PDE input parameters
y_dim = 2  # query point dimension (t, x)


# ── Stage 1 Model Initialization ──────────────────────────────────────────────
# Stage 1 jointly trains the trunk MLP and a time-dependent coefficient tensor A_model.
# A_model[t, k, n] holds the expansion coefficient for basis k, sample n, at time t.
# After training, A_model is factored via QR to produce orthonormal trunk bases Q.

key, subkey_t, subkey_A = jax.random.split(key, num=3)
trunk_model = MLP(trunk_arch, key=subkey_t)
A_model = jax.random.normal(subkey_A, (len(t_grid), num_bases, output_tr.shape[2]))  # (T, K, N)

model      = trunk_model, A_model
best_model = model


# ── Stage 1 Loss Functions ─────────────────────────────────────────────────────

def loss_fn(model):
    """
    MSE loss for stage 1 (trunk + A_model joint training).

    Prediction: pred[t, x, n] = sum_k T[t, x, k] * A[t, k, n]

    Args:
        model: (trunk_model, A_model) tuple

    Returns:
        scalar MSE over (T, X, N)
    """
    trunk_model, A_model = model
    T_MAT  = jax.vmap(lambda t: jax.vmap(lambda x: trunk_model(jnp.stack([t,x])))(x_grid))(t_grid)  # (T, X, K)
    pred_y = jnp.einsum("txk,tkn->txn", T_MAT, A_model)   # (T, X, N)
    diff   = pred_y - output_tr
    return jnp.mean(diff ** 2)


def l2_fn(model):
    """
    Mean absolute L2 norm over samples (diagnostic only, not used in training).

    Args:
        model: (trunk_model, A_model) tuple

    Returns:
        scalar mean L2 error over N
    """
    trunk_model, A_model = model
    T_MAT  = jax.vmap(lambda t: jax.vmap(lambda x: trunk_model(jnp.stack([t, x])))(x_grid))(t_grid)  # (T, X, K)
    pred_y = jnp.einsum("txk,tkn->txn", T_MAT, A_model)   # (T, X, N)
    diff   = pred_y - output_tr
    return jnp.mean(jnp.linalg.norm(diff, axis=(0, 1)))  # mean L2 over N


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
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


# ── Stage 1 Adam Training Loop ─────────────────────────────────────────────────
s_time     = time.time()
start_time = time.time()

for step in range(num_trunk_epochs):
    model, opt_state, loss = train_step(model, opt_state)
    loss_hist.append(float(loss))
    l2_hist.append(float(l2_fn(model)))  # diagnostic, not used in training

    # track best model by minimum training loss
    if float(loss) < min_loss_hist[-1]:
        min_loss_hist.append(float(loss))
        best_model = model
    else:
        min_loss_hist.append(min_loss_hist[-1])

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

frozen_loss = jax.jit(lambda p: loss_fn(eqx.combine(p, static)))
frozen_l2   = jax.jit(lambda p: l2_fn(eqx.combine(p, static)))

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
axes[0].set_title("Trunk Model Training History, MSE, Orth 2-Step")
axes[0].set_yscale('log')
axes[0].set_xlabel("step")
axes[0].set_ylabel("MSE")
axes[1].plot(l2_hist)
axes[1].set_title("Trunk Model Training History, Absolute L2, Orth 2-Step")
axes[1].set_yscale('log')
axes[1].set_xlabel("step")
axes[1].set_ylabel("Absolute L2")
plt.tight_layout()
plt.savefig("orth_td_training_history.png")
plt.close()


# ── QR Factorization of Trunk Bases ───────────────────────────────────────────
# At each time step t, compute the thin QR decomposition of T_MAT[t] (shape X x K).
# This yields orthonormal spatial bases Q[t] and upper-triangular factors R[t].
# Sign correction ensures Q columns vary continuously across time.

trunk_model, A_model = model
T_MAT = jax.vmap(lambda t: jax.vmap(lambda x: trunk_model(jnp.stack([t, x])))(x_grid))(t_grid)  # (T, X, K)

for i in range(len(t_grid)):
    Q, R = scipy.linalg.qr(T_MAT[i], mode='economic')  # Q: (X, K), R: (K, K)
    if i == 0:
        Q_MAT = Q[jnp.newaxis, :, :]   # initialise 3-tensor: (1, X, K)
        R_MAT = R[jnp.newaxis, :, :]
    else:
        Q_MAT = jnp.concatenate([Q_MAT, Q[jnp.newaxis, :, :]], axis=0)  # (t+1, X, K)
        R_MAT = jnp.concatenate([R_MAT, R[jnp.newaxis, :, :]], axis=0)

# copy to numpy for sign-correction (mutable)
Q_sign = np.array(Q_MAT, copy=True)
R_sign = np.array(R_MAT, copy=True)

# ── Sign Correction ────────────────────────────────────────────────────────────
# QR is unique only up to column sign. Flip each basis column at time k so that
# it has positive inner product with the same column at time k-1, enforcing
# temporal continuity of the orthonormal bases.
T_len = Q_sign.shape[0]
n_basis = Q_sign.shape[2]

for k in range(1, T_len):
    for j in range(n_basis):
        if np.dot(Q_sign[k-1, :, j], Q_sign[k, :, j]) < 0:
            Q_sign[k, :, j] *= -1   # flip Q column
            R_sign[k, j, :] *= -1   # flip corresponding R row to keep Q @ R consistent


# ── Orthonormal Basis Visualization ───────────────────────────────────────────
ncols = 4
nrows = int(np.ceil(num_bases / ncols))
T, X  = np.meshgrid(t_grid, x_grid, indexing="ij")

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), constrained_layout=True)
axes = axes.flatten()

for k, ax in enumerate(axes):
    if k < num_bases:
        Q_k = Q_sign[:, :, k]
        im  = ax.contourf(T, X, Q_k, levels=100, cmap='viridis')
        ax.set_title(f"Q[{k}]")
        ax.set_xlabel("t")
        ax.set_ylabel("x")
    else:
        ax.axis("off")

fig.suptitle("Orthogonalized Trunk Bases Across Time, 2-Step QR (Sign Corrected)", fontsize=16)
cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.75)
plt.savefig("orth_td_bases.png")
plt.close()


# ── Stage 2: Branch Network Training ──────────────────────────────────────────
# The branch network learns to predict R(t) @ A(t, :, n) from (u_n, t).
# Target: RA_target[t, k, n] = sum_j R[t, k, j] * A[t, j, n]
# This decouples the branch from the trunk after stage 1.

K = A_model.shape[1]  # number of basis functions
N = A_model.shape[2]  # number of training samples

key, subkey_t, subkey_A = jax.random.split(key, num=3)
branch_model      = MLP(branch_arch, key=subkey_t)
best_branch_model = branch_model

# pre-compute the frozen regression target RA_target = R @ A  for all time steps
RA_model  = jnp.einsum("tij,tjk->tik", R_sign, A_model)  # (T, K, N)
RA_target = jnp.asarray(RA_model)                         # frozen, not updated during branch training


# ── Stage 2 Loss Functions ─────────────────────────────────────────────────────

def branch_loss_fn(branch_model, RA_target, u_train, t_grid, eps=1e-8):
    """
    MSE loss for stage 2 branch training.

    The branch maps (u, t) -> R^K. The target at each (t, n) is RA_target[t, :, n].

    Args:
        branch_model: branch MLP
        RA_target: frozen regression targets, shape (T, K, N)
        u_train: training input parameters, shape (N, 2)
        t_grid: time coordinates, shape (T,)
        eps: unused stability constant (reserved)

    Returns:
        scalar MSE
    """
    B_tnK = jax.vmap(
        lambda t: jax.vmap(
            lambda u: branch_model(jnp.concatenate([u, jnp.array([t])]))
        )(u_train)
    )(t_grid)  # (T, N, K)
    B_MAT = jnp.swapaxes(B_tnK, 1, 2)  # (T, K, N)
    diff  = B_MAT - RA_target
    return jnp.mean(diff ** 2)


def branch_l2_fn(branch_model, RA_target, u_train, t_grid):
    """
    Mean absolute L2 norm over samples for branch (diagnostic only).

    Args:
        branch_model: branch MLP
        RA_target: frozen regression targets, shape (T, K, N)
        u_train: training input parameters, shape (N, 2)
        t_grid: time coordinates, shape (T,)

    Returns:
        scalar mean L2 error over N
    """
    B_tnK = jax.vmap(
        lambda t: jax.vmap(
            lambda u: branch_model(jnp.concatenate([u, jnp.array([t])]))
        )(u_train)
    )(t_grid)  # (T, N, K)
    B_MAT = jnp.swapaxes(B_tnK, 1, 2)  # (T, K, N)
    diff  = B_MAT - RA_target
    return jnp.mean(jnp.linalg.norm(diff, axis=(0, 1)))


# ── Stage 2 Adam Optimizer ─────────────────────────────────────────────────────
@eqx.filter_jit
def branch_train_step(branch_model, opt_state, RA_target, u_train, t_grid):
    """Single Adam gradient update step for the branch network."""
    loss, grads = eqx.filter_value_and_grad(branch_loss_fn)(
        branch_model, RA_target, u_train, t_grid
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
        branch_model, opt_state, RA_target, u_train, t_grid
    )
    branch_loss_hist.append(float(loss))
    branch_l2_hist.append(float(branch_l2_fn(branch_model, RA_target, u_train, t_grid)))

    # track best branch model by minimum training loss
    if float(loss) < branch_min_loss_hist[-1]:
        branch_min_loss_hist.append(float(loss))
        best_branch_model = branch_model
    else:
        branch_min_loss_hist.append(branch_min_loss_hist[-1])

    if step % 10000 == 0:
        end_time = time.time()
        print(f"\rAdam step {step}: loss={float(loss):.3e}", end="", flush=True)

print(f"\nfinal adam loss: {float(loss):.3e}\n")

branch_model = best_branch_model  # restore best checkpoint


# ── Model Export ───────────────────────────────────────────────────────────────
trunkA_path = Path(config.get("trunkA_model_path", "orth_td_trunkA.eqx"))
branch_path = Path(config.get("branch_model_path", "orth_td_branch.eqx"))
meta_path   = Path(config.get("meta_path",          "orth_td_meta.npz"))
qr_path     = Path(config.get("qr_path",             "orth_td_qr_factors.npz"))

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
    N_train=np.asarray(output_tr.shape[2]),
    T_len=np.asarray(len(t_grid)),
)

# sign-corrected QR factors for downstream analysis / reconstruction
np.savez_compressed(qr_path, Q=np.asarray(Q_sign), R=np.asarray(R_sign))

print("\nexported trunk+A model to:", trunkA_path)
print("exported branch model to:", branch_path)
print("exported metadata to:", meta_path)
print("exported QR factors to:", qr_path)


# ── Stage 2 Training Curves ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(branch_loss_hist)
axes[0].plot(branch_min_loss_hist[1:])
axes[0].set_title("Branch Model Training History, MSE, Orth 2-Step")
axes[0].set_yscale('log')
axes[0].set_xlabel("step")
axes[0].set_ylabel("MSE")
axes[1].plot(branch_l2_hist)
axes[1].set_title("Branch Model Training History, Absolute L2, Orth 2-Step")
axes[1].set_yscale('log')
axes[1].set_xlabel("step")
axes[1].set_ylabel("Absolute L2")
plt.tight_layout()
plt.savefig('orth_td_branch_training_history.png')
plt.close()


# ── Inference Utilities ────────────────────────────────────────────────────────

def branch_over_time(branch_model, u, t_grid):
    """
    Evaluate the branch network for a single u at every time step.

    Args:
        branch_model: trained branch MLP
        u: input parameters, shape (u_dim,)
        t_grid: time coordinates, shape (T,)

    Returns:
        b_tk: branch outputs, shape (T, K)
    """
    return jax.vmap(lambda t: branch_model(jnp.concatenate([u, jnp.array([t])])))(t_grid)


def compute_QHb_grid(Q_sign, branch_model, u, t_grid):
    """
    Evaluate the full (T, X) prediction using orthonormal bases Q and branch outputs b.

    Prediction: pred[t, x] = sum_k Q[t, x, k] * b[t, k]

    Args:
        Q_sign: sign-corrected orthonormal trunk bases, shape (T, X, K)
        branch_model: trained branch MLP
        u: input parameters, shape (u_dim,)
        t_grid: time coordinates, shape (T,)

    Returns:
        pred: predicted solution, shape (T, X)
    """
    b_tk = branch_over_time(branch_model, u, t_grid)          # (T, K)
    return jnp.einsum("txk,tk->tx", Q_sign, b_tk)             # (T, X)


def predict_solution_Q_time_dep(Q_sign, branch_model, u, t_grid):
    """
    Alias for compute_QHb_grid; used in batched evaluation helpers below.

    Args:
        Q_sign: sign-corrected orthonormal trunk bases, shape (T, X, K)
        branch_model: trained branch MLP
        u: input parameters, shape (u_dim,)
        t_grid: time coordinates, shape (T,)

    Returns:
        pred: predicted solution, shape (T, X)
    """
    b_tk = branch_over_time(branch_model, u, t_grid)
    return jnp.einsum("txk,tk->tx", Q_sign, b_tk)


def test_mse_time_dep(Q_sign, branch_model, u_test, s_test, t_grid):
    """
    Compute scalar MSE over the full test set.

    Args:
        Q_sign: sign-corrected orthonormal trunk bases, shape (T, X, K)
        branch_model: trained branch MLP
        u_test: test input parameters, shape (N_test, u_dim)
        s_test: ground-truth test solutions, shape (N_test, T, X)
        t_grid: time coordinates, shape (T,)

    Returns:
        scalar MSE over all (N, T, X)
    """
    preds = jax.vmap(
        lambda u: predict_solution_Q_time_dep(Q_sign, branch_model, u, t_grid)
    )(u_test)  # (N_test, T, X)
    return jnp.mean((preds - s_test) ** 2)


def test_mse_vs_time_time_dep(Q_sign, branch_model, u_test, s_test, t_grid):
    """
    Compute MSE as a function of time, averaged over the test set.

    Args:
        Q_sign: sign-corrected orthonormal trunk bases, shape (T, X, K)
        branch_model: trained branch MLP
        u_test: test input parameters, shape (N_test, u_dim)
        s_test: ground-truth test solutions, shape (N_test, T, X)
        t_grid: time coordinates, shape (T,)

    Returns:
        mse_curve: MSE at each time step, shape (T,)
    """
    preds = jax.vmap(
        lambda u: predict_solution_Q_time_dep(Q_sign, branch_model, u, t_grid)
    )(u_test)  # (N_test, T, X)
    diff = preds - s_test
    # MSE over x per (n, t), then mean over N -> (T,)
    return jnp.mean(diff ** 2, axis=(0, 2))


# ── Qualitative Prediction Plots ───────────────────────────────────────────────
# Pick a single test sample for visual inspection
u_rand, s_rand = u_test[20], s_test[20]

rand_pred_tx = compute_QHb_grid(Q_sign, branch_model, jnp.array(u_rand), jnp.array(t_grid))

# Predicted solution
plt.figure(figsize=(7, 4))
plt.imshow(rand_pred_tx, extent=[x_grid[0], x_grid[-1], t_grid[-1], t_grid[0]], aspect="auto", origin="upper")
plt.colorbar(label=r"$Q(t,x)^* b_{\mathrm{NN}}(t)$")
plt.xlabel("x"); plt.ylabel("t")
plt.title("Grid evaluation of $Q^*(t,x)b_{NN}(t)$, Orth 2-Step")
plt.tight_layout()
plt.savefig("orth_td_prediction_grid.png")

# Ground-truth solution
plt.figure(figsize=(7, 4))
plt.imshow(s_rand, extent=[x_grid[0], x_grid[-1], t_grid[-1], t_grid[0]], aspect="auto", origin="upper")
plt.colorbar(label=r"$s$")
plt.xlabel("x"); plt.ylabel("t")
plt.title("Grid evaluation of $s$")
plt.tight_layout()
plt.savefig("orth_td_true_solution_grid.png")

# Pointwise error
plt.figure(figsize=(7, 4))
plt.imshow(rand_pred_tx - s_rand, extent=[x_grid[0], x_grid[-1], t_grid[-1], t_grid[0]], aspect="auto", origin="upper")
plt.colorbar(label=r"$Q(t,x)^* b_{\mathrm{NN}}(t) - s(x,t)$")
plt.xlabel("x"); plt.ylabel("t")
plt.title("Evaluation Error ($Q^*(t,x)b_{NN}(t) - s(x,t)$), Orth 2-Step")
plt.tight_layout()
plt.savefig("orth_td_error_grid.png")
plt.close()


# ── Quantitative Evaluation ────────────────────────────────────────────────────
final_test_mse = test_mse_time_dep(
    jnp.asarray(Q_sign), branch_model, u_test, s_test, t_grid
)

final_test_mse_curve = test_mse_vs_time_time_dep(
    jnp.asarray(Q_sign), branch_model, u_test, s_test, t_grid
)

print("\n==============================")
print(f"Test Avg MSE (time-dependent Q @ b(t,u)): {float(jnp.mean(final_test_mse_curve)):.6e}")
print("==============================\n")

# MSE as a function of time
mse_curve = test_mse_vs_time_time_dep(jnp.asarray(Q_sign), branch_model, u_test, s_test, t_grid)

plt.figure()
plt.plot(np.asarray(t_grid), np.asarray(jax.device_get(mse_curve)))
plt.xlabel("t")
plt.ylabel("mean MSE over x")
plt.title("Test Mean MSE vs time (time-dependent Q @ b(t,u))")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("orth_td_mse_vs_time.png")
plt.close()

print("saved: orth_td_mse_vs_time.png")


# ── Total Runtime ──────────────────────────────────────────────────────────────
master_end_time = time.time()
print("Total Time (sec):", master_end_time - master_start_time)