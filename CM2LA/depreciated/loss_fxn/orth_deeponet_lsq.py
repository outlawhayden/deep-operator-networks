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

dataset_path = Path(config.get('dataset_path', '/Users/haydenoutlaw/Documents/Research/opnet/deep-operator-networks/data/burgers_dataset.npz'))
save_path = Path(config.get('save_path', 'orth_qr_factors.npz'))

trunk_arch = config.get('trunk_arch', [2, 40, 40, 10])
branch_arch = config.get('branch_arch', [2, 60, 60, 10])

trunk_arch[-1] = num_bases
branch_arch[-1] = num_bases

test_size = config.get('test_size', 0.2)
log_frequency = config.get('log_frequency', 10)
lbfgs_log_frequency = config.get('lbfgs_log_frequency', 100)

np.random.seed(seed)
key = jax.random.key(seed)

print("\nconfiguring backend...")
jax.config.update("jax_platform_name", "gpu")
print("backend selected:\n", jax.default_backend())
print("active devices:\n", jax.devices())
print("--------------------\n")

class Linear(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __init__(self, in_size, out_size, key, initializer=he_normal()):
        wkey, bkey = jax.random.split(key)
        self.weight = initializer(wkey, (out_size, in_size), dtype=jnp.float32)
        self.bias = jnp.zeros((out_size,), dtype=jnp.float32)

    def __call__(self, x):
        return self.weight @ x + self.bias


class MLP(eqx.Module):
    layers: list
    activations: list

    def __init__(self, architecture, key, activation=jax.nn.gelu, initializer=he_normal()):
        keys = jax.random.split(key, len(architecture) - 1)
        self.layers = [
            Linear(architecture[i], architecture[i+1], keys[i], initializer=initializer)
            for i in range(len(architecture) - 1)
        ]
        self.activations = [activation] * (len(self.layers) - 1) + [eqx.nn.Identity()]

    def __call__(self, x):
        for layer, act in zip(self.layers, self.activations):
            x = act(layer(x))
        return x


dataset = np.load(dataset_path, allow_pickle=True)
t_grid = jnp.array(dataset['t'])
x_grid = jnp.array(dataset['x'])

data = dataset['samples']
u = np.array([i['params'] for i in data])
s = np.array([i['solution'] for i in data])

n_samp = len(data)
train_indices, test_indices = train_test_split(np.arange(n_samp), test_size=test_size, random_state=seed)
u_train, u_test = jnp.array(u[train_indices]), jnp.array(u[test_indices])
s_train, s_test = jnp.array(s[train_indices]), jnp.array(s[test_indices])

output_tr = jnp.transpose(s_train, axes=(1, 2, 0))
output_test = jnp.transpose(s_test, axes=(1, 2, 0))

tt, xx = jnp.meshgrid(t_grid, x_grid, indexing="ij")
tx_grid = jnp.stack([tt.reshape(-1), xx.reshape(-1)], axis=1)

u_dim = 2
y_dim = 2

key, subkey_t, subkey_A = jax.random.split(key, num=3)
trunk_model = MLP(trunk_arch, key=subkey_t)
A_model = jax.random.normal(subkey_A, (num_bases, output_tr.shape[2]))

model = trunk_model, A_model
best_model = model

def loss_fn(model, inputs, labels):
    trunk_model, A_model = model
    T_MAT = jax.vmap(trunk_model)(inputs)
    pred_y = T_MAT @ A_model
    pred_y_3d = pred_y.reshape(len(t_grid), len(x_grid), -1)
    diff = labels - pred_y_3d
    return jnp.mean(jnp.linalg.norm(diff.reshape(-1, diff.shape[2]), axis=0))

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

@eqx.filter_jit
def train_step(model, opt_state):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, tx_grid, output_tr)
    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

s_time = time.time()
start_time = time.time()
for step in range(num_trunk_epochs):
    model, opt_state, loss = train_step(model, opt_state)
    loss_hist.append(float(loss))
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

model = best_model

plt.plot(loss_hist)
plt.plot(min_loss_hist[1:])
plt.title("Trunk Model Training History, Mean L2, 2-Step")
plt.yscale('log')
plt.savefig('orth_training_history.png')
plt.close()

trunk_model, A_model = model
T_MAT = jax.vmap(trunk_model)(tx_grid)
T_MAT_host = np.asarray(T_MAT)
Q_MAT, R_MAT = scipy.linalg.qr(T_MAT_host, mode="economic")

# fix 1: use lstsq instead of gradient-trained A_model
U_true = np.asarray(output_tr).reshape(len(t_grid) * len(x_grid), -1)  # (T*X, N)
lsq_A = scipy.linalg.lstsq(T_MAT_host, U_true)[0]  # (K, N)
RA_model = (R_MAT @ lsq_A).T  # (N, K)

# fix 2: compute Q_tt for correct prediction formula
Q_tt = scipy.linalg.lstsq(R_MAT.T, T_MAT_host.T)[0].T  # (T*X, K)
Q_tt = jnp.asarray(Q_tt)

Q_MAT = jnp.asarray(Q_MAT)
R_MAT = jnp.asarray(R_MAT)

K = lsq_A.shape[0]
N = lsq_A.shape[1]

key, subkey_t, subkey_A = jax.random.split(key, num=3)
branch_model = MLP(branch_arch, key=subkey_t)

RA_target = jnp.asarray(RA_model)

def branch_loss_fn(branch_model, RA_target, u_train):
    B_MAT = jax.vmap(branch_model)(u_train)
    diff = B_MAT - RA_target
    return jnp.mean(jnp.linalg.norm(diff, axis=1))

@eqx.filter_jit
def branch_train_step(branch_model, opt_state, RA_target, u_train):
    loss, grads = eqx.filter_value_and_grad(branch_loss_fn)(
        branch_model, RA_target, u_train
    )
    updates, opt_state = opt.update(grads, opt_state, branch_model, value=loss)
    branch_model = eqx.apply_updates(branch_model, updates)
    return branch_model, opt_state, loss

branch_loss_hist = []
branch_min_loss_hist = [np.inf]

opt = optax.adam(lr_schedule)
best_branch_model = branch_model
opt_state = opt.init(eqx.filter(branch_model, eqx.is_inexact_array))

for step in range(num_trunk_epochs):
    branch_model, opt_state, loss = branch_train_step(
        branch_model, opt_state, RA_target, u_train
    )
    branch_loss_hist.append(float(loss))
    if float(loss) < float(branch_min_loss_hist[-1]):
        branch_min_loss_hist.append(float(loss))
        best_branch_model = branch_model
    else:
        branch_min_loss_hist.append(float(branch_min_loss_hist[-1]))
    if step % 10000 == 0:
        print(f"\rAdam step {step}: loss={float(loss):.3e}", end="", flush=True)

print(f"\nfinal adam loss: {float(loss):.3e}\n")

branch_model = best_branch_model

plt.plot(branch_loss_hist)
plt.plot(branch_min_loss_hist[1:])
plt.title("Branch Model Training History, Mean L2, 2-Step")
plt.yscale('log')
plt.savefig("orth_branch_training_history.png")
plt.close()

T_MAT_grid = jax.vmap(lambda t: jax.vmap(lambda x: trunk_model(jnp.stack([t, x])))(x_grid))(t_grid)

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
        T_k = T_MAT_grid[:, :, k]
        im = ax.contourf(T, X, T_k, levels=100)
        ax.set_title(f"T[{k}]")
        ax.set_xlabel("t")
        ax.set_ylabel("x")
    else:
        ax.axis("off")

fig.suptitle("Trunk Bases Across Time, 2-Step", fontsize=16)
cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.75)
plt.savefig("orth_bases.png")
plt.close()

trunkA_path = Path(config.get("trunkA_model_path", "orth_trunkA.eqx"))
branch_path = Path(config.get("branch_model_path", "orth_branch.eqx"))
meta_path   = Path(config.get("meta_path", "orth_meta.npz"))
qr_path     = Path(config.get("qr_path", "orth_qr_factors.npz"))

eqx.tree_serialise_leaves(trunkA_path, model)
eqx.tree_serialise_leaves(branch_path, branch_model)

np.savez_compressed(
    meta_path,
    t_grid=np.asarray(t_grid),
    x_grid=np.asarray(x_grid),
    trunk_arch=np.asarray(trunk_arch),
    branch_arch=np.asarray(branch_arch),
    num_bases=np.asarray(num_bases),
    seed=np.asarray(seed),
)

np.savez_compressed(
    qr_path,
    Q=np.asarray(Q_MAT),
    R=np.asarray(R_MAT),
    Q_tt=np.asarray(Q_tt),
)

print("\nexported trunk+A model to:", trunkA_path)
print("exported branch model to:", branch_path)
print("exported metadata to:", meta_path)
print("exported QR factors to:", qr_path)

u_rand, s_rand = u_test[20], s_test[20]

def compute_QHb_grid(Q_tt, branch_model, u, t_grid):
    b = branch_model(u)
    pred_flat = Q_tt @ b
    pred_grid = pred_flat.reshape(len(t_grid), len(x_grid))
    return pred_grid

def predict_solution_from_Q(Q_tt, branch_model, u, t_grid, x_grid):
    b = branch_model(u)
    pred_flat = Q_tt @ b
    return pred_flat.reshape(len(t_grid), len(x_grid))

def test_l2(Q_tt, branch_model, u_test, s_test, t_grid, x_grid):
    preds = jax.vmap(lambda u: predict_solution_from_Q(Q_tt, branch_model, u, t_grid, x_grid))(u_test)
    diff = preds - s_test
    return jnp.mean(jnp.linalg.norm(diff.reshape(diff.shape[0], -1), axis=1))

final_test_l2 = test_l2(Q_tt, branch_model, u_test, s_test, t_grid, x_grid)

print("\n==============================")
print(f"Average Test L2 (Q_tt @ b(u)): {float(final_test_l2):.6e}")
print("==============================\n")

u_rand = u_test[20]
rand_pred_tx = compute_QHb_grid(Q_tt, branch_model, u_rand, t_grid)

plt.figure(figsize=(7, 4))
plt.imshow(
    rand_pred_tx,
    extent=[x_grid[0], x_grid[-1], t_grid[-1], t_grid[0]],
    aspect="auto",
    origin="upper",
)
plt.colorbar(label=r"$Q_{tt}(t,x)^* b_{\mathrm{NN}}(u)$")
plt.xlabel("x")
plt.ylabel("t")
plt.title("Grid evaluation of $Q_{tt}^*(t,x)b_{NN}$, 2-Step")
plt.tight_layout()
plt.savefig("orth_prediction_grid.png")
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
plt.savefig("orth_true_solution_grid.png")
plt.close()

plt.figure(figsize=(7, 4))
plt.imshow(
    rand_pred_tx - s_rand,
    extent=[x_grid[0], x_grid[-1], t_grid[-1], t_grid[0]],
    aspect="auto",
    origin="upper",
)
plt.colorbar(label=r"$Q_{tt}(t,x)^* b_{\mathrm{NN}}(u) - s(x,t)$")
plt.xlabel("x")
plt.ylabel("t")
plt.title("Evaluation Error ($Q_{tt}^*(t,x)b_{NN}(u) - s(x,t)$), 2-Step")
plt.savefig("orth_error_grid.png")
plt.close()

def l2_error_vs_time_Q_branch(Q_tt, branch_model, u_test, s_test, t_grid, x_grid):
    preds = jax.vmap(
        lambda u: predict_solution_from_Q(Q_tt, branch_model, u, t_grid, x_grid)
    )(u_test)
    diff = preds - s_test
    return jnp.mean(jnp.linalg.norm(diff, axis=2), axis=0)

rel_curve = l2_error_vs_time_Q_branch(Q_tt, branch_model, u_test, s_test, t_grid, x_grid)

plt.figure()
plt.plot(jnp.asarray(t_grid), jax.device_get(rel_curve))
plt.xlabel("t")
plt.ylabel("L2 error over x")
plt.title("Test Mean L2 error vs time")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("orth_time_error.png")
plt.close()

master_end_time = time.time()
print("Total Time (sec):", master_end_time - master_start_time)