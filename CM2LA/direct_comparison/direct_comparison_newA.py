## using envml conda environment

import os
import yaml
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
config_path = Path("config.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

gpu_idx = config.get("gpu_idx", 8)
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
seed                = config.get('seed', 42)
num_bases           = config.get('num_bases', 10)
num_trunk_epochs    = config.get('num_trunk_epochs', 6000)
lr_start            = config.get('lr_start', 1e-3)
lr_transition_steps = config.get('lr_transition_steps', 1000)
lr_decay_rate       = config.get('lr_decay_rate', 0.01)
lr_end              = config.get('lr_end', 1e-5)
num_LBFGS_epochs    = config.get('num_LBFGS_epochs', 1000)
eps                 = config.get('eps', 1e-8)

# ── Paths ──────────────────────────────────────────────────────────────────────
dataset_path = Path(config.get('dataset_path', '/Users/haydenoutlaw/Documents/Research/opnet/deep-operator-networks/data/burgers_dataset.npz'))

# ── Network Architectures ──────────────────────────────────────────────────────
trunk_arch_classic   = config.get('trunk_arch', [2, 40, 40, 10])
branch_arch_classic  = config.get('branch_arch', [2, 60, 60, 10])
assert trunk_arch_classic[-1] == branch_arch_classic[-1], "trunk and branch output dims must match"

trunk_arch_orth      = config.get('trunk_arch', [2, 40, 40, 10])[:]
branch_arch_orth     = config.get('branch_arch', [2, 60, 60, 10])[:]
trunk_arch_orth[-1]  = num_bases
branch_arch_orth[-1] = num_bases

trunk_arch_td        = config.get('trunk_arch', [2, 40, 40, 10])[:]
branch_arch_td       = config.get('branch_arch', [2, 60, 60, 10])[:]
trunk_arch_td[-1]    = num_bases
branch_arch_td[-1]   = num_bases
branch_arch_td[0]    = branch_arch_td[0] + 1

# ── Training Parameters ────────────────────────────────────────────────────────
test_size         = config.get('test_size', 0.2)
log_frequency     = config.get('log_frequency', 10)

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
    weight: jax.Array
    bias:   jax.Array

    def __init__(self, in_size, out_size, key, initializer=he_normal()):
        wkey, bkey = jax.random.split(key)
        self.weight = initializer(wkey, (out_size, in_size), dtype=jnp.float32)
        self.bias   = jnp.zeros((out_size,), dtype=jnp.float32)

    def __call__(self, x):
        return self.weight @ x + self.bias


class MLP(eqx.Module):
    layers:      list
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


# ── Dataset Loading ────────────────────────────────────────────────────────────
dataset = np.load(dataset_path, allow_pickle=True)
t_grid  = jnp.array(dataset['t'])
x_grid  = jnp.array(dataset['x'])

data = dataset['samples']
u    = np.array([i['params']   for i in data])
s    = np.array([i['solution'] for i in data])  # (N, T, X)

n_samp = len(data)
train_indices, test_indices = train_test_split(np.arange(n_samp), test_size=test_size, random_state=seed)
u_train, u_test = jnp.array(u[train_indices]), jnp.array(u[test_indices])
s_train, s_test = jnp.array(s[train_indices]), jnp.array(s[test_indices])

output_tr   = jnp.transpose(s_train, axes=(1, 2, 0))  # (T, X, N)
output_test = jnp.transpose(s_test,  axes=(1, 2, 0))

# ── Coordinate Grid ────────────────────────────────────────────────────────────
tt, xx  = jnp.meshgrid(t_grid, x_grid, indexing="ij")
tx_grid = jnp.stack([tt.reshape(-1), xx.reshape(-1)], axis=1)  # (T*X, 2)


# ── LR Schedule ───────────────────────────────────────────────────────────────
lr_schedule = optax.schedules.exponential_decay(
    init_value=lr_start,
    transition_steps=lr_transition_steps,
    decay_rate=lr_decay_rate,
    end_value=lr_end if lr_end is not None else None
)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 1: CLASSIC (joint trunk + branch, no QR)
# ══════════════════════════════════════════════════════════════════════════════

print("\n========== CLASSIC ==========\n")

key, subkey_t, subkey_b = jax.random.split(key, num=3)
classic_trunk  = MLP(trunk_arch_classic,  key=subkey_t)
classic_branch = MLP(branch_arch_classic, key=subkey_b)
classic_model  = classic_trunk, classic_branch
classic_best   = classic_model


def classic_loss_fn(model, inputs, labels):
    trunk_model, branch_model = model
    T_MAT     = jax.vmap(trunk_model)(tx_grid)
    B_MAT     = jax.vmap(branch_model)(inputs)
    pred_y    = T_MAT @ B_MAT.T
    pred_y_3d = pred_y.reshape(len(t_grid), len(x_grid), -1)
    return jnp.mean((labels - pred_y_3d) ** 2)


def classic_l2_fn(model, inputs, labels):
    trunk_model, branch_model = model
    T_MAT     = jax.vmap(trunk_model)(tx_grid)
    B_MAT     = jax.vmap(branch_model)(inputs)
    pred_y    = T_MAT @ B_MAT.T
    pred_y_3d = pred_y.reshape(len(t_grid), len(x_grid), -1)
    diff      = labels - pred_y_3d
    return jnp.mean(jnp.linalg.norm(diff, axis=(0, 1)))


opt_c       = optax.adam(lr_schedule)
opt_state_c = opt_c.init(eqx.filter(classic_model, eqx.is_inexact_array))

classic_loss_hist     = []
classic_min_loss_hist = [np.inf]
classic_l2_hist       = []


@eqx.filter_jit
def classic_train_step(model, opt_state):
    loss, grads = eqx.filter_value_and_grad(classic_loss_fn)(model, u_train, output_tr)
    updates, opt_state = opt_c.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


s_time = start_time = time.time()
for step in range(num_trunk_epochs):
    classic_model, opt_state_c, loss = classic_train_step(classic_model, opt_state_c)
    classic_loss_hist.append(float(loss))
    classic_l2_hist.append(float(classic_l2_fn(classic_model, u_train, output_tr)))
    if float(loss) < classic_min_loss_hist[-1]:
        classic_min_loss_hist.append(float(loss))
        classic_best = classic_model
    else:
        classic_min_loss_hist.append(classic_min_loss_hist[-1])
    if step % log_frequency == 0:
        end_time = time.time()
        print(f"\rAdam step {step}: loss={float(loss):.3e}, {end_time-start_time:.2f}s", end="", flush=True)
        start_time = time.time()
e_time = time.time()
print(f"\nfinal adam loss: {loss:.3e}, total time: {e_time-s_time:.2f}s\n")

classic_best = classic_model


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2: ORTH 2-STEP (time-independent QR)
# ══════════════════════════════════════════════════════════════════════════════

print("\n========== ORTH 2-STEP ==========\n")

key, subkey_t, subkey_A = jax.random.split(key, num=3)
orth_trunk  = MLP(trunk_arch_orth, key=subkey_t)
orth_A      = jax.random.normal(subkey_A, (num_bases, output_tr.shape[2]))  # (K, N)
orth_model  = orth_trunk, orth_A
orth_best   = orth_model


def orth_loss_fn(model, inputs, labels):
    trunk_model, A_model = model
    T_MAT     = jax.vmap(trunk_model)(inputs)
    pred_y    = T_MAT @ A_model
    pred_y_3d = pred_y.reshape(len(t_grid), len(x_grid), -1)
    return jnp.mean((labels - pred_y_3d) ** 2)


def orth_l2_fn(model, inputs, labels):
    trunk_model, A_model = model
    T_MAT     = jax.vmap(trunk_model)(inputs)
    pred_y    = T_MAT @ A_model
    pred_y_3d = pred_y.reshape(len(t_grid), len(x_grid), -1)
    diff      = labels - pred_y_3d
    return jnp.mean(jnp.linalg.norm(diff.reshape(-1, diff.shape[2]), axis=0))


opt_o       = optax.adam(lr_schedule)
opt_state_o = opt_o.init(eqx.filter(orth_model, eqx.is_inexact_array))

orth_loss_hist     = []
orth_min_loss_hist = [np.inf]
orth_l2_hist       = []


@eqx.filter_jit
def orth_train_step(model, opt_state):
    loss, grads = eqx.filter_value_and_grad(orth_loss_fn)(model, tx_grid, output_tr)
    updates, opt_state = opt_o.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


s_time = start_time = time.time()
for step in range(num_trunk_epochs):
    orth_model, opt_state_o, loss = orth_train_step(orth_model, opt_state_o)
    orth_loss_hist.append(float(loss))
    orth_l2_hist.append(float(orth_l2_fn(orth_model, tx_grid, output_tr)))
    if float(loss) < orth_min_loss_hist[-1]:
        orth_min_loss_hist.append(float(loss))
        orth_best = orth_model
    else:
        orth_min_loss_hist.append(float(orth_min_loss_hist[-1]))
    if step % 10000 == 0:
        end_time = time.time()
        print(f"\rAdam step {step}: loss={float(loss):.3e}, {end_time-start_time:.2f}s", end="", flush=True)
        start_time = time.time()
e_time = time.time()
print(f"\nfinal adam loss: {loss:.3e}, total time: {e_time-s_time:.2f}s\n")

orth_best = orth_model

# ── Orth QR Factorization ─────────────────────────────────────────────────────
orth_trunk_trained, orth_A_trained = orth_model
T_MAT_orth   = jax.vmap(orth_trunk_trained)(tx_grid)
Q_orth, R_orth = scipy.linalg.qr(np.asarray(T_MAT_orth), mode="economic")
Q_orth = jnp.asarray(Q_orth)
R_orth = jnp.asarray(R_orth)

# ── Orth Stage 2 Branch ───────────────────────────────────────────────────────
key, subkey_b2 = jax.random.split(key)
orth_branch      = MLP(branch_arch_orth, key=subkey_b2)
orth_branch_best = orth_branch

RA_orth   = (R_orth @ orth_A_trained).T   # (N, K)
RA_orth_t = jnp.asarray(RA_orth)


def orth_branch_loss_fn(branch_model, RA_target, u_train):
    B_MAT = jax.vmap(branch_model)(u_train)
    return jnp.mean((B_MAT - RA_target) ** 2)


def orth_branch_l2_fn(branch_model, RA_target, u_train):
    B_MAT = jax.vmap(branch_model)(u_train)
    return jnp.mean(jnp.linalg.norm(B_MAT - RA_target, axis=1))


opt_ob       = optax.adam(lr_schedule)
opt_state_ob = opt_ob.init(eqx.filter(orth_branch, eqx.is_inexact_array))

orth_branch_loss_hist     = []
orth_branch_min_loss_hist = [np.inf]
orth_branch_l2_hist       = []


@eqx.filter_jit
def orth_branch_train_step(branch_model, opt_state, RA_target, u_train):
    loss, grads = eqx.filter_value_and_grad(orth_branch_loss_fn)(branch_model, RA_target, u_train)
    updates, opt_state = opt_ob.update(grads, opt_state, branch_model)
    branch_model = eqx.apply_updates(branch_model, updates)
    return branch_model, opt_state, loss


s_time = start_time = time.time()
for step in range(num_trunk_epochs):
    orth_branch, opt_state_ob, loss = orth_branch_train_step(orth_branch, opt_state_ob, RA_orth_t, u_train)
    orth_branch_loss_hist.append(float(loss))
    orth_branch_l2_hist.append(float(orth_branch_l2_fn(orth_branch, RA_orth_t, u_train)))
    if float(loss) < float(orth_branch_min_loss_hist[-1]):
        orth_branch_min_loss_hist.append(float(loss))
        orth_branch_best = orth_branch
    else:
        orth_branch_min_loss_hist.append(float(orth_branch_min_loss_hist[-1]))
    if step % 10000 == 0:
        end_time = time.time()
        print(f"\rAdam step {step}: loss={float(loss):.3e}", end="", flush=True)
print(f"\nfinal adam loss: {float(loss):.3e}\n")
orth_branch = orth_branch_best


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 3: ORTH TD 2-STEP (time-dependent QR, closed-form A)
# ══════════════════════════════════════════════════════════════════════════════

print("\n========== ORTH TD 2-STEP ==========\n")

key, subkey_t = jax.random.split(key)
td_trunk = MLP(trunk_arch_td, key=subkey_t)
td_best  = td_trunk


@eqx.filter_jit
def compute_A(trunk_model):
    T_MAT = jax.vmap(lambda t: jax.vmap(lambda x: trunk_model(jnp.stack([t, x])))(x_grid))(t_grid)  # (T, X, K)
    A = jax.vmap(lambda Tt, ot: jnp.linalg.lstsq(Tt, ot)[0])(T_MAT, output_tr)  # (T, K, N)
    return A, T_MAT


@eqx.filter_jit
def td_loss_trunk_only(trunk_model):
    T_MAT = jax.vmap(lambda t: jax.vmap(lambda x: trunk_model(jnp.stack([t, x])))(x_grid))(t_grid)  # (T, X, K)
    A     = jax.vmap(lambda Tt, ot: jnp.linalg.lstsq(Tt, ot)[0])(T_MAT, output_tr)               # (T, K, N)
    pred_y = jnp.einsum("txk,tkn->txn", T_MAT, A)
    return jnp.mean((pred_y - output_tr) ** 2)


def td_l2_fn(trunk_model):
    T_MAT  = jax.vmap(lambda t: jax.vmap(lambda x: trunk_model(jnp.stack([t, x])))(x_grid))(t_grid)
    A      = jax.vmap(lambda Tt, ot: jnp.linalg.lstsq(Tt, ot)[0])(T_MAT, output_tr)
    pred_y = jnp.einsum("txk,tkn->txn", T_MAT, A)
    diff   = pred_y - output_tr
    return jnp.mean(jnp.linalg.norm(diff, axis=(0, 1)))


opt_td       = optax.adam(lr_schedule)
opt_state_td = opt_td.init(eqx.filter(td_trunk, eqx.is_inexact_array))

td_loss_hist     = []
td_min_loss_hist = [np.inf]
td_l2_hist       = []


@eqx.filter_jit
def td_train_step(trunk_model, opt_state):
    loss, grads = eqx.filter_value_and_grad(td_loss_trunk_only)(trunk_model)
    updates, opt_state = opt_td.update(grads, opt_state, trunk_model)
    trunk_model = eqx.apply_updates(trunk_model, updates)
    return trunk_model, opt_state, loss


s_time = start_time = time.time()
for step in range(num_trunk_epochs):
    td_trunk, opt_state_td, loss = td_train_step(td_trunk, opt_state_td)
    td_loss_hist.append(float(loss))
    td_l2_hist.append(float(td_l2_fn(td_trunk)))
    if float(loss) < td_min_loss_hist[-1]:
        td_min_loss_hist.append(float(loss))
        td_best = td_trunk
    else:
        td_min_loss_hist.append(td_min_loss_hist[-1])
    if step % 10000 == 0:
        end_time = time.time()
        print(f"\rAdam step {step}: loss={float(loss):.3e}, {end_time-start_time:.2f}s", end="", flush=True)
        start_time = time.time()
e_time = time.time()
print(f"\nfinal adam loss: {loss:.3e}, total time: {e_time-s_time:.2f}s\n")

td_trunk = td_best

# ── Compute final A from trained trunk ────────────────────────────────────────
td_A_trained, _ = compute_A(td_trunk)  # (T, K, N)

# ── TD QR Factorization with Sign Correction ──────────────────────────────────
T_MAT_td = jax.vmap(lambda t: jax.vmap(lambda x: td_trunk(jnp.stack([t, x])))(x_grid))(t_grid)  # (T, X, K)

Q_td_list = []
R_td_list = []
for i in range(len(t_grid)):
    Q_i, R_i = scipy.linalg.qr(np.asarray(T_MAT_td[i]), mode='economic')
    Q_td_list.append(Q_i)
    R_td_list.append(R_i)

Q_sign = np.stack(Q_td_list, axis=0)  # (T, X, K)
R_sign = np.stack(R_td_list, axis=0)  # (T, K, K)

for k in range(1, len(t_grid)):
    for j in range(num_bases):
        if np.dot(Q_sign[k-1, :, j], Q_sign[k, :, j]) < 0:
            Q_sign[k, :, j] *= -1
            R_sign[k, j, :]  *= -1

# ── TD Stage 2 Branch ─────────────────────────────────────────────────────────
key, subkey_b3 = jax.random.split(key)
td_branch      = MLP(branch_arch_td, key=subkey_b3)
td_branch_best = td_branch

RA_td   = jnp.einsum("tij,tjk->tik", R_sign, td_A_trained)  # (T, K, N)
RA_td_t = jnp.asarray(RA_td)


def td_branch_loss_fn(branch_model, RA_target, u_train, t_grid):
    B_tnK = jax.vmap(
        lambda t: jax.vmap(lambda u: branch_model(jnp.concatenate([u, jnp.array([t])])))(u_train)
    )(t_grid)  # (T, N, K)
    B_MAT = jnp.swapaxes(B_tnK, 1, 2)  # (T, K, N)
    return jnp.mean((B_MAT - RA_target) ** 2)


def td_branch_l2_fn(branch_model, RA_target, u_train, t_grid):
    B_tnK = jax.vmap(
        lambda t: jax.vmap(lambda u: branch_model(jnp.concatenate([u, jnp.array([t])])))(u_train)
    )(t_grid)
    B_MAT = jnp.swapaxes(B_tnK, 1, 2)
    return jnp.mean(jnp.linalg.norm(B_MAT - RA_target, axis=(0, 1)))


opt_tdb       = optax.adam(lr_schedule)
opt_state_tdb = opt_tdb.init(eqx.filter(td_branch, eqx.is_inexact_array))

td_branch_loss_hist     = []
td_branch_min_loss_hist = [np.inf]
td_branch_l2_hist       = []


@eqx.filter_jit
def td_branch_train_step(branch_model, opt_state, RA_target, u_train, t_grid):
    loss, grads = eqx.filter_value_and_grad(td_branch_loss_fn)(branch_model, RA_target, u_train, t_grid)
    updates, opt_state = opt_tdb.update(grads, opt_state, branch_model)
    branch_model = eqx.apply_updates(branch_model, updates)
    return branch_model, opt_state, loss


s_time = start_time = time.time()
for step in range(num_trunk_epochs):
    td_branch, opt_state_tdb, loss = td_branch_train_step(td_branch, opt_state_tdb, RA_td_t, u_train, t_grid)
    td_branch_loss_hist.append(float(loss))
    td_branch_l2_hist.append(float(td_branch_l2_fn(td_branch, RA_td_t, u_train, t_grid)))
    if float(loss) < td_branch_min_loss_hist[-1]:
        td_branch_min_loss_hist.append(float(loss))
        td_branch_best = td_branch
    else:
        td_branch_min_loss_hist.append(td_branch_min_loss_hist[-1])
    if step % 10000 == 0:
        end_time = time.time()
        print(f"\rAdam step {step}: loss={float(loss):.3e}", end="", flush=True)
print(f"\nfinal adam loss: {float(loss):.3e}\n")
td_branch = td_branch_best


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def classic_predict(model, u):
    trunk_model, branch_model = model
    T_MAT     = jax.vmap(trunk_model)(tx_grid)
    b         = branch_model(u)
    return (T_MAT @ b).reshape(len(t_grid), len(x_grid))


def orth_predict(Q_MAT, branch_model, u):
    b = branch_model(u)
    return (Q_MAT @ b).reshape(len(t_grid), len(x_grid))


def td_predict(Q_sign, branch_model, u):
    b_tk = jax.vmap(lambda t: branch_model(jnp.concatenate([u, jnp.array([t])])))(t_grid)  # (T, K)
    return jnp.einsum("txk,tk->tx", Q_sign, b_tk)


# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON PLOTS
# ══════════════════════════════════════════════════════════════════════════════

u_rand, s_rand = u_test[20], s_test[20]

classic_trunk_f, classic_branch_f = classic_model
pred_classic = classic_predict(classic_model, u_rand)
pred_orth    = orth_predict(Q_orth, orth_branch, u_rand)
pred_td      = td_predict(jnp.asarray(Q_sign), td_branch, u_rand)

err_classic = pred_classic - s_rand
err_orth    = pred_orth    - s_rand
err_td      = pred_td      - s_rand

# ── Plot 1: Trunk Training MSE (all three) ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
axes[0].plot(classic_loss_hist);     axes[0].plot(classic_min_loss_hist[1:])
axes[0].set_title("Classic: Trunk/Joint MSE"); axes[0].set_yscale('log')
axes[0].set_xlabel("step");          axes[0].set_ylabel("MSE")

axes[1].plot(orth_loss_hist);        axes[1].plot(orth_min_loss_hist[1:])
axes[1].set_title("Orth 2-Step: Stage 1 MSE"); axes[1].set_yscale('log')
axes[1].set_xlabel("step");          axes[1].set_ylabel("MSE")

axes[2].plot(td_loss_hist);          axes[2].plot(td_min_loss_hist[1:])
axes[2].set_title("Orth TD 2-Step: Stage 1 MSE"); axes[2].set_yscale('log')
axes[2].set_xlabel("step");          axes[2].set_ylabel("MSE")

plt.tight_layout()
plt.savefig("compare_trunk_mse.png")
plt.close()

# ── Plot 2: Branch Training MSE (all three) ───────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
axes[0].plot(classic_loss_hist);         axes[0].plot(classic_min_loss_hist[1:])
axes[0].set_title("Classic: Joint MSE (no separate branch stage)"); axes[0].set_yscale('log')
axes[0].set_xlabel("step");              axes[0].set_ylabel("MSE")

axes[1].plot(orth_branch_loss_hist);     axes[1].plot(orth_branch_min_loss_hist[1:])
axes[1].set_title("Orth 2-Step: Branch MSE"); axes[1].set_yscale('log')
axes[1].set_xlabel("step");              axes[1].set_ylabel("MSE")

axes[2].plot(td_branch_loss_hist);       axes[2].plot(td_branch_min_loss_hist[1:])
axes[2].set_title("Orth TD 2-Step: Branch MSE"); axes[2].set_yscale('log')
axes[2].set_xlabel("step");              axes[2].set_ylabel("MSE")

plt.tight_layout()
plt.savefig("compare_branch_mse.png")
plt.close()

# ── Plot 3: Pointwise error (shared color scale) ──────────────────────────────
vmax = float(max(jnp.abs(err_classic).max(), jnp.abs(err_orth).max(), jnp.abs(err_td).max()))
vmin = -vmax
extent = [float(x_grid[0]), float(x_grid[-1]), float(t_grid[-1]), float(t_grid[0])]

fig, axes = plt.subplots(1, 3, figsize=(18, 4))
im0 = axes[0].imshow(err_classic, extent=extent, aspect="auto", origin="upper", vmin=vmin, vmax=vmax, cmap='RdBu_r')
axes[0].set_title("Classic: Error"); axes[0].set_xlabel("x"); axes[0].set_ylabel("t")

im1 = axes[1].imshow(err_orth,    extent=extent, aspect="auto", origin="upper", vmin=vmin, vmax=vmax, cmap='RdBu_r')
axes[1].set_title("Orth 2-Step: Error"); axes[1].set_xlabel("x"); axes[1].set_ylabel("t")

im2 = axes[2].imshow(err_td,      extent=extent, aspect="auto", origin="upper", vmin=vmin, vmax=vmax, cmap='RdBu_r')
axes[2].set_title("Orth TD 2-Step: Error"); axes[2].set_xlabel("x"); axes[2].set_ylabel("t")

fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
fig.colorbar(im0, cax=cbar_ax, label="pred - true")
plt.savefig("compare_errors.png")
plt.close()

# ── Plot 4: basis functions (all three) ──────────────────────────────────────
T_MAT_classic = jax.vmap(classic_trunk_f)(tx_grid)
bases_classic  = np.array(T_MAT_classic[:, :num_bases].reshape(len(t_grid), len(x_grid), num_bases))

bases_orth = np.array(Q_orth[:, :num_bases].reshape(len(t_grid), len(x_grid), num_bases))

bases_td = Q_sign[:, :, :num_bases]

T_mesh, X_mesh = np.meshgrid(t_grid, x_grid, indexing="ij")

fig, axes = plt.subplots(3, num_bases, figsize=(20, 12), constrained_layout=True)
row_labels = ["Classic", "Orth 2-Step", "Orth TD 2-Step"]
row_bases  = [bases_classic, bases_orth, bases_td]

for row, (label, bases) in enumerate(zip(row_labels, row_bases)):
    bmax_row = float(np.abs(bases).max())
    bmin_row = -bmax_row
    for k in range(num_bases):
        im = axes[row, k].contourf(
            T_mesh, X_mesh, bases[:, :, k],
            levels=100, cmap='viridis', vmin=0, vmax=bmax_row
        )
        axes[row, k].set_title(f"{label} — basis {k}", fontsize=10)
        axes[row, k].set_xlabel("t")
        axes[row, k].set_ylabel("x")
    fig.colorbar(im, ax=axes[row, :].tolist(), shrink=0.6, label=f"{label} basis value")

fig.suptitle("Basis Functions by Model", fontsize=16)
plt.savefig("compare_bases.png")
plt.close()

# ── Plot 5: basis functions — two-step models only ───────────────────────────
fig, axes = plt.subplots(2, num_bases, figsize=(20, 8), constrained_layout=True)
twostep_labels = ["Orth 2-Step", "Orth TD 2-Step"]
twostep_bases  = [bases_orth, bases_td]

for row, (label, bases) in enumerate(zip(twostep_labels, twostep_bases)):
    bmax_row = float(np.abs(bases).max())
    for k in range(num_bases):
        im = axes[row, k].contourf(
            T_mesh, X_mesh, bases[:, :, k],
            levels=100, cmap='viridis', vmin=0, vmax=bmax_row
        )
        axes[row, k].set_title(f"{label} — basis {k}", fontsize=10)
        axes[row, k].set_xlabel("t")
        axes[row, k].set_ylabel("x")
    fig.colorbar(im, ax=axes[row, :].tolist(), shrink=0.6, label=f"{label} basis value")

fig.suptitle("Basis Functions — Two-Step Models", fontsize=16)
plt.savefig("compare_twostep_bases.png")
plt.close()

# ── Plot 6: Pointwise error — two-step models only ───────────────────────────
vmax_ts = float(max(jnp.abs(err_orth).max(), jnp.abs(err_td).max()))
vmin_ts = -vmax_ts

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
im0 = axes[0].imshow(err_orth, extent=extent, aspect="auto", origin="upper",
                     vmin=vmin_ts, vmax=vmax_ts, cmap='RdBu_r')
axes[0].set_title("Orth 2-Step: Error"); axes[0].set_xlabel("x"); axes[0].set_ylabel("t")

im1 = axes[1].imshow(err_td, extent=extent, aspect="auto", origin="upper",
                     vmin=vmin_ts, vmax=vmax_ts, cmap='RdBu_r')
axes[1].set_title("Orth TD 2-Step: Error"); axes[1].set_xlabel("x"); axes[1].set_ylabel("t")

fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
fig.colorbar(im0, cax=cbar_ax, label="pred - true")
plt.savefig("compare_twostep_errors.png")
plt.close()

# ── Plot 7: Test MSE Histogram — Two-Step Models ──────────────────────────────
def compute_test_mse_per_sample(Q_mat, branch_model, u_test, s_test, is_td=False):
    mses = []
    for i in range(len(u_test)):
        u_i = u_test[i]
        s_i = s_test[i]
        if is_td:
            pred = td_predict(jnp.asarray(Q_mat), branch_model, u_i)
        else:
            pred = orth_predict(Q_mat, branch_model, u_i)
        mse = float(jnp.mean((pred - s_i) ** 2))
        mses.append(mse)
    return np.array(mses)

mse_orth = compute_test_mse_per_sample(Q_orth, orth_branch, u_train, s_train, is_td=False)
mse_td   = compute_test_mse_per_sample(Q_sign, td_branch,   u_train, s_train, is_td=True)

log_mse_orth = np.log10(mse_orth)
log_mse_td   = np.log10(mse_td)

bins = np.linspace(
    min(log_mse_orth.min(), log_mse_td.min()),
    max(log_mse_orth.max(), log_mse_td.max()),
    40
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(log_mse_orth, bins=bins, alpha=0.55, label="Orth 2-Step",    color="steelblue")
ax.hist(log_mse_td,   bins=bins, alpha=0.55, label="Orth TD 2-Step", color="darkorange")
ax.axvline(np.mean(log_mse_orth), color="steelblue",  linestyle="--", linewidth=1.5, label=f"Orth mean: {np.mean(log_mse_orth):.2f}")
ax.axvline(np.mean(log_mse_td),   color="darkorange", linestyle="--", linewidth=1.5, label=f"TD mean:   {np.mean(log_mse_td):.2f}")
ax.set_xlabel("log$_{10}$(MSE)")
ax.set_ylabel("Count")
ax.set_title("Training MSE Distribution — Two-Step Models")
ax.legend()
plt.tight_layout()
plt.savefig("compare_twostep_test_mse_hist.png")
plt.close()

# ── Plot 8: Mean MSE Across Time — Two-Step Models ───────────────────────────
def compute_mse_per_timestep(Q_mat, branch_model, u_test, s_test, is_td=False):
    all_preds = []
    for i in range(len(u_test)):
        u_i = u_test[i]
        if is_td:
            pred = td_predict(jnp.asarray(Q_mat), branch_model, u_i)
        else:
            pred = orth_predict(Q_mat, branch_model, u_i)
        all_preds.append(np.array(pred))
    all_preds = np.stack(all_preds, axis=0)
    s_test_np = np.array(s_test)
    return np.mean((all_preds - s_test_np) ** 2, axis=(0, 2))

mse_orth_t = compute_mse_per_timestep(Q_orth, orth_branch, u_train, s_train, is_td=False)
mse_td_t   = compute_mse_per_timestep(Q_sign, td_branch,   u_train, s_train, is_td=True)

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(np.array(t_grid), mse_orth_t, label="Orth 2-Step",    color="steelblue")
ax.semilogy(np.array(t_grid), mse_td_t,   label="Orth TD 2-Step", color="darkorange")
ax.set_xlabel("t")
ax.set_ylabel("Mean MSE")
ax.set_title("Mean Training MSE Across Time — Two-Step Models")
ax.legend()
plt.tight_layout()
plt.savefig("compare_twostep_mse_over_time.png")
plt.close()

fig, ax = plt.subplots(figsize=(10, 5))
ax.semilogy(classic_min_loss_hist[1:],  label="Classic (joint)",           color="forestgreen")
ax.semilogy(orth_min_loss_hist[1:],     label="Orth 2-Step (stage 1)",     color="steelblue")
ax.semilogy(td_min_loss_hist[1:],       label="Orth TD 2-Step (stage 1)",  color="darkorange")
ax.set_xlabel("step")
ax.set_ylabel("Min MSE")
ax.set_title("Stage 1 Min Training MSE — All Models")
ax.legend()
plt.tight_layout()
plt.savefig("compare_stage1_min_mse.png")
plt.close()

# ── Total Runtime ──────────────────────────────────────────────────────────────
master_end_time = time.time()
print("Total Time (sec):", master_end_time - master_start_time)