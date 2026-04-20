## using envml conda environment

import os
import yaml
import sys
from pathlib import Path

config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config_orth.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

gpu_idx = config.get("gpu_idx", 8)
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

import jax.numpy as jnp
import numpy as np
import jax
import optax
import scipy.linalg
import equinox as eqx
from jax.nn.initializers import he_normal
from sklearn.model_selection import train_test_split
import time

# ── Hyperparameters ────────────────────────────────────────────────────────────
seed                = config.get('seed', 42)
num_bases           = config.get('num_bases', 10)
num_trunk_epochs    = config.get('num_trunk_epochs', 6000)
lr_start            = config.get('lr_start', 1e-3)
lr_transition_steps = config.get('lr_transition_steps', 1000)
lr_decay_rate       = config.get('lr_decay_rate', 0.01)
lr_end              = config.get('lr_end', 1e-5)
test_size           = config.get('test_size', 0.2)
log_frequency       = config.get('log_frequency', 10)
save_dir            = Path(config.get('save_dir', 'results/orth'))
save_dir.mkdir(parents=True, exist_ok=True)

dataset_path = Path(config.get('dataset_path'))

trunk_arch  = config.get('trunk_arch',  [2, 40, 40, 10])[:]
branch_arch = config.get('branch_arch', [2, 60, 60, 10])[:]
trunk_arch[-1]  = num_bases
branch_arch[-1] = num_bases

np.random.seed(seed)
key = jax.random.key(seed)

print("\nconfiguring backend...")
jax.config.update("jax_platform_name", "METAL")
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
s    = np.array([i['solution'] for i in data])

n_samp = len(data)
train_indices, test_indices = train_test_split(np.arange(n_samp), test_size=test_size, random_state=seed)
u_train, u_test = jnp.array(u[train_indices]), jnp.array(u[test_indices])
s_train, s_test = jnp.array(s[train_indices]), jnp.array(s[test_indices])

output_tr   = jnp.transpose(s_train, axes=(1, 2, 0))
output_test = jnp.transpose(s_test,  axes=(1, 2, 0))

tt, xx  = jnp.meshgrid(t_grid, x_grid, indexing="ij")
tx_grid = jnp.stack([tt.reshape(-1), xx.reshape(-1)], axis=1)


# ── LR Schedule ───────────────────────────────────────────────────────────────
lr_schedule = optax.schedules.exponential_decay(
    init_value=lr_start,
    transition_steps=lr_transition_steps,
    decay_rate=lr_decay_rate,
    end_value=lr_end if lr_end is not None else None
)


# ── Stage 1: Trunk + A ────────────────────────────────────────────────────────
key, subkey_t, subkey_A = jax.random.split(key, num=3)
orth_trunk = MLP(trunk_arch, key=subkey_t)
orth_A     = jax.random.normal(subkey_A, (num_bases, output_tr.shape[2]))
orth_model = orth_trunk, orth_A


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


print("\n========== ORTH 2-STEP: Stage 1 ==========\n")
s_time = start_time = time.time()
for step in range(num_trunk_epochs):
    orth_model, opt_state_o, loss = orth_train_step(orth_model, opt_state_o)
    orth_loss_hist.append(float(loss))
    orth_l2_hist.append(float(orth_l2_fn(orth_model, tx_grid, output_tr)))
    if float(loss) < orth_min_loss_hist[-1]:
        orth_min_loss_hist.append(float(loss))
    else:
        orth_min_loss_hist.append(float(orth_min_loss_hist[-1]))
    if step % log_frequency == 0:
        end_time = time.time()
        print(f"\rAdam step {step}: loss={float(loss):.3e}, {end_time-start_time:.2f}s", end="", flush=True)
        start_time = time.time()
e_time = time.time()
print(f"\nfinal adam loss: {loss:.3e}, total time: {e_time-s_time:.2f}s\n")


# ── QR Factorization ──────────────────────────────────────────────────────────
orth_trunk_trained, orth_A_trained = orth_model
T_MAT_orth = jax.vmap(orth_trunk_trained)(tx_grid)
Q_orth, R_orth = scipy.linalg.qr(np.asarray(T_MAT_orth), mode="economic")
Q_orth = jnp.asarray(Q_orth)
R_orth = jnp.asarray(R_orth)

RA_orth   = (R_orth @ orth_A_trained).T
RA_orth_t = jnp.asarray(RA_orth)


# ── Stage 2: Branch ───────────────────────────────────────────────────────────
key, subkey_b2 = jax.random.split(key)
orth_branch      = MLP(branch_arch, key=subkey_b2)
orth_branch_best = orth_branch


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


print("\n========== ORTH 2-STEP: Stage 2 ==========\n")
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
    if step % log_frequency == 0:
        print(f"\rAdam step {step}: loss={float(loss):.3e}", end="", flush=True)
print(f"\nfinal adam loss: {float(loss):.3e}\n")
orth_branch = orth_branch_best


# ── Save ──────────────────────────────────────────────────────────────────────
eqx.tree_serialise_leaves(save_dir / "trunk.eqx",  orth_trunk_trained)
eqx.tree_serialise_leaves(save_dir / "branch.eqx", orth_branch)

np.savez(
    save_dir / "data.npz",
    t_grid                  = np.array(t_grid),
    x_grid                  = np.array(x_grid),
    tx_grid                 = np.array(tx_grid),
    u_train                 = np.array(u_train),
    u_test                  = np.array(u_test),
    s_train                 = np.array(s_train),
    s_test                  = np.array(s_test),
    train_indices           = train_indices,
    test_indices            = test_indices,
    Q_orth                  = np.array(Q_orth),
    R_orth                  = np.array(R_orth),
    trunk_loss_hist         = np.array(orth_loss_hist),
    trunk_min_loss_hist     = np.array(orth_min_loss_hist[1:]),
    trunk_l2_hist           = np.array(orth_l2_hist),
    branch_loss_hist        = np.array(orth_branch_loss_hist),
    branch_min_loss_hist    = np.array(orth_branch_min_loss_hist[1:]),
    branch_l2_hist          = np.array(orth_branch_l2_hist),
    trunk_arch              = np.array(trunk_arch),
    branch_arch             = np.array(branch_arch),
)

print(f"Saved to {save_dir}")