## using envml conda environment

import os
import yaml
import sys
from pathlib import Path

config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config_classic.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

gpu_idx = config.get("gpu_idx", 8)
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

import jax.numpy as jnp
import numpy as np
import jax
import optax
import scipy
import equinox as eqx
from jax.nn.initializers import he_normal
from sklearn.model_selection import train_test_split
import time

# ── Hyperparameters ────────────────────────────────────────────────────────────
seed                = config.get('seed', 42)
num_trunk_epochs    = config.get('num_trunk_epochs', 6000)
lr_start            = config.get('lr_start', 1e-3)
lr_transition_steps = config.get('lr_transition_steps', 1000)
lr_decay_rate       = config.get('lr_decay_rate', 0.01)
lr_end              = config.get('lr_end', 1e-5)
test_size           = config.get('test_size', 0.2)
log_frequency       = config.get('log_frequency', 10)
save_dir            = Path(config.get('save_dir', 'results/classic'))
save_dir.mkdir(parents=True, exist_ok=True)

dataset_path = Path(config.get('dataset_path'))

trunk_arch  = config.get('trunk_arch',  [2, 40, 40, 10])
branch_arch = config.get('branch_arch', [2, 60, 60, 10])
assert trunk_arch[-1] == branch_arch[-1], "trunk and branch output dims must match"

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


# ── Model Init ────────────────────────────────────────────────────────────────
key, subkey_t, subkey_b = jax.random.split(key, num=3)
classic_trunk  = MLP(trunk_arch,  key=subkey_t)
classic_branch = MLP(branch_arch, key=subkey_b)
classic_model  = classic_trunk, classic_branch


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


print("\n========== CLASSIC ==========\n")
s_time = start_time = time.time()
for step in range(num_trunk_epochs):
    classic_model, opt_state_c, loss = classic_train_step(classic_model, opt_state_c)
    classic_loss_hist.append(float(loss))
    classic_l2_hist.append(float(classic_l2_fn(classic_model, u_train, output_tr)))
    if float(loss) < classic_min_loss_hist[-1]:
        classic_min_loss_hist.append(float(loss))
    else:
        classic_min_loss_hist.append(classic_min_loss_hist[-1])
    if step % log_frequency == 0:
        end_time = time.time()
        print(f"\rAdam step {step}: loss={float(loss):.3e}, {end_time-start_time:.2f}s", end="", flush=True)
        start_time = time.time()
e_time = time.time()
print(f"\nfinal adam loss: {loss:.3e}, total time: {e_time-s_time:.2f}s\n")


# ── Save ──────────────────────────────────────────────────────────────────────
classic_trunk_f, classic_branch_f = classic_model
eqx.tree_serialise_leaves(save_dir / "trunk.eqx",  classic_trunk_f)
eqx.tree_serialise_leaves(save_dir / "branch.eqx", classic_branch_f)

np.savez(
    save_dir / "data.npz",
    t_grid             = np.array(t_grid),
    x_grid             = np.array(x_grid),
    tx_grid            = np.array(tx_grid),
    u_train            = np.array(u_train),
    u_test             = np.array(u_test),
    s_train            = np.array(s_train),
    s_test             = np.array(s_test),
    train_indices      = train_indices,
    test_indices       = test_indices,
    loss_hist          = np.array(classic_loss_hist),
    min_loss_hist      = np.array(classic_min_loss_hist[1:]),
    l2_hist            = np.array(classic_l2_hist),
    trunk_arch         = np.array(trunk_arch),
    branch_arch        = np.array(branch_arch),
)

print(f"Saved to {save_dir}")