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
import scipy.linalg
from sklearn.model_selection import train_test_split
import equinox as eqx
from jax.nn.initializers import he_normal
import time

jax.config.update("jax_platform_name", "gpu")

# ── Hyperparameters ────────────────────────────────────────────────────────────
num_trials          = config.get('num_trials', 20)
num_bases           = config.get('num_bases', 10)
num_trunk_epochs    = config.get('num_trunk_epochs', 6000)
lr_start            = config.get('lr_start', 1e-3)
lr_transition_steps = config.get('lr_transition_steps', 1000)
lr_decay_rate       = config.get('lr_decay_rate', 0.01)
lr_end              = config.get('lr_end', 1e-5)
num_LBFGS_epochs    = config.get('num_LBFGS_epochs', 1000)
test_size           = config.get('test_size', 0.2)

trunk_arch_base  = config.get('trunk_arch',  [2, 40, 40, 10])
branch_arch_base = config.get('branch_arch', [2, 60, 60, 10])

dataset_path = Path(config.get('dataset_path', 'burgers_dataset.npz'))
results_path = Path(config.get('results_path', 'trial_l2_errors.npz'))

# ── Dataset Loading ────────────────────────────────────────────────────────────
dataset = np.load(dataset_path, allow_pickle=True)
t_grid  = jnp.array(dataset['t'])
x_grid  = jnp.array(dataset['x'])
data    = dataset['samples']
u_all   = np.array([i['params']   for i in data])
s_all   = np.array([i['solution'] for i in data])  # (N, T, X)
n_samp  = len(data)

tt, xx  = jnp.meshgrid(t_grid, x_grid, indexing="ij")
tx_grid = jnp.stack([tt.reshape(-1), xx.reshape(-1)], axis=1)  # (T*X, 2)


# ── Model Definitions ──────────────────────────────────────────────────────────
class Linear(eqx.Module):
    weight: jax.Array
    bias:   jax.Array

    def __init__(self, in_size, out_size, key, initializer=he_normal()):
        wkey, _ = jax.random.split(key)
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


def make_lr_schedule():
    return optax.schedules.exponential_decay(
        init_value=lr_start,
        transition_steps=lr_transition_steps,
        decay_rate=lr_decay_rate,
        end_value=lr_end,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ORTH 2-STEP (time-independent QR)
# ══════════════════════════════════════════════════════════════════════════════

def train_orth(u_train, output_tr, u_test, s_test, key):
    trunk_arch  = trunk_arch_base[:]
    branch_arch = branch_arch_base[:]
    trunk_arch[-1]  = num_bases
    branch_arch[-1] = num_bases

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    key, subkey_t, subkey_A = jax.random.split(key, 3)
    trunk_model = MLP(trunk_arch, key=subkey_t)
    A_model     = jax.random.normal(subkey_A, (num_bases, output_tr.shape[2]))
    model       = trunk_model, A_model

    def loss_fn(model, inputs, labels):
        tm, am    = model
        T_MAT     = jax.vmap(tm)(inputs)
        pred_y_3d = (T_MAT @ am).reshape(len(t_grid), len(x_grid), -1)
        return jnp.mean((labels - pred_y_3d) ** 2)

    opt       = optax.adam(make_lr_schedule())
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def train_step(model, opt_state):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, tx_grid, output_tr)
        updates, opt_state = opt.update(grads, opt_state, model)
        return eqx.apply_updates(model, updates), opt_state, loss

    best_model  = model
    best_loss   = np.inf

    for step in range(num_trunk_epochs):
        model, opt_state, loss = train_step(model, opt_state)
        if float(loss) < best_loss:
            best_loss  = float(loss)
            best_model = model
        if step % 10000 == 0:
            print(f"  step {step}/{num_trunk_epochs}  loss={float(loss):.4e}"); import sys; sys.stdout.flush()
    model = best_model

    params = eqx.filter(model, eqx.is_inexact_array)
    static = jax.tree_util.tree_map(lambda x: None if eqx.is_inexact_array(x) else x, model)
    frozen_loss = jax.jit(lambda p: loss_fn(eqx.combine(p, static), tx_grid, output_tr))
    _ = frozen_loss(params); jax.block_until_ready(_)
    lbfgs = LBFGS(fun=frozen_loss, maxiter=num_LBFGS_epochs, tol=1e-9, history_size=20, implicit_diff=True, stepsize=-1.0)
    params, lbfgs_state = lbfgs.run(params)
    model = eqx.combine(params, static)

    # ── QR ────────────────────────────────────────────────────────────────────
    trunk_model, A_model = model
    T_MAT = jax.vmap(trunk_model)(tx_grid)
    Q_MAT, R_MAT = scipy.linalg.qr(np.asarray(T_MAT), mode="economic")
    Q_MAT = jnp.asarray(Q_MAT)
    R_MAT = jnp.asarray(R_MAT)

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    RA_target = jnp.asarray((R_MAT @ A_model).T)  # (N, K)

    key, subkey_b = jax.random.split(key)
    branch_model  = MLP(branch_arch, key=subkey_b)

    def branch_loss_fn(branch_model, RA_target, u_train):
        B_MAT = jax.vmap(branch_model)(u_train)
        return jnp.mean((B_MAT - RA_target) ** 2)

    opt_b       = optax.adam(make_lr_schedule())
    opt_state_b = opt_b.init(eqx.filter(branch_model, eqx.is_inexact_array))

    @eqx.filter_jit
    def branch_step(branch_model, opt_state):
        loss, grads = eqx.filter_value_and_grad(branch_loss_fn)(branch_model, RA_target, u_train)
        updates, opt_state = opt_b.update(grads, opt_state, branch_model)
        return eqx.apply_updates(branch_model, updates), opt_state, loss

    best_branch = branch_model
    best_loss   = np.inf
    for step in range(num_trunk_epochs):
        branch_model, opt_state_b, loss = branch_step(branch_model, opt_state_b)
        if float(loss) < best_loss:
            best_loss   = float(loss)
            best_branch = branch_model
        if step % 10000 == 0:
            print(f"  step {step}/{num_trunk_epochs}  loss={float(loss):.4e}"); import sys; sys.stdout.flush()
    branch_model = best_branch

    # ── Test L2 ───────────────────────────────────────────────────────────────
    def predict(u):
        b = branch_model(u)
        return (Q_MAT @ b).reshape(len(t_grid), len(x_grid))

    preds = jax.vmap(predict)(u_test)          # (N_test, T, X)
    diff  = preds - s_test
    l2_per_sample = jnp.linalg.norm(diff.reshape(diff.shape[0], -1), axis=1)  # (N_test,)
    return float(jnp.mean(l2_per_sample))


# ══════════════════════════════════════════════════════════════════════════════
# ORTH TD 2-STEP (time-dependent QR)
# ══════════════════════════════════════════════════════════════════════════════

def train_orth_td(u_train, output_tr, u_test, s_test, key):
    trunk_arch  = trunk_arch_base[:]
    branch_arch = branch_arch_base[:]
    trunk_arch[-1]  = num_bases
    branch_arch[-1] = num_bases
    branch_arch[0]  = branch_arch[0] + 1

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    key, subkey_t, subkey_A = jax.random.split(key, 3)
    trunk_model = MLP(trunk_arch, key=subkey_t)
    A_model     = jax.random.normal(subkey_A, (len(t_grid), num_bases, output_tr.shape[2]))
    model       = trunk_model, A_model

    def loss_fn(model):
        tm, am = model
        T_MAT  = jax.vmap(lambda t: jax.vmap(lambda x: tm(jnp.stack([t, x])))(x_grid))(t_grid)
        pred_y = jnp.einsum("txk,tkn->txn", T_MAT, am)
        return jnp.mean((pred_y - output_tr) ** 2)

    opt       = optax.adam(make_lr_schedule())
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def train_step(model, opt_state):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = opt.update(grads, opt_state, model)
        return eqx.apply_updates(model, updates), opt_state, loss

    best_model = model 
    best_loss  = np.inf
    for step in range(num_trunk_epochs):
        model, opt_state, loss = train_step(model, opt_state)
        if float(loss) < best_loss:
            best_loss  = float(loss)
            best_model = model
        if step % 10000 == 0:
            print(f"  step {step}/{num_trunk_epochs}  loss={float(loss):.4e}"); import sys; sys.stdout.flush()
    model = best_model

    params = eqx.filter(model, eqx.is_inexact_array)
    static = jax.tree_util.tree_map(lambda x: None if eqx.is_inexact_array(x) else x, model)
    frozen_loss = jax.jit(lambda p: loss_fn(eqx.combine(p, static)))
    _ = frozen_loss(params); jax.block_until_ready(_)
    lbfgs = LBFGS(fun=frozen_loss, maxiter=num_LBFGS_epochs, tol=1e-9, history_size=20, implicit_diff=True, stepsize=-1.0)
    params, lbfgs_state = lbfgs.run(params)
    model = eqx.combine(params, static)

    # ── QR with sign correction ────────────────────────────────────────────────
    trunk_model, A_model = model
    T_MAT = jax.vmap(lambda t: jax.vmap(lambda x: trunk_model(jnp.stack([t, x])))(x_grid))(t_grid)

    Q_list, R_list = [], []
    for i in range(len(t_grid)):
        Q_i, R_i = scipy.linalg.qr(np.asarray(T_MAT[i]), mode='economic')
        Q_list.append(Q_i)
        R_list.append(R_i)
    Q_sign = np.stack(Q_list, axis=0)
    R_sign = np.stack(R_list, axis=0)
    for k in range(1, len(t_grid)):
        for j in range(num_bases):
            if np.dot(Q_sign[k-1, :, j], Q_sign[k, :, j]) < 0:
                Q_sign[k, :, j] *= -1
                R_sign[k, j, :] *= -1

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    RA_target = jnp.asarray(jnp.einsum("tij,tjk->tik", R_sign, A_model))  # (T, K, N)

    key, subkey_b = jax.random.split(key)
    branch_model  = MLP(branch_arch, key=subkey_b)

    def branch_loss_fn(branch_model, RA_target, u_train, t_grid):
        B_tnK = jax.vmap(
            lambda t: jax.vmap(lambda u: branch_model(jnp.concatenate([u, jnp.array([t])])))(u_train)
        )(t_grid)
        B_MAT = jnp.swapaxes(B_tnK, 1, 2)
        return jnp.mean((B_MAT - RA_target) ** 2)

    opt_b       = optax.adam(make_lr_schedule())
    opt_state_b = opt_b.init(eqx.filter(branch_model, eqx.is_inexact_array))

    @eqx.filter_jit
    def branch_step(branch_model, opt_state):
        loss, grads = eqx.filter_value_and_grad(branch_loss_fn)(branch_model, RA_target, u_train, t_grid)
        updates, opt_state = opt_b.update(grads, opt_state, branch_model)
        return eqx.apply_updates(branch_model, updates), opt_state, loss

    best_branch = branch_model
    best_loss   = np.inf
    for step in range(num_trunk_epochs):
        branch_model, opt_state_b, loss = branch_step(branch_model, opt_state_b)
        if float(loss) < best_loss:
            best_loss   = float(loss)
            best_branch = branch_model
        if step % 10000 == 0:
            print(f"  step {step}/{num_trunk_epochs}  loss={float(loss):.4e}"); import sys; sys.stdout.flush()
    branch_model = best_branch

    # ── Test L2 ───────────────────────────────────────────────────────────────
    Q_sign_jax = jnp.asarray(Q_sign)

    def predict(u):
        b_tk = jax.vmap(lambda t: branch_model(jnp.concatenate([u, jnp.array([t])])))(t_grid)
        return jnp.einsum("txk,tk->tx", Q_sign_jax, b_tk)

    preds = jax.vmap(predict)(u_test)
    diff  = preds - s_test
    l2_per_sample = jnp.linalg.norm(diff.reshape(diff.shape[0], -1), axis=1)
    return float(jnp.mean(l2_per_sample))


# ══════════════════════════════════════════════════════════════════════════════
# TRIAL LOOP
# ══════════════════════════════════════════════════════════════════════════════

orth_l2_errors    = []
orth_td_l2_errors = []

master_key = jax.random.key(0)

for trial in range(num_trials):
    master_key, trial_key = jax.random.split(master_key)
    seed_trial = int(jax.random.randint(trial_key, (), 0, 2**31 - 1))

    print(f"\n{'='*60}")
    print(f"Trial {trial+1}/{num_trials}  seed={seed_trial}")
    print(f"{'='*60}")

    np.random.seed(seed_trial)
    train_indices, test_indices = train_test_split(
        np.arange(n_samp), test_size=test_size, random_state=seed_trial
    )
    u_train = jnp.array(u_all[train_indices])
    u_test  = jnp.array(u_all[test_indices])
    s_train = jnp.array(s_all[train_indices])
    s_test  = jnp.array(s_all[test_indices])
    output_tr = jnp.transpose(s_train, axes=(1, 2, 0))  # (T, X, N)

    key_orth, key_td = jax.random.split(trial_key)

    t0 = time.time()
    l2_orth = train_orth(u_train, output_tr, u_test, s_test, key_orth)
    print(f"Orth 2-Step    mean test L2: {l2_orth:.4e}  ({time.time()-t0:.1f}s)")
    orth_l2_errors.append(l2_orth)

    t0 = time.time()
    l2_td = train_orth_td(u_train, output_tr, u_test, s_test, key_td)
    print(f"Orth TD 2-Step mean test L2: {l2_td:.4e}  ({time.time()-t0:.1f}s)")
    orth_td_l2_errors.append(l2_td)

    # save incrementally after each trial so progress isn't lost
    np.savez(
        results_path,
        orth_l2_errors=np.array(orth_l2_errors),
        orth_td_l2_errors=np.array(orth_td_l2_errors),
    )
    print(f"saved to {results_path}")

print("\n\nAll trials complete.")
print(f"Orth 2-Step    mean={np.mean(orth_l2_errors):.4e}  std={np.std(orth_l2_errors):.4e}")
print(f"Orth TD 2-Step mean={np.mean(orth_td_l2_errors):.4e}  std={np.std(orth_td_l2_errors):.4e}")