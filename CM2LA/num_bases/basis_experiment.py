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
import scipy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import equinox as eqx
import scipy.linalg
from jax.nn.initializers import he_normal
import time


master_start_time = time.time()

seed = config.get('seed', 42)
num_trunk_epochs = config.get('num_trunk_epochs', 6000)
lr_start = config.get('lr_start', 1e-3)
lr_transition_steps = config.get('lr_transition_steps', 1000)
lr_decay_rate = config.get('lr_decay_rate', 0.01)
lr_end = config.get('lr_end', 1e-5)
eps = config.get('eps', 1e-8)

dataset_path = Path(config.get('dataset_path', '/Users/haydenoutlaw/Documents/Research/opnet/deep-operator-networks/data/burgers_dataset.npz'))

trunk_arch_base = config.get('trunk_arch', [2, 40, 40, 10])
branch_arch_base = config.get('branch_arch', [2, 60, 60, 10])

test_size = config.get('test_size', 0.2)
log_frequency = config.get('log_frequency', 10)

np.random.seed(seed)
key = jax.random.key(seed)

print("\nconfiguring backend...")
jax.config.update("jax_platform_name", "gpu")
print("backend selected:\n", jax.default_backend())
print("active devices:\n", jax.devices())
print("--------------------\n")


## SHARED CLASS DEFINITIONS

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

    def __init__(self, architecture, key, activation=jax.nn.relu, initializer=he_normal()):
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


## SHARED DATA LOADING

dataset = np.load(dataset_path, allow_pickle=True)
t_grid = jnp.array(dataset['t'])
x_grid = jnp.array(dataset['x'])

data = dataset['samples']
u = np.array([i['params'] for i in data])
s = np.array([i['solution'] for i in data])  # (n, t, x)

n_samp = len(data)
train_indices, test_indices = train_test_split(np.arange(n_samp), test_size=test_size, random_state=seed)
u_train, u_test = jnp.array(u[train_indices]), jnp.array(u[test_indices])
s_train, s_test = jnp.array(s[train_indices]), jnp.array(s[test_indices])

output_tr = jnp.transpose(s_train, axes=(1, 2, 0))   # (T, X, N)
output_test = jnp.transpose(s_test, axes=(1, 2, 0))

tt, xx = jnp.meshgrid(t_grid, x_grid, indexing="ij")
tx_grid = jnp.stack([tt.reshape(-1), xx.reshape(-1)], axis=1)  # (T*X, 2)

lr_schedule = optax.schedules.exponential_decay(
    init_value=lr_start,
    transition_steps=lr_transition_steps,
    decay_rate=lr_decay_rate,
    end_value=lr_end if lr_end is not None else None
)


## SHARED TRAINING LOOP

def run_training_loop(model, loss_fn, num_epochs, log_tag):
    opt = optax.adam(lr_schedule)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
    loss_hist = []
    min_loss_hist = [np.inf]
    best_model = model

    @eqx.filter_jit
    def train_step(model, opt_state):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = opt.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    s_time = time.time()
    start_time = time.time()
    for step in range(num_epochs):
        model, opt_state, loss = train_step(model, opt_state)
        loss_hist.append(float(loss))
        if float(loss) < min_loss_hist[-1]:
            min_loss_hist.append(float(loss))
            best_model = model
        else:
            min_loss_hist.append(min_loss_hist[-1])
        if step % 10000 == 0:
            end_time = time.time()
            print(f"\r[{log_tag}] Adam step {step}: loss={float(loss):.3e}, {end_time-start_time:.2f}s", end="", flush=True)
            start_time = time.time()
    e_time = time.time()
    print(f"\n[{log_tag}] final adam loss: {float(loss):.3e}, total time: {e_time-s_time:.2f}s\n")
    return best_model, loss_hist, min_loss_hist


def compute_qr_sign_corrected(T_MAT_3D, t_grid):
    """T_MAT_3D: (T, X, K) -> Q_sign (T, X, K), R_sign (T, K, K)"""
    Q_list, R_list = [], []
    for i in range(len(t_grid)):
        Q, R = scipy.linalg.qr(T_MAT_3D[i], mode='economic')
        Q_list.append(Q)
        R_list.append(R)
    Q_sign = np.stack(Q_list, axis=0)
    R_sign = np.stack(R_list, axis=0)
    T = Q_sign.shape[0]
    n = Q_sign.shape[2]
    for k in range(1, T):
        for j in range(n):
            if np.dot(Q_sign[k-1, :, j], Q_sign[k, :, j]) < 0:
                Q_sign[k, :, j] *= -1
                R_sign[k, j, :] *= -1
    return Q_sign, R_sign


## SWEEP OVER NUM_BASES

bases_range = range(5, 0, -1)
results = {
    'classic': {},
    'orth': {},
    'orth_td': {},
}

stored_models = {}  # num_bases -> {'classic': model, 'orth': (Q, branch), 'td': (Q, branch)}

for num_bases in bases_range:
    print("\n" + "=" * 60)
    print(f"NUM_BASES = {num_bases}")
    print("=" * 60)

    trunk_arch = list(trunk_arch_base)
    branch_arch = list(branch_arch_base)
    trunk_arch[-1] = num_bases
    branch_arch[-1] = num_bases

    ###############################################################################
    ## CLASSIC
    ###############################################################################
    print("=" * 60)
    print("TRAINING: CLASSIC")
    print("=" * 60)

    key, subkey_t, subkey_b = jax.random.split(key, num=3)
    classic_trunk = MLP(trunk_arch, key=subkey_t)
    classic_branch = MLP(branch_arch, key=subkey_b)
    classic_model = classic_trunk, classic_branch

    def classic_loss_fn(model):
        trunk_model, branch_model = model
        T_MAT = jax.vmap(trunk_model)(tx_grid)
        B_MAT = jax.vmap(branch_model)(u_train)
        pred_y = T_MAT @ B_MAT.T
        pred_y_3d = pred_y.reshape(len(t_grid), len(x_grid), -1)
        diff = output_tr - pred_y_3d
        diff_flat = diff.reshape(-1, diff.shape[-1])
        labels_flat = output_tr.reshape(-1, output_tr.shape[-1])
        return jnp.mean(jnp.linalg.norm(diff_flat, axis=0) /
                        (jnp.linalg.norm(labels_flat, axis=0) + eps))

    classic_model, classic_loss_hist, classic_min_loss_hist = run_training_loop(
        classic_model, classic_loss_fn, num_trunk_epochs, "classic"
    )

    def predict_classic(model, u, tx_grid, t_grid, x_grid):
        trunk_model, branch_model = model
        T_MAT = jax.vmap(trunk_model)(tx_grid)
        b = branch_model(u)
        return (T_MAT @ b).reshape(len(t_grid), len(x_grid))

    def test_rel_l2_classic(model, u_test, s_test):
        preds = jax.vmap(lambda u: predict_classic(model, u, tx_grid, t_grid, x_grid))(u_test)
        diff = preds - s_test
        return jnp.mean(jnp.linalg.norm(diff.reshape(diff.shape[0], -1), axis=-1) /
                        (jnp.linalg.norm(s_test.reshape(s_test.shape[0], -1), axis=-1) + eps))

    classic_test_rel_l2 = test_rel_l2_classic(classic_model, u_test, s_test)
    print(f"\n[classic] Average Test Rel L2: {float(classic_test_rel_l2):.6e}\n")
    results['classic'][num_bases] = float(classic_test_rel_l2)

    ###############################################################################
    ## ORTH (global QR, 2-step)
    ###############################################################################
    print("=" * 60)
    print("TRAINING: ORTH (global QR)")
    print("=" * 60)

    key, subkey_t, subkey_A = jax.random.split(key, num=3)
    orth_trunk = MLP(trunk_arch, key=subkey_t)
    orth_A = jax.random.normal(subkey_A, (num_bases, output_tr.shape[2]))
    orth_model = orth_trunk, orth_A

    def orth_trunk_loss_fn(model):
        trunk_model, A_model = model
        T_MAT = jax.vmap(trunk_model)(tx_grid)
        pred_y = T_MAT @ A_model
        pred_y_3d = pred_y.reshape(len(t_grid), len(x_grid), -1)
        diff = output_tr - pred_y_3d
        diff_flat = diff.reshape(-1, diff.shape[-1])
        labels_flat = output_tr.reshape(-1, output_tr.shape[-1])
        return jnp.mean(jnp.linalg.norm(diff_flat, axis=0) /
                        (jnp.linalg.norm(labels_flat, axis=0) + eps))

    orth_model, orth_loss_hist, orth_min_loss_hist = run_training_loop(
        orth_model, orth_trunk_loss_fn, num_trunk_epochs, "orth-trunk"
    )

    orth_trunk_model, orth_A_model = orth_model
    orth_T_MAT = jax.vmap(orth_trunk_model)(tx_grid)
    orth_T_host = np.asarray(orth_T_MAT)
    orth_Q, orth_R = scipy.linalg.qr(orth_T_host, mode='economic')
    orth_Q = jnp.asarray(orth_Q)
    orth_R = jnp.asarray(orth_R)

    key, subkey_b = jax.random.split(key)
    orth_branch = MLP(branch_arch, key=subkey_b)
    orth_RA_target = (orth_R @ orth_A_model).T  # (N, K)

    def orth_branch_loss_fn(branch_model):
        B_MAT = jax.vmap(branch_model)(u_train)  # (N, K)
        diff = B_MAT - orth_RA_target
        return jnp.mean(jnp.linalg.norm(diff, axis=-1) /
                        (jnp.linalg.norm(orth_RA_target, axis=-1) + eps))

    orth_branch, orth_branch_loss_hist, orth_branch_min_loss_hist = run_training_loop(
        orth_branch, orth_branch_loss_fn, num_trunk_epochs, "orth-branch"
    )

    def predict_orth(Q, branch_model, u):
        b = branch_model(u)
        return (Q @ b).reshape(len(t_grid), len(x_grid))

    def test_rel_l2_orth(Q, branch_model, u_test, s_test):
        preds = jax.vmap(lambda u: predict_orth(Q, branch_model, u))(u_test)
        diff = preds - s_test
        return jnp.mean(jnp.linalg.norm(diff.reshape(diff.shape[0], -1), axis=-1) /
                        (jnp.linalg.norm(s_test.reshape(s_test.shape[0], -1), axis=-1) + eps))

    orth_test_rel_l2 = test_rel_l2_orth(orth_Q, orth_branch, u_test, s_test)
    print(f"\n[orth] Average Test Rel L2 (Q @ b(u)): {float(orth_test_rel_l2):.6e}\n")
    results['orth'][num_bases] = float(orth_test_rel_l2)

    ###############################################################################
    ## ORTH_TD (time-dependent QR, 2-step)
    ###############################################################################
    print("=" * 60)
    print("TRAINING: ORTH_TD (time-dependent QR)")
    print("=" * 60)

    td_branch_arch = list(branch_arch)
    td_branch_arch[0] = td_branch_arch[0] + 1  # extra t input

    key, subkey_t, subkey_A = jax.random.split(key, num=3)
    td_trunk = MLP(trunk_arch, key=subkey_t)
    td_A = jax.random.normal(subkey_A, (len(t_grid), num_bases, output_tr.shape[2]))  # (T, K, N)
    td_model = td_trunk, td_A

    def td_trunk_loss_fn(model):
        trunk_model, A_model = model
        T_MAT = jax.vmap(lambda t: jax.vmap(lambda x: trunk_model(jnp.stack([x, t])))(x_grid))(t_grid)
        pred_y = jnp.einsum("txk,tkn->txn", T_MAT, A_model)
        diff = output_tr - pred_y
        diff_flat = diff.reshape(-1, diff.shape[-1])
        labels_flat = output_tr.reshape(-1, output_tr.shape[-1])
        return jnp.mean(jnp.linalg.norm(diff_flat, axis=0) /
                        (jnp.linalg.norm(labels_flat, axis=0) + eps))

    td_model, td_loss_hist, td_min_loss_hist = run_training_loop(
        td_model, td_trunk_loss_fn, num_trunk_epochs, "orth_td-trunk"
    )

    td_trunk_model, td_A_model = td_model
    td_T_MAT_3D = np.asarray(
        jax.vmap(lambda t: jax.vmap(lambda x: td_trunk_model(jnp.stack([x, t])))(x_grid))(t_grid)
    )  # (T, X, K)
    td_Q_sign, td_R_sign = compute_qr_sign_corrected(td_T_MAT_3D, t_grid)

    td_RA_target = jnp.einsum("tij,tjk->tik", td_R_sign, td_A_model)  # (T, K, N)

    key, subkey_b = jax.random.split(key)
    td_branch = MLP(td_branch_arch, key=subkey_b)

    def td_branch_loss_fn(branch_model):
        B_tnK = jax.vmap(
            lambda t: jax.vmap(
                lambda u: branch_model(jnp.concatenate([u, jnp.array([t])]))
            )(u_train)
        )(t_grid)  # (T, N, K)
        B_MAT = jnp.swapaxes(B_tnK, 1, 2)  # (T, K, N)
        diff = B_MAT - td_RA_target
        diff_flat = diff.reshape(-1, diff.shape[-1])
        target_flat = td_RA_target.reshape(-1, td_RA_target.shape[-1])
        return jnp.mean(jnp.linalg.norm(diff_flat, axis=0) /
                        (jnp.linalg.norm(target_flat, axis=0) + eps))

    td_branch, td_branch_loss_hist, td_branch_min_loss_hist = run_training_loop(
        td_branch, td_branch_loss_fn, num_trunk_epochs, "orth_td-branch"
    )

    def predict_td(Q_sign, branch_model, u, t_grid):
        b_tk = jax.vmap(lambda t: branch_model(jnp.concatenate([u, jnp.array([t])])))(t_grid)  # (T, K)
        return jnp.einsum("txk,tk->tx", Q_sign, b_tk)  # (T, X)

    def test_rel_l2_td(Q_sign, branch_model, u_test, s_test, t_grid):
        preds = jax.vmap(lambda u: predict_td(Q_sign, branch_model, u, t_grid))(u_test)
        diff = preds - s_test
        return jnp.mean(jnp.linalg.norm(diff.reshape(diff.shape[0], -1), axis=-1) /
                        (jnp.linalg.norm(s_test.reshape(s_test.shape[0], -1), axis=-1) + eps))

    td_Q_jax = jnp.asarray(td_Q_sign)
    td_test_rel_l2 = test_rel_l2_td(td_Q_jax, td_branch, u_test, s_test, t_grid)
    print(f"\n[orth_td] Average Test Rel L2 (time-dep Q @ b(t,u)): {float(td_test_rel_l2):.6e}\n")
    results['orth_td'][num_bases] = float(td_test_rel_l2)

    print(f"\n[num_bases={num_bases}] classic={float(classic_test_rel_l2):.6e}  orth={float(orth_test_rel_l2):.6e}  orth_td={float(td_test_rel_l2):.6e}")


    stored_models[num_bases] = {
        'classic': classic_model,
        'orth': (orth_Q, orth_branch),
        'td': (td_Q_jax, td_branch),
    }
###############################################################################
## COMPARATIVE SUMMARY
###############################################################################
print("\n" + "=" * 60)
print("SWEEP RESULTS")
print("=" * 60)
for nb in bases_range:
    print(f"  num_bases={nb}  classic={results['classic'][nb]:.6e}  orth={results['orth'][nb]:.6e}  orth_td={results['orth_td'][nb]:.6e}")
print("=" * 60)

x_vals = list(bases_range)
plt.figure()
plt.plot(x_vals, [results['classic'][nb] for nb in x_vals], marker='o', label='classic')
plt.plot(x_vals, [results['orth'][nb] for nb in x_vals], marker='o', label='orth')
plt.plot(x_vals, [results['orth_td'][nb] for nb in x_vals], marker='o', label='orth_td')
plt.xlabel("num_bases")
plt.ylabel("mean test relative L2 error")
plt.title("Test Rel L2 vs num_bases")
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("sweep_rel_l2_vs_num_bases.png")
plt.close()

###############################################################################
## PER-TIMESTEP RELATIVE L2 ERROR
###############################################################################

def per_time_rel_l2_classic(model, u_test, s_test):
    trunk_model, branch_model = model
    T_MAT = jax.vmap(trunk_model)(tx_grid)
    preds = jax.vmap(lambda u: (T_MAT @ branch_model(u)).reshape(len(t_grid), len(x_grid)))(u_test)
    diff = preds - s_test
    numer = jnp.linalg.norm(diff, axis=-1)
    denom = jnp.linalg.norm(s_test, axis=-1) + eps
    return jnp.mean(numer / denom, axis=0)  # (T,)

def per_time_rel_l2_orth(Q, branch_model, u_test, s_test):
    preds = jax.vmap(lambda u: (Q @ branch_model(u)).reshape(len(t_grid), len(x_grid)))(u_test)
    diff = preds - s_test
    numer = jnp.linalg.norm(diff, axis=-1)
    denom = jnp.linalg.norm(s_test, axis=-1) + eps
    return jnp.mean(numer / denom, axis=0)

def per_time_rel_l2_td(Q_sign, branch_model, u_test, s_test):
    preds = jax.vmap(lambda u: predict_td(Q_sign, branch_model, u, t_grid))(u_test)
    diff = preds - s_test
    numer = jnp.linalg.norm(diff, axis=-1)
    denom = jnp.linalg.norm(s_test, axis=-1) + eps
    return jnp.mean(numer / denom, axis=0)

t_np = np.asarray(t_grid)
bases_list = list(bases_range)
n_bases = len(bases_list)

fig, axes = plt.subplots(1, n_bases, figsize=(4 * n_bases, 4), sharey=True)
if n_bases == 1:
    axes = [axes]

for ax, nb in zip(axes, bases_list):
    m = stored_models[nb]
    err_classic = per_time_rel_l2_classic(m['classic'], u_test, s_test)
    err_orth    = per_time_rel_l2_orth(m['orth'][0], m['orth'][1], u_test, s_test)
    err_td      = per_time_rel_l2_td(m['td'][0], m['td'][1], u_test, s_test)
    ax.semilogy(t_np, np.asarray(err_classic), label='classic')
    ax.semilogy(t_np, np.asarray(err_orth),    label='orth')
    ax.semilogy(t_np, np.asarray(err_td),      label='orth_td')
    ax.set_title(f'num_bases={nb}')
    ax.set_xlabel('t')
    ax.grid(True, alpha=0.3)
    if ax is axes[0]:
        ax.set_ylabel('mean test relative L2 error')

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
fig.suptitle('Per-Timestep Relative L2 Error by Architecture and num_bases')
plt.tight_layout()
plt.savefig("per_time_rel_l2.png")
plt.close()

master_end_time = time.time()
print("Total Time (sec):", master_end_time - master_start_time)