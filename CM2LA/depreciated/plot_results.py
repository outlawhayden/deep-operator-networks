## using envml conda environment

import os
import yaml
import sys
from pathlib import Path

# Load all three configs to find save_dirs
config_paths = {
    'classic': Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config_classic.yaml"),
    'orth':    Path(sys.argv[2]) if len(sys.argv) > 2 else Path("config_orth.yaml"),
    'td':      Path(sys.argv[3]) if len(sys.argv) > 3 else Path("config_td.yaml"),
}
configs = {}
for name, cp in config_paths.items():
    with open(cp, 'r') as f:
        configs[name] = yaml.safe_load(f)

gpu_idx = configs['classic'].get("gpu_idx", 8)
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

import jax.numpy as jnp
import numpy as np
import jax
import equinox as eqx
from jax.nn.initializers import he_normal
import matplotlib.pyplot as plt

plot_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("plots")
plot_dir.mkdir(parents=True, exist_ok=True)

print("\nconfiguring backend...")
jax.config.update("jax_platform_name", "METAL")
print("backend selected:\n", jax.default_backend())
print("active devices:\n", jax.devices())
print("--------------------\n")


# ── Model Definitions (needed to deserialise weights) ─────────────────────────
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


# ── Load Classic ──────────────────────────────────────────────────────────────
classic_dir  = Path(configs['classic']['save_dir'])
classic_data = np.load(classic_dir / "data.npz")

t_grid  = jnp.array(classic_data['t_grid'])
x_grid  = jnp.array(classic_data['x_grid'])
tx_grid = jnp.array(classic_data['tx_grid'])
u_train = jnp.array(classic_data['u_train'])
u_test  = jnp.array(classic_data['u_test'])
s_train = jnp.array(classic_data['s_train'])
s_test  = jnp.array(classic_data['s_test'])

output_tr   = jnp.transpose(s_train, axes=(1, 2, 0))
output_test = jnp.transpose(s_test,  axes=(1, 2, 0))

classic_trunk_arch  = classic_data['trunk_arch'].tolist()
classic_branch_arch = classic_data['branch_arch'].tolist()

key = jax.random.key(configs['classic'].get('seed', 42))
key, sk_t, sk_b = jax.random.split(key, 3)
classic_trunk  = eqx.tree_deserialise_leaves(classic_dir / "trunk.eqx",  MLP(classic_trunk_arch,  key=sk_t))
classic_branch = eqx.tree_deserialise_leaves(classic_dir / "branch.eqx", MLP(classic_branch_arch, key=sk_b))
classic_model  = classic_trunk, classic_branch

classic_loss_hist     = classic_data['loss_hist']
classic_min_loss_hist = classic_data['min_loss_hist']


# ── Load Orth ─────────────────────────────────────────────────────────────────
orth_dir  = Path(configs['orth']['save_dir'])
orth_data = np.load(orth_dir / "data.npz")

orth_trunk_arch  = orth_data['trunk_arch'].tolist()
orth_branch_arch = orth_data['branch_arch'].tolist()
Q_orth           = jnp.array(orth_data['Q_orth'])

key, sk_t, sk_b = jax.random.split(key, 3)
orth_trunk  = eqx.tree_deserialise_leaves(orth_dir / "trunk.eqx",  MLP(orth_trunk_arch,  key=sk_t))
orth_branch = eqx.tree_deserialise_leaves(orth_dir / "branch.eqx", MLP(orth_branch_arch, key=sk_b))

orth_trunk_loss_hist     = orth_data['trunk_loss_hist']
orth_trunk_min_loss_hist = orth_data['trunk_min_loss_hist']
orth_branch_loss_hist    = orth_data['branch_loss_hist']
orth_branch_min_loss_hist = orth_data['branch_min_loss_hist']


# ── Load TD ───────────────────────────────────────────────────────────────────
td_dir  = Path(configs['td']['save_dir'])
td_data = np.load(td_dir / "data.npz")

td_trunk_arch  = td_data['trunk_arch'].tolist()
td_branch_arch = td_data['branch_arch'].tolist()
Q_sign         = td_data['Q_sign']
num_bases      = int(td_data['num_bases'])

key, sk_t, sk_b = jax.random.split(key, 3)
td_trunk  = eqx.tree_deserialise_leaves(td_dir / "trunk.eqx",  MLP(td_trunk_arch,  key=sk_t))
td_branch = eqx.tree_deserialise_leaves(td_dir / "branch.eqx", MLP(td_branch_arch, key=sk_b))

td_trunk_loss_hist      = td_data['trunk_loss_hist']
td_trunk_min_loss_hist  = td_data['trunk_min_loss_hist']
td_branch_loss_hist     = td_data['branch_loss_hist']
td_branch_min_loss_hist = td_data['branch_min_loss_hist']


# ── Inference Helpers ─────────────────────────────────────────────────────────
def classic_predict(model, u):
    trunk_model, branch_model = model
    T_MAT = jax.vmap(trunk_model)(tx_grid)
    b     = branch_model(u)
    return (T_MAT @ b).reshape(len(t_grid), len(x_grid))


def orth_predict(Q_MAT, branch_model, u):
    b = branch_model(u)
    return (Q_MAT @ b).reshape(len(t_grid), len(x_grid))


def td_predict(Q_sign, branch_model, u):
    b_tk = jax.vmap(lambda t: branch_model(jnp.concatenate([u, jnp.array([t])])))(t_grid)
    return jnp.einsum("txk,tk->tx", Q_sign, b_tk)


# ── Shared test/train samples ─────────────────────────────────────────────────
u_rand,    s_rand    = u_test[20],  s_test[20]
u_rand_tr, s_rand_tr = u_train[20], s_train[20]

pred_classic    = classic_predict(classic_model, u_rand)
pred_orth       = orth_predict(Q_orth, orth_branch, u_rand)
pred_td         = td_predict(jnp.asarray(Q_sign), td_branch, u_rand)

pred_classic_tr = classic_predict(classic_model, u_rand_tr)
pred_orth_tr    = orth_predict(Q_orth, orth_branch, u_rand_tr)
pred_td_tr      = td_predict(jnp.asarray(Q_sign), td_branch, u_rand_tr)

err_classic    = pred_classic    - s_rand
err_orth       = pred_orth       - s_rand
err_td         = pred_td         - s_rand
err_classic_tr = pred_classic_tr - s_rand_tr
err_orth_tr    = pred_orth_tr    - s_rand_tr
err_td_tr      = pred_td_tr      - s_rand_tr

extent = [float(x_grid[0]), float(x_grid[-1]), float(t_grid[-1]), float(t_grid[0])]
T_mesh, X_mesh = np.meshgrid(t_grid, x_grid, indexing="ij")
num_trunk_epochs = len(classic_loss_hist)


# ── Plot 1: Trunk Training MSE ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
axes[0].plot(classic_loss_hist);          axes[0].plot(classic_min_loss_hist)
axes[0].set_title("Classic: Trunk/Joint MSE"); axes[0].set_yscale('log')
axes[0].set_xlabel("step");               axes[0].set_ylabel("MSE")

axes[1].plot(orth_trunk_loss_hist);       axes[1].plot(orth_trunk_min_loss_hist)
axes[1].set_title("Orth 2-Step: Stage 1 MSE"); axes[1].set_yscale('log')
axes[1].set_xlabel("step");               axes[1].set_ylabel("MSE")

axes[2].plot(td_trunk_loss_hist);         axes[2].plot(td_trunk_min_loss_hist)
axes[2].set_title("Orth TD 2-Step: Stage 1 MSE"); axes[2].set_yscale('log')
axes[2].set_xlabel("step");               axes[2].set_ylabel("MSE")

plt.tight_layout()
plt.savefig(plot_dir / "compare_trunk_mse.png")
plt.close()


# ── Plot 2: Branch Training MSE ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
axes[0].plot(classic_loss_hist);          axes[0].plot(classic_min_loss_hist)
axes[0].set_title("Classic: Joint MSE (no separate branch stage)"); axes[0].set_yscale('log')
axes[0].set_xlabel("step");               axes[0].set_ylabel("MSE")

axes[1].plot(orth_branch_loss_hist);      axes[1].plot(orth_branch_min_loss_hist)
axes[1].set_title("Orth 2-Step: Branch MSE"); axes[1].set_yscale('log')
axes[1].set_xlabel("step");               axes[1].set_ylabel("MSE")

axes[2].plot(td_branch_loss_hist);        axes[2].plot(td_branch_min_loss_hist)
axes[2].set_title("Orth TD 2-Step: Branch MSE"); axes[2].set_yscale('log')
axes[2].set_xlabel("step");               axes[2].set_ylabel("MSE")

plt.tight_layout()
plt.savefig(plot_dir / "compare_branch_mse.png")
plt.close()


# ── Plot 3: Pointwise error (test) ────────────────────────────────────────────
vmax = float(max(jnp.abs(err_classic).max(), jnp.abs(err_orth).max(), jnp.abs(err_td).max()))
vmin = -vmax

fig, axes = plt.subplots(1, 3, figsize=(18, 4))
axes[0].imshow(err_classic, extent=extent, aspect="auto", origin="upper", vmin=vmin, vmax=vmax, cmap='RdBu_r')
axes[0].set_title("Classic: Error"); axes[0].set_xlabel("x"); axes[0].set_ylabel("t")

axes[1].imshow(err_orth,    extent=extent, aspect="auto", origin="upper", vmin=vmin, vmax=vmax, cmap='RdBu_r')
axes[1].set_title("Orth 2-Step: Error"); axes[1].set_xlabel("x"); axes[1].set_ylabel("t")

im2 = axes[2].imshow(err_td, extent=extent, aspect="auto", origin="upper", vmin=vmin, vmax=vmax, cmap='RdBu_r')
axes[2].set_title("Orth TD 2-Step: Error"); axes[2].set_xlabel("x"); axes[2].set_ylabel("t")

fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
fig.colorbar(im2, cax=cbar_ax, label="pred - true")
plt.savefig(plot_dir / "compare_errors.png")
plt.close()


# ── Plot 4: First 4 basis functions ──────────────────────────────────────────
T_MAT_classic = jax.vmap(classic_trunk)(tx_grid)
bases_classic  = np.array(T_MAT_classic[:, :4].reshape(len(t_grid), len(x_grid), 4))
bases_orth     = np.array(Q_orth[:, :4].reshape(len(t_grid), len(x_grid), 4))
bases_td       = Q_sign[:, :, :4]

fig, axes = plt.subplots(3, 4, figsize=(20, 12), constrained_layout=True)
row_labels = ["Classic", "Orth 2-Step", "Orth TD 2-Step"]
row_bases  = [bases_classic, bases_orth, bases_td]

for row, (label, bases) in enumerate(zip(row_labels, row_bases)):
    bmax_row = float(np.abs(bases).max())
    for k in range(4):
        im = axes[row, k].contourf(T_mesh, X_mesh, bases[:, :, k], levels=100, cmap='plasma', vmin=0, vmax=bmax_row)
        axes[row, k].set_title(f"{label} — basis {k}", fontsize=10)
        axes[row, k].set_xlabel("t"); axes[row, k].set_ylabel("x")
    fig.colorbar(im, ax=axes[row, :].tolist(), shrink=0.6, label=f"{label} basis value")

fig.suptitle("First 4 Basis Functions by Model", fontsize=16)
plt.savefig(plot_dir / "compare_bases.png")
plt.close()


# ── Plot 5: First 4 basis functions — two-step only ──────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(20, 8), constrained_layout=True)
for row, (label, bases) in enumerate(zip(["Orth 2-Step", "Orth TD 2-Step"], [bases_orth, bases_td])):
    bmax_row = float(np.abs(bases).max())
    for k in range(4):
        im = axes[row, k].contourf(T_mesh, X_mesh, bases[:, :, k], levels=100, cmap='plasma', vmin=0, vmax=bmax_row)
        axes[row, k].set_title(f"{label} — basis {k}", fontsize=10)
        axes[row, k].set_xlabel("t"); axes[row, k].set_ylabel("x")
    fig.colorbar(im, ax=axes[row, :].tolist(), shrink=0.6, label=f"{label} basis value")

fig.suptitle("First 4 Basis Functions — Two-Step Models", fontsize=16)
plt.savefig(plot_dir / "compare_twostep_bases.png")
plt.close()


# ── Plot 6: Pointwise error — two-step only ───────────────────────────────────
vmax_ts = float(max(jnp.abs(err_orth).max(), jnp.abs(err_td).max()))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].imshow(err_orth, extent=extent, aspect="auto", origin="upper", vmin=-vmax_ts, vmax=vmax_ts, cmap='RdBu_r')
axes[0].set_title("Orth 2-Step: Error"); axes[0].set_xlabel("x"); axes[0].set_ylabel("t")

im1 = axes[1].imshow(err_td, extent=extent, aspect="auto", origin="upper", vmin=-vmax_ts, vmax=vmax_ts, cmap='RdBu_r')
axes[1].set_title("Orth TD 2-Step: Error"); axes[1].set_xlabel("x"); axes[1].set_ylabel("t")

fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
fig.colorbar(im1, cax=cbar_ax, label="pred - true")
plt.savefig(plot_dir / "compare_twostep_errors.png")
plt.close()


# ── Plot 3 (train): Pointwise error ───────────────────────────────────────────
vmax_tr = float(max(jnp.abs(err_classic_tr).max(), jnp.abs(err_orth_tr).max(), jnp.abs(err_td_tr).max()))

fig, axes = plt.subplots(1, 3, figsize=(18, 4))
axes[0].imshow(err_classic_tr, extent=extent, aspect="auto", origin="upper", vmin=-vmax_tr, vmax=vmax_tr, cmap='RdBu_r')
axes[0].set_title("Classic: Error (train)"); axes[0].set_xlabel("x"); axes[0].set_ylabel("t")

axes[1].imshow(err_orth_tr, extent=extent, aspect="auto", origin="upper", vmin=-vmax_tr, vmax=vmax_tr, cmap='RdBu_r')
axes[1].set_title("Orth 2-Step: Error (train)"); axes[1].set_xlabel("x"); axes[1].set_ylabel("t")

im2 = axes[2].imshow(err_td_tr, extent=extent, aspect="auto", origin="upper", vmin=-vmax_tr, vmax=vmax_tr, cmap='RdBu_r')
axes[2].set_title("Orth TD 2-Step: Error (train)"); axes[2].set_xlabel("x"); axes[2].set_ylabel("t")

fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
fig.colorbar(im2, cax=cbar_ax, label="pred - true")
plt.savefig(plot_dir / "compare_errors_train.png")
plt.close()


# ── Plot 4 (train): First 4 basis functions ───────────────────────────────────
fig, axes = plt.subplots(3, 4, figsize=(20, 12), constrained_layout=True)
for row, (label, bases) in enumerate(zip(
    ["Classic (train)", "Orth 2-Step (train)", "Orth TD 2-Step (train)"],
    [bases_classic, bases_orth, bases_td]
)):
    bmax_row = float(np.abs(bases).max())
    for k in range(4):
        im = axes[row, k].contourf(T_mesh, X_mesh, bases[:, :, k], levels=100, cmap='plasma', vmin=0, vmax=bmax_row)
        axes[row, k].set_title(f"{label} — basis {k}", fontsize=10)
        axes[row, k].set_xlabel("t"); axes[row, k].set_ylabel("x")
    fig.colorbar(im, ax=axes[row, :].tolist(), shrink=0.6, label=f"{label} basis value")

fig.suptitle("First 4 Basis Functions by Model (Train)", fontsize=16)
plt.savefig(plot_dir / "compare_bases_train.png")
plt.close()


# ── Plot 5 (train): Two-step basis functions ──────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(20, 8), constrained_layout=True)
for row, (label, bases) in enumerate(zip(
    ["Orth 2-Step (train)", "Orth TD 2-Step (train)"],
    [bases_orth, bases_td]
)):
    bmax_row = float(np.abs(bases).max())
    for k in range(4):
        im = axes[row, k].contourf(T_mesh, X_mesh, bases[:, :, k], levels=100, cmap='plasma', vmin=0, vmax=bmax_row)
        axes[row, k].set_title(f"{label} — basis {k}", fontsize=10)
        axes[row, k].set_xlabel("t"); axes[row, k].set_ylabel("x")
    fig.colorbar(im, ax=axes[row, :].tolist(), shrink=0.6, label=f"{label} basis value")

fig.suptitle("First 4 Basis Functions — Two-Step Models (Train)", fontsize=16)
plt.savefig(plot_dir / "compare_twostep_bases_train.png")
plt.close()


# ── Plot 6 (train): Two-step pointwise error ──────────────────────────────────
vmax_ts_tr = float(max(jnp.abs(err_orth_tr).max(), jnp.abs(err_td_tr).max()))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].imshow(err_orth_tr, extent=extent, aspect="auto", origin="upper", vmin=-vmax_ts_tr, vmax=vmax_ts_tr, cmap='RdBu_r')
axes[0].set_title("Orth 2-Step: Error (train)"); axes[0].set_xlabel("x"); axes[0].set_ylabel("t")

im1 = axes[1].imshow(err_td_tr, extent=extent, aspect="auto", origin="upper", vmin=-vmax_ts_tr, vmax=vmax_ts_tr, cmap='RdBu_r')
axes[1].set_title("Orth TD 2-Step: Error (train)"); axes[1].set_xlabel("x"); axes[1].set_ylabel("t")

fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
fig.colorbar(im1, cax=cbar_ax, label="pred - true")
plt.savefig(plot_dir / "compare_twostep_errors_train.png")
plt.close()


# ── Plot 7: Training history (trunk) ─────────────────────────────────────────
plt.figure(figsize=(12, 4))
plt.semilogy(range(num_trunk_epochs), classic_min_loss_hist,     label="Classic")
plt.semilogy(range(num_trunk_epochs), orth_trunk_min_loss_hist,  label="2-Step")
plt.semilogy(range(num_trunk_epochs), td_trunk_min_loss_hist,    label="Time Dep 2-Step")
plt.title("Min MSE Value across Training (Trunk Network)")
plt.xlabel("Epochs"); plt.ylabel("MSE"); plt.legend()
plt.savefig(plot_dir / "compare_twostep_training_history.png")
plt.close()

# ── Plot 7: Training history (branch) ────────────────────────────────────────
plt.figure(figsize=(12, 4))
plt.semilogy(range(num_trunk_epochs), classic_min_loss_hist,      label="Classic")
plt.semilogy(range(num_trunk_epochs), orth_branch_min_loss_hist,  label="2-Step")
plt.semilogy(range(num_trunk_epochs), td_branch_min_loss_hist,    label="Time Dep 2-Step")
plt.title("Min MSE Value across Training (Branch Network)")
plt.xlabel("Epochs"); plt.ylabel("MSE"); plt.legend()
plt.savefig(plot_dir / "compare_twostep_training_history_branch.png")
plt.close()


# ── Plot 8 & 9: MSE Histograms ────────────────────────────────────────────────
def compute_mse_per_sample(predict_fn, u_data, s_data):
    mses = []
    for i in range(len(u_data)):
        pred = predict_fn(u_data[i])
        mses.append(float(jnp.mean((pred - s_data[i]) ** 2)))
    return np.array(mses)

classic_mse_train = compute_mse_per_sample(lambda u: classic_predict(classic_model, u), u_train, s_train)
orth_mse_train    = compute_mse_per_sample(lambda u: orth_predict(Q_orth, orth_branch, u), u_train, s_train)
td_mse_train      = compute_mse_per_sample(lambda u: td_predict(jnp.asarray(Q_sign), td_branch, u), u_train, s_train)

classic_mse_test  = compute_mse_per_sample(lambda u: classic_predict(classic_model, u), u_test, s_test)
orth_mse_test     = compute_mse_per_sample(lambda u: orth_predict(Q_orth, orth_branch, u), u_test, s_test)
td_mse_test       = compute_mse_per_sample(lambda u: td_predict(jnp.asarray(Q_sign), td_branch, u), u_test, s_test)

for split, (c_mse, o_mse, td_mse) in [("train", (classic_mse_train, orth_mse_train, td_mse_train)),
                                        ("test",  (classic_mse_test,  orth_mse_test,  td_mse_test))]:
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = 30
    ax.hist(c_mse,  bins=bins, alpha=0.6, label=f"Classic (mean={c_mse.mean():.3e})")
    ax.hist(o_mse,  bins=bins, alpha=0.6, label=f"Orth 2-Step (mean={o_mse.mean():.3e})")
    ax.hist(td_mse, bins=bins, alpha=0.6, label=f"Orth TD 2-Step (mean={td_mse.mean():.3e})")
    ax.axvline(c_mse.mean(),  linestyle='--', linewidth=1.5, color='C0')
    ax.axvline(o_mse.mean(),  linestyle='--', linewidth=1.5, color='C1')
    ax.axvline(td_mse.mean(), linestyle='--', linewidth=1.5, color='C2')
    ax.set_xlabel("MSE"); ax.set_ylabel("Count")
    ax.set_title(f"Per-Sample MSE Distribution ({split})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / f"compare_mse_histogram_{split}.png")
    plt.close()


# ── Plot 10: Per-Timestep MSE ─────────────────────────────────────────────────
def classic_pred_batch_train():
    trunk_model, branch_model = classic_model
    T_MAT = jax.vmap(trunk_model)(tx_grid)
    B_MAT = jax.vmap(branch_model)(u_train)
    return (T_MAT @ B_MAT.T).reshape(len(t_grid), len(x_grid), -1)

def orth_pred_batch_train():
    B_MAT = jax.vmap(orth_branch)(u_train)
    return (Q_orth @ B_MAT.T).reshape(len(t_grid), len(x_grid), -1)

def td_pred_batch_train():
    Q_jnp = jnp.asarray(Q_sign)
    B_tnK = jax.vmap(
        lambda t: jax.vmap(lambda u: td_branch(jnp.concatenate([u, jnp.array([t])])))(u_train)
    )(t_grid)
    return jnp.einsum("txk,tnk->txn", Q_jnp, B_tnK)

def compute_mse_per_timestep(predict_fn_batch, s_data):
    pred = predict_fn_batch()
    diff = pred - jnp.transpose(s_data, (1, 2, 0))
    return np.array(jnp.mean(diff ** 2, axis=(1, 2)))

mse_per_t_classic = compute_mse_per_timestep(classic_pred_batch_train, s_train)
mse_per_t_orth    = compute_mse_per_timestep(orth_pred_batch_train,    s_train)
mse_per_t_td      = compute_mse_per_timestep(td_pred_batch_train,      s_train)

plt.figure(figsize=(10, 7))
plt.semilogy(t_grid, mse_per_t_classic, label="Classic")
plt.semilogy(t_grid, mse_per_t_orth,    label="Orth 2-Step")
plt.semilogy(t_grid, mse_per_t_td,      label="Orth TD 2-Step")
plt.xlabel("t"); plt.ylabel("Mean MSE")
plt.title("Per-Timestep MSE (Train) — averaged over x and samples")
plt.legend()
plt.tight_layout()
plt.savefig(plot_dir / "compare_mse_per_timestep_train.png")
plt.close()

print(f"All plots saved to {plot_dir}")