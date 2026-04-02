
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

# Dataset paths
dataset_path = Path(config.get('dataset_path', '/Users/haydenoutlaw/Documents/Research/opnet/deep-operator-networks/data/burgers_dataset.npz'))
save_path = Path(config.get('save_path', 'orth_qr_factors.npz'))

# Network architectures
trunk_arch = config.get('trunk_arch', [2, 40, 40, 10])
branch_arch = config.get('branch_arch', [2, 60, 60, 10])

trunk_arch[-1] = num_bases
branch_arch[-1] = num_bases

# Training parameters
test_size = config.get('test_size', 0.2)
log_frequency = config.get('log_frequency', 10)
lbfgs_log_frequency = config.get('lbfgs_log_frequency', 100)

# Set random seeds
np.random.seed(seed)
key = jax.random.key(seed)

print("\nconfiguring backend...")

jax.config.update("jax_platform_name", "gpu")


print("backend selected:\n", jax.default_backend())
print("active devices:\n", jax.devices())
print("--------------------\n")

## EQUINOX CLASS DEFINITIONS
# Linear Model (Wx + b)
class Linear(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __init__(self, in_size, out_size, key, initializer = he_normal()):
        wkey, bkey = jax.random.split(key)
        self.weight = initializer(wkey, (out_size, in_size), dtype=jnp.float32)
        self.bias = jnp.zeros((out_size,), dtype=jnp.float32)

    def __call__(self, x):
        return self.weight @ x + self.bias


# Basic MultiLayer Perceptron Primitive
class MLP(eqx.Module):
    layers: list
    activations: list

    def __init__(self, architecture, key, activation = jax.nn.gelu, initializer = he_normal()): ## switched to GELU act
        """
        architecture: List[in, hidden1, hidden2, ..., out]
        key: random key
        activation: jax.nn act fxn
        """
        keys = jax.random.split(key, len(architecture) - 1)
        self.layers = [
            Linear(architecture[i], architecture[i+1], keys[i], initializer = initializer) for i in range(len(architecture) - 1)]
        self.activations = [activation] * (len(self.layers) - 1) + [eqx.nn.Identity()] # no activation on last layer

    def __call__(self, x):
        for layer, act in zip(self.layers, self.activations):
            x = act(layer(x))
        return x


# load external dataset
dataset = np.load(dataset_path, allow_pickle = True)
t_grid = jnp.array(dataset['t'])
x_grid = jnp.array(dataset['x'])

data = dataset['samples']
u = np.array([i['params'] for i in data])
s = np.array([i['solution'] for i in data]) # s is shape (n,t,x)

n_samp = len(data)
train_indices, test_indices = train_test_split(np.arange(n_samp), test_size = test_size, random_state = seed)
u_train, u_test = jnp.array(u[train_indices]), jnp.array(u[test_indices])
s_train, s_test = jnp.array(s[train_indices]), jnp.array(s[test_indices])


output_tr = jnp.transpose(s_train, axes=(1,2,0)) 
output_test = jnp.transpose(s_test, axes = (1,2,0))


tt, xx = jnp.meshgrid(t_grid, x_grid, indexing="ij")  # tt,xx are (T, X)
tx_grid = jnp.stack([tt.reshape(-1), xx.reshape(-1)], axis=1)  # (T*X, 2)


u_dim = 2 # in this case, two 
y_dim = 2 # (t,x)


key, subkey_t, subkey_A = jax.random.split(key, num=3)
trunk_model = MLP(trunk_arch, key = subkey_t)
A_model = jax.random.normal(subkey_A, (num_bases, output_tr.shape[2])) # now A_model is a 2D matrix


model = trunk_model, A_model
best_model = model

output_scale = jnp.mean(jnp.linalg.norm(output_tr.reshape(-1, output_tr.shape[2]), axis=0))

def loss_fn(model, inputs, labels):
    trunk_model, A_model = model
    T_MAT = jax.vmap(trunk_model)(inputs)
    pred_y = T_MAT @ A_model
    pred_y_3d = pred_y.reshape(len(t_grid), len(x_grid), -1)  # (T, X, N)
    diff = labels - pred_y_3d
    return jnp.mean(jnp.linalg.norm(diff.reshape(-1, diff.shape[2]), axis=0)) / output_scale



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
# standard eqx training step
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
    loss_hist.append(float(loss)) # store as list
    if float(loss) < min_loss_hist[-1]:
        min_loss_hist.append(float(loss))
        best_model = model
    else:
        min_loss_hist.append(float(min_loss_hist[-1]))
    if step % 10000 == 0: # new carriage print, avoid tqdm for memory
        end_time = time.time()
        print(f"\rAdam step {step}: loss={float(loss):.3e}, {end_time-start_time:.2f}s", end="", flush=True)
        start_time = time.time()
e_time = time.time()
print(f"\nfinal adam loss: {loss:.3e}, total time: {e_time-s_time:.2f}s\n")

model = best_model # use best version

# opt = optax.lbfgs()

# for step in range(num_LBFGS_epochs):
#     model, opt_state, loss = train_step(model, opt_state)
#     loss_hist.append(float(loss))
#     if step % 100 == 0:
#         print(
#             f"\r{' ' * 80}\rLBFGS step {step}: loss={float(loss):.3e}",
#             end="",
#             flush=True
#         )

plt.plot(loss_hist)
plt.plot(min_loss_hist[1:])
plt.title(f"Trunk Model Training History, Mean L2, 2-Step")
plt.yscale('log')
plt.savefig('orth_training_history.png')
plt.close()

# ## QR Factorizations at Time Data


trunk_model, A_model = model
# SAME LOGIC AS ABOVE - T_MAT is trunk matrix output of entire [x_grid, t_grid]
T_MAT = jax.vmap(trunk_model)(tx_grid)
T_MAT_host = np.asarray(T_MAT)
Q_MAT, R_MAT = scipy.linalg.qr(T_MAT_host, mode="economic")

# (c) convert Q/R back to jax for later matmuls with branch_model outputs
Q_MAT = jnp.asarray(Q_MAT)
R_MAT = jnp.asarray(R_MAT)

# ## branch network training
K = A_model.shape[0]  # number of basis functions = 10
N = A_model.shape[1]  # number of samples = 352

key, subkey_t, subkey_A = jax.random.split(key, num=3)
branch_model = MLP(branch_arch, key=subkey_t)

B_MAT = jax.vmap(branch_model)(u_train)  # Shape: (352, 10)

# RA_model should match B_MAT shape: (N, K) = (352, 10)
RA_model = (R_MAT @ A_model).T  # Transpose to get (352, 10)



RA_target = jnp.asarray(RA_model)  # (T, K, N), frozen data

def branch_loss_fn(branch_model, RA_target, u_train, eps=1e-8):
    B_MAT = jax.vmap(branch_model)(u_train)  # (N, K)
    diff = B_MAT - RA_target
    return jnp.mean(jnp.linalg.norm(diff, axis=1))  # norm over K per sample, mean over N

# ---- optimizer ----
@eqx.filter_jit # does require filter value and grad because of frozen RA target
def branch_train_step(branch_model, opt_state, RA_target, u_train):
    loss, grads = eqx.filter_value_and_grad(branch_loss_fn)(
        branch_model, RA_target, u_train
    )
    updates, opt_state = opt.update(grads, opt_state, branch_model)
    branch_model = eqx.apply_updates(branch_model, updates)
    return branch_model, opt_state, loss

branch_loss_hist = []
branch_min_loss_hist = [np.inf]

opt = optax.adam(lr_schedule)

best_branch_model = branch_model
# IMPORTANT: init on branch_model only
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
        end_time = time.time()
        print(f"\rAdam step {step}: loss={float(loss):.3e}",
              end="", flush=True)

# opt = optax.lbfgs()

# for step in range(num_LBFGS_epochs):
#     branch_model, opt_state, loss = branch_train_step(branch_model, opt_state, RA_target, u_train, t_grid)
#     branch_loss_hist.append(float(loss))
#     if step % 10 == 0:
#         end_time = time.time()
#         print(f"\rLBFGS step {step}: loss={float(loss):.3e}",
#               end="", flush=True)

print(f"\nfinal adam loss: {float(loss):.3e}\n")

branch_model = best_branch_model # use best version

plt.plot(branch_loss_hist)
plt.plot(branch_min_loss_hist[1:])
plt.title(f"Branch Model Training History, Mean L2, 2-Step")
plt.yscale('log')
plt.savefig("orth_branch_training_history.png")
plt.close()


T_MAT = jax.vmap(lambda t: jax.vmap(lambda x: trunk_model(jnp.stack([t,x])))(x_grid))(t_grid)

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
        T_k = T_MAT[:, :, k]

        # 2D top-down plot
        im = ax.contourf(T, X, T_k, levels=100)

        ax.set_title(f"T[{k}]")
        ax.set_xlabel("t")
        ax.set_ylabel("x")
    else:
        ax.axis("off")

fig.suptitle(
    f"Trunk Bases Across Time, 2-Step",
    fontsize=16
)

# attach a single colorbar for the whole figure
cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.75)

plt.savefig("orth_bases.png")
plt.close()




trunkA_path = Path(config.get("trunkA_model_path", "orth_trunkA.eqx"))
branch_path = Path(config.get("branch_model_path", "orth_branch.eqx"))
meta_path   = Path(config.get("meta_path", "orth_meta.npz"))
qr_path     = Path(config.get("qr_path", "orth_qr_factors.npz"))  # optional

# 1) Save trunk + A_model pytree
eqx.tree_serialise_leaves(trunkA_path, model)  # model = (trunk_model, A_model)

# 2) Save branch model pytree
eqx.tree_serialise_leaves(branch_path, branch_model)

# 3) Save small metadata (useful for rebuilding templates when loading)
np.savez_compressed(
    meta_path,
    t_grid=np.asarray(t_grid),
    x_grid=np.asarray(x_grid),
    trunk_arch=np.asarray(trunk_arch),
    branch_arch=np.asarray(branch_arch),
    num_bases=np.asarray(num_bases),
    seed=np.asarray(seed),
)

# 4) (Optional) also save Q/R if you want them for analysis/reconstruction
np.savez_compressed(
    qr_path,
    Q=np.asarray(Q_MAT),
    R=np.asarray(R_MAT),
)

print("\nexported trunk+A model to:", trunkA_path)
print("exported branch model to:", branch_path)
print("exported metadata to:", meta_path)
print("exported QR factors to:", qr_path)




u_rand, s_rand = u_test[20], s_test[20] # s is true values, u is input parameters

def compute_QHb_grid(Q_MAT, branch_model, u, t_grid):
    """
    Q_MAT: (T*X, K) storing Q flattened
    u: (u_dim,)
    returns: (T, X) with entries Q(t,x)^T * b(u)
    """
    # Get branch output for this u
    b = branch_model(u)  # Shape: (K,)
    
    # Compute Q @ b to get flattened predictions
    pred_flat = Q_MAT @ b  # Shape: (T*X,)
    
    # Reshape to (T, X) grid
    pred_grid = pred_flat.reshape(len(t_grid), len(x_grid))  # Shape: (T, X)
    return pred_grid


def predict_solution_from_Q(Q_MAT, branch_model, u, t_grid, x_grid):
    """Returns predicted solution grid (T, X) for one u using Q_MAT @ b(u)."""
    b = branch_model(u)                   # (K,)
    pred_flat = Q_MAT @ b                 # (T*X,)
    return pred_flat.reshape(len(t_grid), len(x_grid))  # (T, X)

def test_l2(Q_MAT, branch_model, u_test, s_test, t_grid, x_grid, eps=1e-12):
    preds = jax.vmap(lambda u: predict_solution_from_Q(Q_MAT, branch_model, u, t_grid, x_grid))(u_test)
    diff = preds - s_test
    return jnp.mean(jnp.linalg.norm(diff.reshape(diff.shape[0], -1), axis=1))  # mean L2 over N


final_test_l2 = test_l2(jnp.asarray(Q_MAT), branch_model, u_test, s_test, t_grid, x_grid)

print("\n==============================")
print(f"Average Test  L2 (Q @ b(u)): {float(final_test_l2):.6e}")
print("==============================\n")

# Example usage
u_rand = u_test[20]
rand_pred_tx = compute_QHb_grid(Q_MAT, branch_model, u_rand, t_grid)


plt.figure(figsize=(7, 4))
plt.imshow(
    rand_pred_tx,
    extent=[x_grid[0], x_grid[-1], t_grid[-1], t_grid[0]],
    aspect="auto",
    origin="upper",
)
plt.colorbar(label=r"$Q(t,x)^* b_{\mathrm{NN}}(u)$")
plt.xlabel("x")
plt.ylabel("t")
plt.title("Grid evaluation of $Q^*(t,x)b_{NN}$, 2-Step")
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
plt.colorbar(label=r"$Q(t,x)^* b_{\mathrm{NN}}(u) - s(x,t)$")
plt.xlabel("x")
plt.ylabel("t")
plt.title("Evaluation Error ($Q^*(t,x)b_{NN}(u) - s(x,u)$), 2-Step")
plt.savefig("orth_error_grid.png")
plt.close()






def l2_error_vs_time_Q_branch(
    Q_MAT,
    branch_model,
    u_test,
    s_test,
    t_grid,
    x_grid,
    ):
    """
    Returns L2 error curve over time, shape (T,).
    """
    preds = jax.vmap(
        lambda u: predict_solution_from_Q(Q_MAT, branch_model, u, t_grid, x_grid)
    )(u_test)  # (N, T, X)

    diff = preds - s_test  # (N, T, X)

    # L2 over space (X) -> (N, T)

    rel_curve = jnp.mean(jnp.linalg.norm(diff, axis=2), axis=0)  # norm over x, mean over N -> (T,)
    return rel_curve

rel_curve = l2_error_vs_time_Q_branch(
    jnp.asarray(Q_MAT), branch_model, u_test, s_test, t_grid, x_grid
)

# Plot (move data to host for matplotlib)
plt.figure()
plt.plot(jnp.asarray(t_grid), jax.device_get(rel_curve))
plt.xlabel("t")
plt.ylabel("L2 error over x")
plt.title("Test Mean L2 error vs time")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("orth_time_error.png")

master_end_time = time.time()

print("Total Time (sec):", master_end_time - master_start_time)