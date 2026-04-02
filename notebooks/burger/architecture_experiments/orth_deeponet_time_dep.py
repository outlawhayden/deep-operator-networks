## using envml conda enviro

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




seed = config.get('seed', 42)
num_bases = config.get('num_bases', 10)
num_trunk_epochs = config.get('num_trunk_epochs', 6000)
lr = config.get('lr', 1e-3)
num_LBFGS_epochs = config.get('num_LBFGS_epochs', 1000)
eps = config.get('eps', 1e-8)

# Dataset paths
dataset_path = Path(config.get('dataset_path', '/Users/haydenoutlaw/Documents/Research/opnet/deep-operator-networks/data/burgers_dataset.npz'))
save_path = Path(config.get('save_path', 'orth_qr_td_factors.npz'))

# Network architectures
trunk_arch = config.get('trunk_arch', [2, 40, 40, 10])
branch_arch = config.get('branch_arch', [2, 60, 60, 10])

branch_arch[0] = branch_arch[0] + 1 # account for t now being an input parameter to branch net

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

    def __init__(self, architecture, key, activation = jax.nn.relu, initializer = he_normal()):
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

# %% [markdown]
# ## loading dataset

# %%
# load external dataset
dataset = np.load(dataset_path, allow_pickle = True)
t_grid = jnp.array(dataset['t'])
x_grid = jnp.array(dataset['x'])


data = dataset['samples']
u = np.array([i['params'] for i in data])
s = np.array([i['solution'] for i in data]) # s is shape (n,t,x)

# disabled functionality for train/test split for now
n_samp = len(data)
train_indices, test_indices = train_test_split(np.arange(n_samp), test_size = test_size, random_state = seed)
u_train, u_test = jnp.array(u[train_indices]), jnp.array(u[test_indices])
s_train, s_test = jnp.array(s[train_indices]), jnp.array(s[test_indices])


output_tr = jnp.transpose(s_train, axes=(1,2,0)) 

tt,xx = jnp.meshgrid(t_grid, x_grid, indexing = "ij")
tx_grid = jnp.concatenate([tt.flatten()[:,None], xx.flatten()[:,None]], axis=1) #xt_grid[:,1] is x, xt_grid[:,0] is t

## initializing model, loss_fn

u_dim = 2 # in this case, two 
y_dim = 2 # (t,x)

# define architectures for component networks

key, subkey_t, subkey_A = jax.random.split(key, num=3)
trunk_model = MLP(trunk_arch,key = subkey_t)
A_model = jax.random.normal(subkey_A, (len(t_grid), num_bases, output_tr.shape[2])) # now A_model is a 3-tensor -- indexed as t, branch, n. as opposed to one matrix for entire xt domain

model = trunk_model, A_model
best_model = model

def loss_fn(model):
    trunk_model, A_model = model
    T_MAT = jax.vmap(lambda t: jax.vmap(lambda x: trunk_model(jnp.stack([x,t])))(x_grid))(t_grid) # trunk model matrix. output of model given input entire [x_grid, t_grid]
    pred_y = jnp.einsum("txk, tkn -> txn", T_MAT, A_model)  # (T,X,N)
    #pred_y = jnp.einsum("txN, tNk -> txk",T_MAT, A_model) # einsum is a tensor contraction. keep t index (time) and at each t index, do (x,N) times (N,k) -> (x,k). N = num_fxns, k = num_samples
    return jnp.mean((output_tr - pred_y) ** 2) # mean squared error


opt = optax.adam(lr)
opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

loss_hist = []
min_loss_hist = [np.inf]
# standard eqx training step
@eqx.filter_jit
def train_step(model, opt_state):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
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
        min_loss_hist.append(min_loss_hist[-1])
    if step % 10 == 0: # new carriage print, avoid tqdm for memory
        end_time = time.time()
        print(f"\rAdam step {step}: loss={float(loss):.3e}, {end_time-start_time:.2f}s", end="", flush=True)
        start_time = time.time()
e_time = time.time()
print(f"\nfinal adam loss: {loss:.3e}, total time: {e_time-s_time:.2f}s\n")

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


model = best_model


plt.plot(loss_hist)
plt.plot(min_loss_hist[1:])
plt.title(f"Trunk Model Training History, MSE, Orth 2-Step")
plt.yscale('log')
plt.savefig("orth_td_training_history.png")
plt.close()

# ## QR Factorizations at Time Data

trunk_model, A_model = model
# SAME LOGIC AS ABOVE - T_MAT is trunk matrix output of entire [x_grid, t_grid]
T_MAT = jax.vmap(lambda t: jax.vmap(lambda x: trunk_model(jnp.stack([x,t])))(x_grid))(t_grid)

# for index i of t_grid (subsampled here every 2)
for i in range(len(t_grid)):
    # take QR of T_MAT[i,:,:]
    Q,R = scipy.linalg.qr(T_MAT[i], mode = 'economic')

    if i == 0:
        Q_MAT = Q[jnp.newaxis, :, :] # make the 3 tensor
        R_MAT = R[jnp.newaxis,:, :]
    else:
        Q_MAT = jnp.concatenate([Q_MAT, Q[jnp.newaxis, :, :]], axis = 0) # stack Q_MAT in first dimension
        R_MAT = jnp.concatenate([R_MAT, R[jnp.newaxis, :, :]], axis = 0)
    

Q_sign = np.array(Q_MAT, copy = True) #make np.array copies of tensors of QR factors
R_sign = np.array(R_MAT, copy = True)

# account for nonuniqueness in QR algorithm -- pick the closest Q columns by sign to ensure Q factors are time continuous
T = Q_sign.shape[0]
n = Q_sign.shape[2]

for k in range(1, T):          # time
    for j in range(n):         # basis index
        # choose sign to match previous time, just based on the first column
        if np.dot(Q_sign[k-1, :, j], Q_sign[k, :, j]) < 0:
            Q_sign[k, :, j] *= -1      # flip basis column of Q
            R_sign[k, j, :] *= -1      # flip corresponding row of R



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
        Q_k = Q_sign[:, :, k]

        # 2D top-down plot
        im = ax.contourf(T, X, Q_k, levels=100, cmap='viridis')

        ax.set_title(f"Q[{k}]")
        ax.set_xlabel("t")
        ax.set_ylabel("x")
    else:
        ax.axis("off")

fig.suptitle(
    f"Orthogonalized Trunk Bases Across Time (Sign Corrected)",
    fontsize=16
)

# attach a single colorbar for the whole figure
cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.75)

plt.savefig("orth_td_bases.png")
plt.close()



# ## branch network training

K = A_model.shape[1] # number of basis functions
N = A_model.shape[2] # number of systems 



key, subkey_t, subkey_A = jax.random.split(key, num=3)
branch_model = MLP(branch_arch,key = subkey_t)
best_branch_model = branch_model



B_tnk = jax.vmap(
    lambda t: jax.vmap(
        lambda u: branch_model(jnp.concatenate([u, jnp.array([t])]))
    )(u_train)   # u_train shape (N,2)
)(t_grid)        # (T,N,K)

B_MAT = jnp.swapaxes(B_tnk, 1, 2)  # (T, K, N)


RA_model = jnp.einsum("tij, tjk -> tik", R_sign, A_model) ## IS THIS LOGIC RIGHT??



RA_target = jnp.asarray(RA_model)  # (T, K, N), frozen data

def branch_loss_fn(branch_model, RA_target, u_train, t_grid, eps=1e-8): # vmap u_train (parameters) and t_grid through branch model -- loss is B_NN - R^*A^*
    B_tnK = jax.vmap(
        lambda t: jax.vmap(
            lambda u: branch_model(jnp.concatenate([u, jnp.array([t])]))
        )(u_train)
    )(t_grid)  # (T, N, K)

    B_MAT = jnp.swapaxes(B_tnK, 1, 2)  # (T, K, N) # does require an internal tensor transpose
    return jnp.mean((B_MAT - RA_target) ** 2)

# ---- optimizer ----

@eqx.filter_jit # does require filter value and grad because of frozen RA target
def branch_train_step(branch_model, opt_state, RA_target, u_train, t_grid):
    loss, grads = eqx.filter_value_and_grad(branch_loss_fn)( 
        branch_model, RA_target, u_train, t_grid
    )
    updates, opt_state = opt.update(grads, opt_state, branch_model)
    branch_model = eqx.apply_updates(branch_model, updates)
    return branch_model, opt_state, loss

branch_loss_hist = []
branch_min_loss_hist = [np.inf]
opt = optax.adam(lr)

# IMPORTANT: init on branch_model only
opt_state = opt.init(eqx.filter(branch_model, eqx.is_inexact_array))

for step in range(num_trunk_epochs):
    branch_model, opt_state, loss = branch_train_step(
        branch_model, opt_state, RA_target, u_train, t_grid
    )
    branch_loss_hist.append(float(loss))
    if float(loss) < branch_min_loss_hist[-1]:
        branch_min_loss_hist.append(float(loss))
        best_branch_model = branch_model
    else:
        branch_min_loss_hist.append(branch_min_loss_hist[-1])
    if step % 10 == 0:
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
#             end="", flush=True)



print(f"\nfinal adam loss: {float(loss):.3e}\n")

branch_model = best_branch_model

trunkA_path = Path(config.get("trunkA_model_path", "orth_td_trunkA.eqx"))
branch_path = Path(config.get("branch_model_path", "orth_td_branch.eqx"))
meta_path   = Path(config.get("meta_path", "orth_td_meta.npz"))
qr_path     = Path(config.get("qr_path", "orth_td_qr_factors.npz"))  # optional

# 1) Save trunk + time-dependent A_model pytree
eqx.tree_serialise_leaves(trunkA_path, model)  # model = (trunk_model, A_model)

# 2) Save branch model pytree
eqx.tree_serialise_leaves(branch_path, branch_model)

# 3) Save lightweight metadata needed to rebuild templates when loading
np.savez_compressed(
    meta_path,
    t_grid=np.asarray(t_grid),
    x_grid=np.asarray(x_grid),
    trunk_arch=np.asarray(trunk_arch),
    branch_arch=np.asarray(branch_arch),
    num_bases=np.asarray(num_bases),
    seed=np.asarray(seed),
    N_train=np.asarray(output_tr.shape[2]),      # critical
    T_len=np.asarray(len(t_grid)),              # sanity / convenience
)

# 4) Optional: save derived QR/sign-corrected factors for analysis/reconstruction
np.savez_compressed(
    qr_path,
    Q=np.asarray(Q_sign),
    R=np.asarray(R_sign),
)

print("\nexported trunk+A model to:", trunkA_path)
print("exported branch model to:", branch_path)
print("exported metadata to:", meta_path)
print("exported QR factors to:", qr_path)




plt.plot(branch_loss_hist)
plt.plot(branch_min_loss_hist)
plt.title(f"Branch Model Training History, MSE, Orth 2-Step")
plt.yscale('log')
plt.savefig('orth_td_branch_training_history.png')
plt.close()


u_rand, s_rand = u_test[20], s_test[20] # s is true values, u is input parameters

def branch_over_time(branch_model, u, t_grid):
    # u: (u_dim,), t_grid: (T,), returns (T, K)
    return jax.vmap(lambda t: branch_model(jnp.concatenate([u, jnp.array([t])])))(t_grid)

def compute_QHb_grid(Q_sign, branch_model, u, t_grid):
    """
    Q_sign: (T, X, K)  storing Q(t,x,k)
    u:      (u_dim,)
    returns: (T, X) with entries sum_k Q[t,x,k] * b[t,k]
    """
    b_tk = branch_over_time(branch_model, u, t_grid)          # (T, K)
    grid_tx = jnp.einsum("txk,tk->tx", Q_sign, b_tk)  # (T, X)
    return grid_tx

rand_pred_tx = compute_QHb_grid(Q_sign, branch_model, jnp.array(u_rand), jnp.array(t_grid))

plt.figure(figsize=(7, 4))
plt.imshow(
    rand_pred_tx,
    extent=[x_grid[0], x_grid[-1], t_grid[-1], t_grid[0]],
    aspect="auto",
    origin="upper"
)
plt.colorbar(label=r"$Q(t,x)^* b_{\mathrm{NN}}(t)$")
plt.xlabel("x")
plt.ylabel("t")
plt.title("Grid evaluation of $Q^*(t,x)b_{NN}(t)$, Orth 2-Step")
plt.tight_layout()
plt.savefig("orth_td_prediction_grid.png")


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
plt.savefig("orth_td_true_solution_grid.png")


plt.figure(figsize=(7, 4))
plt.imshow(
    rand_pred_tx - s_rand,
    extent=[x_grid[0], x_grid[-1], t_grid[-1], t_grid[0]],
    aspect="auto",
    origin="upper"
)
plt.colorbar(label=r"$Q(t,x)^* b_{\mathrm{NN}}(t) - s(x,t)$")
plt.xlabel("x")
plt.ylabel("t")
plt.title("Evaluation Error ($Q^*(t,x)b_{NN}(t) - s(x,t)$), Orth 2-Step")
plt.savefig("orth_td_error_grid.png")
plt.close()


def predict_solution_Q_time_dep(Q_sign, branch_model, u, t_grid):
    """
    Q_sign: (T, X, K)
    u: (u_dim,)
    returns: (T, X)
    """
    b_tk = branch_over_time(branch_model, u, t_grid)  # (T, K)
    pred_tx = jnp.einsum("txk,tk->tx", Q_sign, b_tk)  # (T, X)
    return pred_tx

def test_mse_time_dep(Q_sign, branch_model, u_test, s_test, t_grid):
    """
    Average MSE over test set:
      mean_{n,t,x} (pred_n(t,x) - s_test[n,t,x])^2
    """
    preds = jax.vmap(lambda u: predict_solution_Q_time_dep(Q_sign, branch_model, u, t_grid))(u_test)  # (N_test, T, X)
    return jnp.mean((preds - s_test) ** 2)

final_test_mse = test_mse_time_dep(
    jnp.asarray(Q_sign),       # (T, X, K)
    branch_model,
    u_test,                    # (N_test, u_dim)
    s_test,                    # (N_test, T, X)
    t_grid
)

print("\n==============================")
print(f"Average Test MSE (time-dependent Q @ b(t,u)): {float(final_test_mse):.6e}")
print("==============================\n")


