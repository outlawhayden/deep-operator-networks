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
from tqdm import tqdm
from matplotlib import cbook, cm
from matplotlib.colors import LightSource
import time
from pathlib import Path

seed = 42
np.random.seed(seed)
key = jax.random.key(seed)

num_bases = 10
num_trunk_epochs = 6000
lr = 1e-3
num_LBFGS_epochs = 1000
eps = 1e-8

print("\nconfiguring backend...")
jax.config.update("jax_platform_name", "metal")
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


dataset = np.load('/Users/haydenoutlaw/Documents/Research/opnet/deep-operator-networks/data/burgers_dataset.npz', allow_pickle=True)
t_grid = jnp.array(dataset['t'])
x_grid = jnp.array(dataset['x'])
print("grid is (t,x) = ", len(t_grid), len(x_grid))

data = dataset['samples']
u = np.array([i['params'] for i in data])
s = np.array([i['solution'] for i in data])

n_samp = len(data)
train_indices, test_indices = train_test_split(np.arange(n_samp), test_size=0.2, random_state=seed)
u_train, u_test = jnp.array(u[train_indices]), jnp.array(u[test_indices])
s_train, s_test = jnp.array(s[train_indices]), jnp.array(s[test_indices])

print('label dataset size is (n, coord) = ', u_train.shape)
print('input dataset size is (n, x, t) = ', s_train.shape)

output_tr = jnp.transpose(s_train, axes=(1, 2, 0))
print(f"subsampled dataset size is (t,x,n) = {output_tr.shape}")

tt, xx = jnp.meshgrid(t_grid, x_grid)
tx_grid = jnp.concatenate([tt.flatten()[:, None], xx.flatten()[:, None]], axis=1)
print("tx_grid is of shape (n, coord) = ", tx_grid.shape)

u_dim = 2
y_dim = 2
trunk_arch = [y_dim, 40, 40, num_bases]

key, subkey_t, subkey_A = jax.random.split(key, num=3)
trunk_model = MLP(trunk_arch, key=subkey_t)
A_model = jax.random.normal(subkey_A, (num_bases, output_tr.shape[2]))
print("A_model is shape = ", A_model.shape)

T_MAT = jax.vmap(trunk_model)(tx_grid)
print(T_MAT.shape)
model = trunk_model, A_model

def loss_fn(model):
    trunk_model, A_model = model
    T_MAT = jax.vmap(trunk_model)(tx_grid)
    pred_y = T_MAT @ A_model
    pred_y_3d = pred_y.reshape(len(t_grid), len(x_grid), -1)
    return jnp.sqrt(jnp.sum((output_tr - pred_y_3d) ** 2))

opt = optax.adam(lr)
opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

loss_hist = []

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
    loss_hist.append(float(loss))
    if step % 10 == 0:
        end_time = time.time()
        print(f"\rAdam step {step}: loss={float(loss):.3e}, {end_time-start_time:.2f}s", end="", flush=True)
        start_time = time.time()
e_time = time.time()
print(f"\nfinal adam loss: {loss:.3e}, total time: {e_time-s_time:.2f}s\n")

opt = optax.lbfgs()
opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

for step in range(num_LBFGS_epochs):
    model, opt_state, loss = train_step(model, opt_state)
    loss_hist.append(float(loss))
    if step % 100 == 0:
        print(f"\r{' ' * 80}\rLBFGS step {step}: loss={float(loss):.3e}", end="", flush=True)

plt.plot(loss_hist)
plt.title("Trunk Model Training History, Absolute L2 Error")
plt.yscale('log')
plt.show()

trunk_model, A_model = model
T_MAT = jax.vmap(trunk_model)(tx_grid)
print("T_MAT is shape (t*x,n) = ", T_MAT.shape)

Q_MAT, R_MAT = scipy.linalg.qr(T_MAT, mode='economic')
print(f"Q_MAT is of shape (t*x,n) = {Q_MAT.shape}")
print(f"R_MAT is of shape {R_MAT.shape}")
print("shapes agree?", Q_MAT.shape == T_MAT.shape)

save_path = Path("qr_factors.npz")
np.savez_compressed(
    save_path,
    Q=np.asarray(Q_MAT),
    R=np.asarray(R_MAT),
    t_grid=np.asarray(t_grid),
    x_grid=np.asarray(x_grid)
)
print("exported to:", save_path)

K = A_model.shape[0]
N = A_model.shape[1]

branch_arch = [u_train.shape[1], 60, 60, K]

key, subkey_t, subkey_A = jax.random.split(key, num=3)
branch_model = MLP(branch_arch, key=subkey_t)

B_MAT = jax.vmap(branch_model)(u_train)
print("Branch Model is shape:", B_MAT.shape)

RA_model = (R_MAT @ A_model).T
print("RA_Model is shape:", RA_model.shape)
print(f"B_MAT shape: {B_MAT.shape}, RA_model shape: {RA_model.shape}")
print(f"Shapes match: {B_MAT.shape == RA_model.shape}")

RA_target = jnp.asarray(RA_model)

def branch_loss_fn(branch_model, RA_target, u_train):
    B_MAT = jax.vmap(branch_model)(u_train)
    return jnp.sqrt(jnp.sum((B_MAT - RA_target) ** 2))

@eqx.filter_jit
def branch_train_step(branch_model, opt_state, RA_target, u_train, t_grid):
    loss, grads = eqx.filter_value_and_grad(branch_loss_fn)(
        branch_model, RA_target, u_train
    )
    updates, opt_state = opt.update(grads, opt_state, branch_model)
    branch_model = eqx.apply_updates(branch_model, updates)
    return branch_model, opt_state, loss

branch_loss_hist = []

opt = optax.adam(lr)
opt_state = opt.init(eqx.filter(branch_model, eqx.is_inexact_array))

start_time = time.time()
for step in range(num_trunk_epochs):
    branch_model, opt_state, loss = branch_train_step(
        branch_model, opt_state, RA_target, u_train, t_grid
    )
    branch_loss_hist.append(float(loss))
    if step % 10 == 0:
        print(f"\rAdam step {step}: loss={float(loss):.3e}", end="", flush=True)

opt = optax.lbfgs()
opt_state = opt.init(eqx.filter(branch_model, eqx.is_inexact_array))

for step in range(num_LBFGS_epochs):
    branch_model, opt_state, loss = branch_train_step(
        branch_model, opt_state, RA_target, u_train, t_grid
    )
    branch_loss_hist.append(float(loss))
    if step % 10 == 0:
        print(f"\rLBFGS step {step}: loss={float(loss):.3e}", end="", flush=True)

print(f"\nfinal loss: {float(loss):.3e}\n")

plt.plot(branch_loss_hist)
plt.title("Branch Model Training History, Absolute L2 Error")
plt.yscale('log')
plt.show()

u_rand, s_rand = u_train[20], s_train[20]

def compute_QHb_grid(Q_MAT, branch_model, u, t_grid):
    b = branch_model(u)
    pred_flat = Q_MAT @ b
    pred_grid = pred_flat.reshape(len(t_grid), len(x_grid))
    return pred_grid

u_rand = u_train[20]
rand_pred_tx = compute_QHb_grid(Q_MAT, branch_model, u_rand, t_grid)
print(f"Prediction shape: {rand_pred_tx.shape}")

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
plt.title("Grid evaluation of $Q^*(t,x)b_{NN}$")
plt.tight_layout()
plt.show()

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
plt.show()

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
plt.title("Evaluation Error ($Q^*(t,x)b_{NN}(t) - s(x,t)$), Absolute L2")
plt.tight_layout()
plt.show()