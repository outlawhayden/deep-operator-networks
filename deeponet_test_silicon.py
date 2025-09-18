import jax.numpy as jnp
import numpy as np
import jax
import optax
import scipy
from scipy.spatial.distance import cdist
from scipy.integrate import RK45
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import equinox as eqx
from jax.nn.initializers import variance_scaling
from jax.nn.initializers import he_normal
from tqdm import tqdm

seed = 42
np.random.seed(42)
key = jax.random.key(seed)
batch_size = 32
num_branches = 40
num_epochs = 10000
lr = 1e-4

print("\nconfiguring backend...")
jax.config.update("jax_platform_name", "metal")


print("backend selected:\n", jax.default_backend())
print("active devices:\n", jax.devices())
print("--------------------\n")

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
        architecture: [in, hidden1, hidden2, ..., out]
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

# DeepONet Class Definition
class DeepONet(eqx.Module):
    branch: MLP
    trunk: MLP

    def __init__(self, branch_arch, trunk_arch, key, num_branches, activation=jax.nn.relu, initializer = he_normal()): 
        bkey, tkey = jax.random.split(key)
        self.branch = MLP(branch_arch, bkey, activation, initializer = initializer)
        self.trunk = MLP(trunk_arch, tkey, activation, initializer = initializer)

    def __call__(self, x):
        u, y = x
        branch_out = self.branch(jnp.atleast_1d(u))
        trunk_out = self.trunk(jnp.atleast_1d(y))
        return jnp.inner(branch_out, trunk_out)


# L2 loss function
def loss_fn(model, x, y):
    pred_y = jax.vmap(model)(x)
    return jnp.mean((y - pred_y) ** 2)

# load functional dataset
print("loading dataset...")
try:
    dataset = np.load("data/integral_dataset.npy")
    print("dataset loaded\n")
except:
    print("error loading dataset\n")

# split into components
u_all = dataset[:,:,1] # function at x_i
x_all = dataset[:,:,0] # x_i sensor locations
F_all = dataset[:,:,2] # target fxn, in this case also at x_i

# total 1d x time mesh size
n_traj, n_points = u_all.shape

# make 1d x time mesh
u_vectors = np.repeat(u_all, repeats = n_points, axis = 0)

y_points = x_all.reshape(-1,1)
targets = F_all.reshape(-1)

# train_test_split using sklearn
indices = np.arange(len(targets))
train_idx, test_idx = train_test_split(indices, test_size = 0.33, random_state = seed)

X_u_train, X_u_test = jnp.array(u_vectors[train_idx]), jnp.array(u_vectors[test_idx])
X_y_train, X_y_test = jnp.array(y_points[train_idx]), jnp.array(y_points[test_idx])
y_train, y_test = jnp.array(targets[train_idx]), jnp.array(targets[test_idx])


# batching utility
def get_batch(batch_size, U,Y,T,key):
    key, subkey = jax.random.split(key)
    idx = jax.random.choice(subkey, len(U), (batch_size,), replace = False)
    return (U[idx], Y[idx]), T[idx], key

# initialize training history
train_hist = np.zeros(num_epochs)
test_hist = np.zeros(num_epochs)

# equinox training step 
@eqx.filter_jit
def train_step(model, opt_state, x, y, optimizer):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y) # get loss and gradients
    updates, opt_state = optimizer.update(grads, opt_state, model) # update step
    model = eqx.apply_updates(model, updates) # apply updates
    return model, opt_state, loss

# testing loss step, jit wrap on loss_fn
@eqx.filter_jit
def eval_step(model, x, y): 
    return loss_fn(model, x, y)

u_dim = u_vectors.shape[1]
y_dim = 1
# define architectures for component networks
branch_arch = [u_dim, 40, 40,  num_branches]
trunk_arch = [y_dim, 40, 40, num_branches]



# model definition
print(f"initializing model, b_arch: {branch_arch}, t_arch: {trunk_arch}, num_branches: {num_branches}")
try:
    model = DeepONet(branch_arch = branch_arch, trunk_arch = trunk_arch, num_branches = num_branches, key = key)
    print("model initialized\n")
except:
    print("error initializing model\n")
# optimizer - adam optax builtin. chain with gradient clipping
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(lr))
opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

train_hist, test_hist = [],[]

# training loop, wrapped in tqdm
print("beginning training...")
for epoch in tqdm(range(num_epochs)):
    # using batch utility
    (u_batch, y_batch), t_batch, key = get_batch(batch_size, X_u_train, X_y_train, y_train, key)
    # step model
    model, opt_state, train_loss = train_step(model, opt_state, (u_batch, y_batch), t_batch, optimizer)

    # calculate and log losses
    test_loss = eval_step(model, (X_u_test, X_y_test), y_test)

    train_hist.append(float(train_loss))
    test_hist.append(float(test_loss))

print(test_hist[-1])
print("training complete\n")
# serialize leaves via equinox
print("saving out...")
try:
    eqx.tree_serialise_leaves("deeponet_model.eqx", model)
    print("model saved to deeponet_model.eqx")
except:
    print("error exporting, model not saved")

# package export loss values
np.savez("deeponet_losses.npz", train=np.array(train_hist), test=np.array(test_hist))


