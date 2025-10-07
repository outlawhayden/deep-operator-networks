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
import scipy.linalg
from jax.nn.initializers import variance_scaling
from jax.nn.initializers import he_normal
from tqdm import tqdm

seed = 42
np.random.seed(seed)
key = jax.random.key(seed)
batch_size = 64
num_branches = 40
num_trunk_epochs = 100000
num_branch_epochs = 100000
lr = 1e-5

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


dataset = np.load('/Users/haydenoutlaw/Documents/Research/opnet/deep-operator-networks/data/burgers_dataset.npz', allow_pickle = True)
t_grid = jnp.array(dataset['t'])
x_grid = jnp.array(dataset['x'])
data = dataset['samples']
u = np.array([i['params'] for i in data])
s = np.array([i['solution'] for i in data])

n_samp = len(data)
train_indices, test_indices = train_test_split(np.arange(n_samp), test_size = 0.33, random_state = seed)
u_train, u_test = jnp.array(u[train_indices]), jnp.array(u[test_indices])
s_train, s_test = jnp.array(s[train_indices]), jnp.array(s[test_indices])


# --- batch utility ---
def get_point_batch(batch_size, params, s, x, t, key):
    """
    Build a batch of random samples, each as (params, [x,t]), target = s.
    
    Returns:
        u_batch: (batch_size, u_dim)
        y_batch: (batch_size, 2)  # (x,t)
        s_batch: (batch_size,)
        new_key
    """
    N, nt, nx = s.shape
    
    key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
    sample_idx = jax.random.randint(subkey1, (batch_size,), 0, N)
    t_idx      = jax.random.randint(subkey2, (batch_size,), 0, nt)
    x_idx      = jax.random.randint(subkey3, (batch_size,), 0, nx)
    
    u_batch = params[sample_idx]                # (batch_size, 2)
    y_batch = jnp.stack([x[x_idx], t[t_idx]], axis=1)  # (batch_size, 2)
    s_batch = s[sample_idx, t_idx, x_idx]       # (batch_size,)
    return (u_batch, y_batch), s_batch, key



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


## define just training and loss on branch components
@eqx.filter_jit
def branch_loss(branch_params, model, x, y):
    # model, updated branch params-> loss
    branch = eqx.combine(branch_params, branch_static) # basically undoes eqx.partition - restructures pytree
    model = eqx.tree_at(lambda m: m.branch, model, branch) # takes in model, and replaces branch with updated branch
    pred_y = jax.vmap(model)(x) # vmap model to x
    return jnp.mean((y - pred_y)**2)

@eqx.filter_jit
def branch_step(branch_params, opt_state, model, x, y, optimizer):
    loss, grads = eqx.filter_value_and_grad(branch_loss)(branch_params, model, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, branch_params)
    branch_params = eqx.apply_updates(branch_params, updates)
    return branch_params, opt_state, loss


## define just training and loss on trunk components
@eqx.filter_jit
def trunk_loss(trunk_params, model ,x, y):
    trunk = eqx.combine(trunk_params, trunk_static)
    model = eqx.tree_at(lambda m: m.trunk, model, trunk)
    pred_y = jax.vmap(model)(x)
    return jnp.mean((y - pred_y)**2)


@eqx.filter_jit
def trunk_step(trunk_params, opt_state, model, x, y, optimizer):
    loss, grads = eqx.filter_value_and_grad(trunk_loss)(trunk_params, model, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, trunk_params)
    trunk_params = eqx.apply_updates(trunk_params, updates)
    return trunk_params, opt_state, loss



u_dim = u_train.shape[1] # in this case, two 
y_dim = 2 # (x,t)
# define architectures for component networks
branch_arch = [u_dim, 40, 40, num_branches] 
trunk_arch  = [y_dim, 40, 40, num_branches] 


# model definition
print(f"initializing model, b_arch: {branch_arch}, t_arch: {trunk_arch}, num_branches: {num_branches}")
try:
    model = DeepONet(branch_arch = branch_arch, trunk_arch = trunk_arch, num_branches = num_branches, key = key)
    print("model initialized\n")
except:
    print("error initializing model\n")



branch_params, branch_static = eqx.partition(model.branch, eqx.is_inexact_array)
trunk_params, trunk_static = eqx.partition(model.trunk, eqx.is_inexact_array)



branch_opt = optax.adam(lr)
branch_state = branch_opt.init(branch_params)

trunk_opt = optax.adam(lr)
trunk_state = trunk_opt.init(trunk_params)

trunk_train_hist, trunk_test_hist = [],[]
branch_train_hist, branch_test_hist = [], []


print("beginning branch training...")
for epoch in tqdm(range(num_branch_epochs)):
    (u_batch, y_batch), s_batch, key = get_point_batch(batch_size, u_train, s_train, x_grid, t_grid, key)
    
    branch_params, branch_state, train_loss = branch_step(branch_params, branch_state, model, (u_batch, y_batch), s_batch, branch_opt)


    branch = eqx.combine(branch_params, branch_static)
    model = eqx.tree_at(lambda m: m.branch, model, branch)

    # test on full test set
    (u_all, y_all), s_all, _ = get_point_batch(512, u_test, s_test, x_grid, t_grid, key)
    test_loss = eval_step(model, (u_all, y_all), s_all)

    branch_train_hist.append(float(train_loss))
    branch_test_hist.append(float(test_loss))


last_branch_layer = model.branch.layers[-1]
W = last_branch_layer.weight


W_np = np.array(W)        # move weight to NumPy
Q_np, _ = scipy.linalg.qr(W_np.T, mode="economic")
Q = jnp.array(Q_np.T)     # back to JAX, shape matches W

new_last_branch = eqx.tree_at(
    lambda l: l.weight, last_branch_layer, Q)

model = eqx.tree_at(lambda m: m.branch.layers[-1], model, new_last_branch)




branch_params, branch_static = eqx.partition(model.branch, eqx.is_inexact_array)
trunk_params, trunk_static = eqx.partition(model.trunk, eqx.is_inexact_array)



print("beginning trunk training...")
for epoch in tqdm(range(num_trunk_epochs)):
    (u_batch, y_batch), s_batch, key = get_point_batch(batch_size, u_train, s_train, x_grid, t_grid, key)

    trunk_params, trunk_state, train_loss = trunk_step(trunk_params, trunk_state, model, (u_batch, y_batch), s_batch, trunk_opt)

    trunk = eqx.combine(trunk_params, trunk_static)
    model = eqx.tree_at(lambda m: m.trunk, model, trunk)

    (u_all, y_all), s_all, _ = get_point_batch(512, u_test, s_test, x_grid, t_grid, key)
    test_loss = eval_step(model, (u_all, y_all), s_all)

    trunk_train_hist.append(float(train_loss))
    trunk_test_hist.append(float(test_loss))








print(trunk_test_hist[-1])
print(branch_test_hist[-1])
print("training complete\n")
# serialize leaves via equinox
print("saving out...")
try:
    eqx.tree_serialise_leaves("deeponet_burgers_model.eqx", model)
    print("model saved to deeponet_burgers_model.eqx")
except:
    print("error exporting, model not saved")

# package export loss values
np.savez("deeponet_trunk_losses.npz", train=np.array(trunk_train_hist), test=np.array(trunk_test_hist))
np.savez("deeponet_branch_losses.npz", train=np.array(branch_train_hist), test=np.array(branch_test_hist))
