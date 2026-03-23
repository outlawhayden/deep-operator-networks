
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
trunk_model = MLP(trunk_arch,key = subkey_t)
A_model = jax.random.normal(subkey_A, (num_bases, output_tr.shape[2])) # now A_model is a 2D matrix


model = trunk_model, A_model
best_model = model

master_end_time = time.time()


print("training dataset size:", u_train.shape, s_train.shape)
print("testing dataset size", u_test.shape, s_test.shape)
print("Total Time (sec):", master_end_time - master_start_time)