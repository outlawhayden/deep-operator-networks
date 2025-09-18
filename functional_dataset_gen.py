import jax.numpy as jnp
import numpy as np
import jax
from flax import nnx
import optax
from flax.training import train_state
import scipy
from scipy.spatial.distance import cdist
from scipy.integrate import RK45
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid

seed = 42
np.random.seed(42)
n_traj = 10000 # number of initial conditions
n_points = 100 # number of samples on domain
length_scale = 0.25 # smoothness of ICs
variance = 1.0 # variance of ICs
x_left = 0
x_right = 1


print("Number of Trajectories:", n_traj)
print("Number of Samples/Trajectory:", n_points)
print("Dataset Size:", n_traj * n_points)


# kernel for gaussian random field function
def rbf_kernel(X, Y, length_scale = 1.0, variance = 1.0):
    dists = cdist(X, Y, metric = 'sqeuclidean')
    return variance * np.exp(-0.5 * dists/ length_scale ** 2)

# sample from gaussian random field, how you generate random initial conditions
# n_points = smoothness on domain [x_left,x_right], length_scale controls smoothness
def sample_grf(n_points = 100, n_dims = 1, length_scale = 1.0, variance = 1.0, n_samples = 5, random_state = seed):
    rng = np.random.default_rng(random_state)
    if n_dims == 1:
        X = np.linspace(x_left, x_right, n_points)[:, None]
    else:
        X = rng.uniform(x_left, x_right, size = (n_points, n_dims))

    K = rbf_kernel(X,X, length_scale = length_scale, variance = variance)
    samples = rng.multivariate_normal(mean = np.zeros(n_points), cov = K, size = n_samples)
    return X, samples

# samples are u(x)
x, samples = sample_grf(n_points = n_points, n_dims = 1, length_scale= length_scale, variance = variance, n_samples = n_traj, random_state = seed)
x = x.reshape(-1)

integrals = np.zeros_like(samples) # preallocate memory

# use cumulative trapezoid function to approximate int_0^x u(x) = F
# could be vectorized?
for i in range(n_traj):
    u = samples[i]
    F = cumulative_trapezoid(u,x, initial = 0)
    integrals[i] = F

print("Samples shape:", samples.shape)
print("Integrals shape:", integrals.shape)

# Repeat x across all trajectories: shape (n_traj, n_points)
x_tiled = np.tile(x, (n_traj, 1))

# Stack along the last axis → (n_traj, n_points, 3)
dataset = np.stack([x_tiled, samples, integrals], axis=-1)
print("Total Dataset Shape:", dataset.shape)
np.save('integral_dataset.npy', dataset) # needs to be .npy because dataset is 3d, otherwise ravel to csv
print("Saved to: integral_dataset.npy")
print("Done")

"""
axis 0 (index i, length n_traj: trajectory number (which IC))
axis 1 (index j, length n_points): position on domain 
axis 2 (index k): which attribute. 0 = x_k, 1 = u_i(x_k), 2 = F_i(x_j)
"""