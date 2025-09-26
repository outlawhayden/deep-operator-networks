import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve
from scipy.spatial.distance import cdist

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
D = 0.001
k = 0.001
Nx = 128
Nt = 128
x_left, x_right = 0.0, 1.0
t_final = 1.0

n_traj = 100   # number of independent GRF source terms
length_scale = 0.2
variance = 1.0
seed = 42

dx = (x_right - x_left) / (Nx - 1)
dt = t_final / Nt
x = np.linspace(x_left, x_right, Nx)
t = np.linspace(0, t_final, Nt+1)

# ------------------------------------------------------------
# GRF sampler
# ------------------------------------------------------------
def rbf_kernel(X, Y, length_scale=1.0, variance=1.0):
    dists = cdist(X, Y, metric="sqeuclidean")
    return variance * np.exp(-0.5 * dists / length_scale**2)

def sample_grf(n_points, length_scale=1.0, variance=1.0, n_samples=1, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    X = np.linspace(x_left, x_right, n_points)[:, None]
    K = rbf_kernel(X, X, length_scale, variance)
    samples = rng.multivariate_normal(mean=np.zeros(n_points), cov=K, size=n_samples)
    return X.flatten(), samples

# ------------------------------------------------------------
# Build Crank–Nicolson matrices (interior points only)
# ------------------------------------------------------------
main_diag = -2.0 * np.ones(Nx)
off_diag = np.ones(Nx-1)
L = diags([off_diag, main_diag, off_diag], [-1,0,1], format="csr") / dx**2
L = L[1:-1,1:-1]

I = eye(Nx-2, format="csr")
A = I - 0.5*dt*(D*L - k*I)
B = I + 0.5*dt*(D*L - k*I)

# ------------------------------------------------------------
# Generate dataset
# ------------------------------------------------------------
rng = np.random.default_rng(seed)
_, v_samples = sample_grf(Nx, length_scale, variance, n_samples=n_traj, rng=rng)

sources = []
solutions = []

for i in range(n_traj):
    # Strictly positive source term
    v = v_samples[i]
    u_source = v - np.min(v) + 1.0
    sources.append(u_source)

    # Solve PDE with Crank–Nicolson
    s = np.zeros(Nx)            # initial condition: zero
    U = np.zeros((Nt+1, Nx))    # solution over time
    U[0,:] = s

    for n in range(Nt):
        s_interior = s[1:-1]
        rhs = B @ s_interior + dt * u_source[1:-1]   # forcing
        s_new_interior = spsolve(A, rhs)
        s_new = np.zeros_like(s)
        s_new[1:-1] = s_new_interior
        s = s_new
        U[n+1,:] = s

    solutions.append(U)

sources = np.array(sources)       # (n_traj, Nx)
solutions = np.array(solutions)   # (n_traj, Nt+1, Nx)

# ------------------------------------------------------------
# Save dataset
# ------------------------------------------------------------
dataset = {
    "x": x,               # spatial grid
    "t": t,               # time grid
    "sources": sources,   # GRF source terms, shape (n_traj, Nx)
    "solutions": solutions  # PDE solutions, shape (n_traj, Nt+1, Nx)
}

np.save("pde_source_solution_dataset.npy", dataset, allow_pickle=True)
print("Saved to: pde_source_solution_dataset.npy")
print("Dataset keys:", dataset.keys())
print("sources shape:", sources.shape)
print("solutions shape:", solutions.shape)
