# === DeepONet Evaluation and Visualization Script ===

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# ----------------------------------------------------
# Config
# ----------------------------------------------------
## DOUBLE CHECK - MUST MATCH TRAINING ARCHITECTURE
seed = 42
np.random.seed(seed)
num_branches = 40

# ----------------------------------------------------
# Model Definitions (must match training script)
# ----------------------------------------------------
class Linear(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __init__(self, in_size, out_size, key):
        wkey, bkey = jax.random.split(key)
        self.weight = jax.random.normal(wkey, (out_size, in_size))
        self.bias = jax.random.normal(bkey, (out_size))

    def __call__(self, x):
        return self.weight @ x + self.bias


class MLP(eqx.Module):
    layers: list
    activations: list

    def __init__(self, architecture, key, activation=jax.nn.relu):
        keys = jax.random.split(key, len(architecture) - 1)
        self.layers = [
            Linear(architecture[i], architecture[i + 1], keys[i])
            for i in range(len(architecture) - 1)
        ]
        self.activations = [activation] * (len(self.layers) - 1) + [eqx.nn.Identity()]

    def __call__(self, x):
        for layer, act in zip(self.layers, self.activations):
            x = act(layer(x))
        return x


class DeepONet(eqx.Module):
    branch: MLP
    trunk: MLP

    def __init__(self, branch_arch, trunk_arch, key, num_branches, activation=jax.nn.relu):
        bkey, tkey = jax.random.split(key)
        self.branch = MLP(branch_arch, bkey, activation)
        self.trunk = MLP(trunk_arch, tkey, activation)

    def __call__(self, x):
        u, y = x
        branch_out = self.branch(jnp.atleast_1d(u))
        trunk_out = self.trunk(jnp.atleast_1d(y))
        return jnp.inner(branch_out, trunk_out)

# ----------------------------------------------------
# Load dataset (must match training preprocessing)
# ----------------------------------------------------
print("loading dataset...")
try:
    dataset = np.load("data/integral_dataset.npy")
    print("dataset loaded\n")
except:
    print("error loading dataset\n")


u_all = dataset[:, :, 1]  # function at x_i
x_all = dataset[:, :, 0]  # sensor locations
F_all = dataset[:, :, 2]  # target values

n_traj, n_points = u_all.shape
u_vectors = np.repeat(u_all, repeats=n_points, axis=0)
y_points = x_all.reshape(-1, 1)
targets = F_all.reshape(-1)

indices = np.arange(len(targets))
train_idx, test_idx = train_test_split(indices, test_size=0.33, random_state=seed)

X_u_test = jnp.array(u_vectors[test_idx])
X_y_test = jnp.array(y_points[test_idx])
y_test = jnp.array(targets[test_idx])

u_dim = u_vectors.shape[1]
y_dim = 1
branch_arch = [u_dim, 40, 40, num_branches]
trunk_arch = [y_dim, 40, 40, num_branches]

# ----------------------------------------------------
# Load trained model
# ----------------------------------------------------
key = jax.random.key(seed)
print("loading DeepONet...")
try:
    model = DeepONet(branch_arch, trunk_arch, key, num_branches)
    model = eqx.tree_deserialise_leaves("deeponet_model.eqx", model)
    print("model loaded.\n")
except:
    print("error loading model.\n")
# ----------------------------------------------------
# Plot training & test losses
# ----------------------------------------------------
loss_data = np.load("deeponet_losses.npz")
train_losses = loss_data["train"]
test_losses = loss_data["test"]

plt.figure(figsize=(12, 6))
sns.lineplot(x=np.arange(len(train_losses)), y=train_losses, label="Train Loss")
sns.lineplot(x=np.arange(len(test_losses)), y=test_losses, label="Test Loss")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss (log scale)")
plt.title("Training vs Test Loss (DeepONet)")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curves.png", dpi=300)
plt.close()

print("Saved loss render to loss_curves.png")

# ----------------------------------------------------
# Evaluate on full domain for one test trajectory
# ----------------------------------------------------
# pick one trajectory index from test set
traj_idx = np.random.randint(0, n_traj)

u_fixed = u_all[traj_idx]              # function values at sensors
y_domain = x_all[traj_idx]             # domain points
true_curve = F_all[traj_idx]           # true integral values

# Repeat u_fixed for every y in the domain
u_repeated = jnp.array([u_fixed] * len(y_domain))
y_points_full = jnp.array(y_domain).reshape(-1, 1)

# Model predictions across the domain
pred_curve = jax.vmap(lambda u, y: model((u, y)))(u_repeated, y_points_full)

# Plot full prediction vs ground truth
plt.figure(figsize=(12, 6))
plt.plot(y_domain, true_curve, label="True Curve", linewidth=3)
plt.plot(y_domain, pred_curve, label="Predicted Curve", linewidth = 3, linestyle="--")
plt.xlabel("x")
plt.ylabel("Integral Value")
plt.title("DeepONet Prediction Across Domain")
plt.legend()
plt.tight_layout()
plt.savefig("full_domain_prediction.png", dpi=300)
plt.close()

print("Saved full domain prediction to full_domain_prediction.png")

plt.figure(figsize = (12,6))
plt.plot(y_domain, true_curve - pred_curve, label = "Error", linewidth = 3)
plt.xlabel("x")
plt.ylabel("Error")
plt.title("DeepONet Approximation Errror (True - Pred)")
plt.legend()
plt.tight_layout()
plt.savefig("full_domain_error.png", dpi = 300)

print("Saved prediction error to full_domain_error.png")
print("\nexiting...")
