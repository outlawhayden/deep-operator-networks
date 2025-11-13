import jax.numpy as jnp
import numpy as np
import jax
import os
import optax
import scipy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import equinox as eqx
import scipy.linalg
from jax.nn.initializers import he_normal
from tqdm import tqdm
import csv

# -------------------------------------------------
# config
# -------------------------------------------------
seed = 42
np.random.seed(seed)
jax.config.update("jax_platform_name", "METAL")

batch_size = 64
num_branches = 15
num_trunk_epochs = 120000
lr = 1e-5
num_LBFGS_epochs = 10000
time_block = 5  # grow the time window in chunks of 5

print("\nconfiguring backend...")
print("backend selected:\n", jax.default_backend())
print("active devices:\n", jax.devices())
print("--------------------\n")

# -------------------------------------------------
# load the full dataset once
# -------------------------------------------------
dataset = np.load(
    "/Users/haydenoutlaw/Documents/Research/opnet/deep-operator-networks/data/burgers_dataset.npz",
    allow_pickle=True,
)
full_t_grid_all = jnp.array(dataset["t"])
x_grid_full = jnp.array(dataset["x"])
data_all = dataset["samples"]
u_all = np.array([i["params"] for i in data_all])
s_all = np.array([i["solution"] for i in data_all])

total_timesteps_available = len(full_t_grid_all)

# figure out which time prefixes we'll run
time_prefix_list = list(range(time_block, total_timesteps_available + 1, time_block))
# if not divisible by block size, include the tail
if time_prefix_list[-1] != total_timesteps_available:
    time_prefix_list.append(total_timesteps_available)
print("time segments:\n", time_prefix_list)
results = []

csv_path = "time_sweep_results.csv"
write_header = not os.path.exists(csv_path)
csv_file = open(csv_path, "a", newline="")
writer = csv.writer(csv_file)

if write_header:
    writer.writerow(["num_timesteps_used", "t_max_value", "final_loss", "deepOnet_err"])
    csv_file.flush()


# -------------------------------------------------
# sweep over increasing time windows
# -------------------------------------------------
for num_timesteps in time_prefix_list:
    print("\n==============================")
    print(f"Training with first {num_timesteps} time steps")
    print("==============================\n")

    # fresh key per experiment but reproducible
    key = jax.random.key(seed)
    key = jax.random.fold_in(key, num_timesteps)

    # -----------------------------------------------------------------
    # slice dataset to first num_timesteps in time
    # -----------------------------------------------------------------
    t_grid = full_t_grid_all[:num_timesteps]   # shape (num_timesteps,)
    x_grid = x_grid_full                       # unchanged
    s = s_all[:, :num_timesteps, :]            # (n_samples, num_timesteps, Nx)
    u = u_all                                  # (n_samples, 2) params

    n_samp = len(data_all)
    train_indices, test_indices = train_test_split(
        np.arange(n_samp), test_size=0.33, random_state=seed
    )
    u_train, u_test = jnp.array(u[train_indices]), jnp.array(u[test_indices])
    s_train, s_test = jnp.array(s[train_indices]), jnp.array(s[test_indices])

    # build xt grid for these timesteps
    xx, tt = jnp.meshgrid(x_grid, t_grid)
    xt_grid = jnp.concatenate(
        [xx.flatten()[:, None], tt.flatten()[:, None]], axis=1
    )
    print("xt_grid shape:", xt_grid.shape)

    # reshape solutions the same way you did before, but generalizing 1005
    # 1005 = 5 * len(x_grid) in your original run
    num_x = len(x_grid)
    ss_grid = s_train.reshape([-1, num_timesteps * num_x]).T
    print("ss_grid shape:", ss_grid.shape)

    # -------------------------------------------------
    # define your models exactly like before
    # -------------------------------------------------

    # Linear Model (Wx + b)
    class Linear(eqx.Module):
        weight: jax.Array
        bias: jax.Array

        def __init__(self, in_size, out_size, key, initializer=he_normal()):
            wkey, bkey = jax.random.split(key)
            self.weight = initializer(
                wkey, (out_size, in_size), dtype=jnp.float32
            )
            self.bias = jnp.zeros((out_size,), dtype=jnp.float32)

        def __call__(self, x):
            return self.weight @ x + self.bias

    # Basic MultiLayer Perceptron Primitive
    class MLP(eqx.Module):
        layers: list
        activations: list

        def __init__(
            self,
            architecture,
            key,
            activation=jax.nn.relu,
            initializer=he_normal(),
        ):
            """
            architecture: List[in, hidden1, hidden2, ..., out]
            key: random key
            activation: jax.nn act fxn
            """
            keys = jax.random.split(key, len(architecture) - 1)
            self.layers = [
                Linear(
                    architecture[i],
                    architecture[i + 1],
                    keys[i],
                    initializer=initializer,
                )
                for i in range(len(architecture) - 1)
            ]
            # no activation on last layer
            self.activations = [activation] * (len(self.layers) - 1) + [
                eqx.nn.Identity()
            ]

        def __call__(self, x):
            for layer, act in zip(self.layers, self.activations):
                x = act(layer(x))
            return x

    u_dim = 2  # params dim (stays 2)
    y_dim = 2  # (x,t)
    branch_arch = [u_dim, 40, 40, num_branches]
    trunk_arch = [y_dim, 40, 40, num_branches]

    key, subkey_t, subkey_A = jax.random.split(key, num=3)
    trunk_model = MLP(trunk_arch, key=subkey_t)
    A_model = jax.random.normal(subkey_A, (num_branches, s_train.shape[0]))

    model = (trunk_model, A_model)

    def deeponet_step1(model, u_tr):
        trunk_model, A_model = model
        trunk_mat = jax.vmap(trunk_model)(u_tr)
        return (trunk_mat @ A_model)

    def loss_fn(model, u_in, s_true):
        pred_y = deeponet_step1(model, u_in)
        return jnp.mean((s_true - pred_y) ** 2)
    



    # -------------------------------------------------
    # 1. Pretrain with Adam (unchanged)
    # -------------------------------------------------
    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    loss_hist = []

    @eqx.filter_jit
    def train_step(model, opt_state, batch):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, *batch)
        updates, opt_state = opt.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    for step in tqdm(range(num_trunk_epochs), desc=f"Adam T={num_timesteps}", leave=False):
        model, opt_state, loss = train_step(model, opt_state, (xt_grid, ss_grid))
        loss_hist.append(float(loss))
        if step % 1000 == 0:
            print(f"[T={num_timesteps}] Adam step {step}: loss={float(loss):.3e}")

    print(f"[T={num_timesteps}] final adam loss:", float(loss))

    # -------------------------------------------------
    # 2. LBFGS refinement (exact same pattern you used)
    # -------------------------------------------------
    opt = optax.lbfgs()  # rebind the global name 'opt'; train_step will now
                         # see this new opt in subsequent calls

    for step in tqdm(range(num_LBFGS_epochs), desc=f"LBFGS T={num_timesteps}", leave=False):
        model, opt_state, loss = train_step(model, opt_state, (xt_grid, ss_grid))
        loss_hist.append(float(loss))
        if step % 1000 == 0:
            print(f"[T={num_timesteps}] LBFGS step {step}: loss={float(loss):.3e}")

    # -------------------------------------------------
    # final loss for this time prefix
    # -------------------------------------------------
    final_loss_val = float(loss_fn(model, xt_grid, ss_grid))
    t_max_val = float(t_grid[-1])

    print(
        f"[T={num_timesteps}] Final training loss after LBFGS: {final_loss_val:.6e}"
    )
    print(f"[T={num_timesteps}] t_max = {t_max_val}")

    # we'll also compute your "deepOnet_err" metric you had before
    deepOnet_err_val = float(np.mean(ss_grid - deeponet_step1(model, xt_grid)))


    # stash result
    results.append(
        {
            "num_timesteps_used": int(num_timesteps),
            "t_max_value": t_max_val,
            "final_loss": final_loss_val,
            "deepOnet_err": deepOnet_err_val,
        }
    )

# -------------------------------------------------
# write sweep summary to CSV
# -------------------------------------------------
        
    writer.writerow(
        [
            int(num_timesteps),
            t_max_val,
            final_loss_val,
            deepOnet_err_val,
        ]
        )
    csv_file.flush()

print("\nAll done. Results written to", csv_path)

