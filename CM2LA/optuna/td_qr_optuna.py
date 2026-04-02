## hyperparameter optimization using optuna
## optimizes trunk/branch architectures and training parameters
## saves study data after each trial

import os
import yaml
from pathlib import Path

config_path = Path("config.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

gpu_idx = config.get("gpu_idx", 9)
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

import jax.numpy as jnp
import numpy as np
import jax
import optax
from jaxopt import LBFGS
import scipy
import scipy.linalg
import equinox as eqx
from jax.nn.initializers import he_normal
from sklearn.model_selection import train_test_split
import time
import optuna
import json

jax.config.update("jax_platform_name", "gpu")
print("backend selected:\n", jax.default_backend())
print("active devices:\n", jax.devices())
print("--------------------\n")

# Fixed params
seed = config.get('seed', 42)
num_bases = 4
eps = config.get('eps', 1e-8)
dataset_path = Path(config.get('dataset_path', 'burgers_dataset.npz'))
test_size = config.get('test_size', 0.2)

np.random.seed(seed)

# Load dataset once (outside objective)
dataset = np.load(dataset_path, allow_pickle=True)
t_grid = jnp.array(dataset['t'])
x_grid = jnp.array(dataset['x'])
data = dataset['samples']
u = np.array([i['params'] for i in data])
s = np.array([i['solution'] for i in data])

n_samp = len(data)
train_indices, test_indices = train_test_split(np.arange(n_samp), test_size=test_size, random_state=seed)
u_train, u_test = jnp.array(u[train_indices]), jnp.array(u[test_indices])
s_train, s_test = jnp.array(s[train_indices]), jnp.array(s[test_indices])
output_tr = jnp.transpose(s_train, axes=(1, 2, 0))


## EQUINOX CLASS DEFINITIONS
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

    def __init__(self, architecture, key, activation=jax.nn.leaky_relu, initializer=he_normal()):
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


def run_trial(trial_params, key):
    trunk_arch = trial_params['trunk_arch']
    branch_arch = trial_params['branch_arch']
    num_trunk_epochs = trial_params['num_trunk_epochs']
    num_LBFGS_epochs = trial_params['num_LBFGS_epochs']
    lr_start = trial_params['lr_start']
    lr_transition_steps = trial_params['lr_transition_steps']
    lr_decay_rate = trial_params['lr_decay_rate']
    lr_end = trial_params['lr_end']
    lr_start_branch = trial_params['lr_start_branch']
    lr_transition_steps_branch = trial_params['lr_transition_steps_branch']
    lr_decay_rate_branch = trial_params['lr_decay_rate_branch']
    lr_end_branch = trial_params['lr_end_branch']

    trunk_arch[-1] = num_bases
    branch_arch[-1] = num_bases
    branch_arch[0] = branch_arch[0] + 1

    key, subkey_t, subkey_A = jax.random.split(key, num=3)
    trunk_model = MLP(trunk_arch, key=subkey_t)
    A_model = jax.random.normal(subkey_A, (len(t_grid), num_bases, output_tr.shape[2]))
    model = trunk_model, A_model

    def loss_fn(model):
        trunk_model, A_model = model
        T_MAT = jax.vmap(lambda t: jax.vmap(lambda x: trunk_model(jnp.stack([x, t])))(x_grid))(t_grid)
        pred_y = jnp.einsum("txk, tkn -> txn", T_MAT, A_model)
        diff = pred_y - output_tr
        return jnp.mean(jnp.linalg.norm(diff, axis=(0, 1)))

    lr_schedule = optax.schedules.exponential_decay(
        init_value=lr_start,
        transition_steps=lr_transition_steps,
        decay_rate=lr_decay_rate,
        end_value=lr_end
    )
    opt = optax.adam(lr_schedule)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def train_step(model, opt_state):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = opt.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    for step in range(num_trunk_epochs):
        model, opt_state, loss = train_step(model, opt_state)
        if step % 10000 == 0:
            print(f"\r  [trunk] Adam step {step}: loss={float(loss):.3e}", end="", flush=True)
    print(f"\n  [trunk] final adam loss: {float(loss):.3e}")

    # LBFGS trunk
    params = eqx.filter(model, eqx.is_inexact_array)
    static = jax.tree_util.tree_map(lambda x: None if eqx.is_inexact_array(x) else x, model)
    frozen_loss = jax.jit(lambda p: loss_fn(eqx.combine(p, static)))
    _ = frozen_loss(params)
    jax.block_until_ready(_)

    lbfgs_solver = LBFGS(fun=frozen_loss, maxiter=num_LBFGS_epochs, tol=1e-6,
                         history_size=10, implicit_diff=False, stepsize=-1.0)
    params, lbfgs_state = lbfgs_solver.run(params)
    print(f"  [trunk] LBFGS final loss: {float(lbfgs_state.value):.3e}")
    model = eqx.combine(params, static)

    # QR factorization
    trunk_model, A_model = model
    T_MAT = jax.vmap(lambda t: jax.vmap(lambda x: trunk_model(jnp.stack([x, t])))(x_grid))(t_grid)

    Q_MAT = None
    R_MAT = None
    for i in range(len(t_grid)):
        Q, R = scipy.linalg.qr(T_MAT[i], mode='economic')
        if i == 0:
            Q_MAT = Q[jnp.newaxis, :, :]
            R_MAT = R[jnp.newaxis, :, :]
        else:
            Q_MAT = jnp.concatenate([Q_MAT, Q[jnp.newaxis, :, :]], axis=0)
            R_MAT = jnp.concatenate([R_MAT, R[jnp.newaxis, :, :]], axis=0)

    Q_sign = np.array(Q_MAT, copy=True)
    R_sign = np.array(R_MAT, copy=True)
    T_len = Q_sign.shape[0]
    n = Q_sign.shape[2]
    for k in range(1, T_len):
        for j in range(n):
            if np.dot(Q_sign[k-1, :, j], Q_sign[k, :, j]) < 0:
                Q_sign[k, :, j] *= -1
                R_sign[k, j, :] *= -1

    # Branch training
    key, subkey_b = jax.random.split(key)
    branch_model = MLP(branch_arch, key=subkey_b)

    RA_model = jnp.einsum("tij, tjk -> tik", R_sign, A_model)
    RA_target = jnp.asarray(RA_model)

    def branch_loss_fn(branch_model, RA_target, u_train, t_grid):
        B_tnK = jax.vmap(
            lambda t: jax.vmap(
                lambda u: branch_model(jnp.concatenate([u, jnp.array([t])]))
            )(u_train)
        )(t_grid)
        B_MAT = jnp.swapaxes(B_tnK, 1, 2)
        diff = B_MAT - RA_target
        return jnp.mean(jnp.linalg.norm(diff, axis=(0, 1)))

    lr_schedule_branch = optax.schedules.exponential_decay(
    init_value=lr_start_branch,
    transition_steps=lr_transition_steps_branch,
    decay_rate=lr_decay_rate_branch,
    end_value=lr_end_branch
)
    opt_b = optax.adam(lr_schedule_branch)
    opt_state_b = opt_b.init(eqx.filter(branch_model, eqx.is_inexact_array))

    @eqx.filter_jit
    def branch_train_step(branch_model, opt_state, RA_target, u_train, t_grid):
        loss, grads = eqx.filter_value_and_grad(branch_loss_fn)(
            branch_model, RA_target, u_train, t_grid
        )
        updates, opt_state = opt_b.update(grads, opt_state, branch_model)
        branch_model = eqx.apply_updates(branch_model, updates)
        return branch_model, opt_state, loss

    for step in range(num_trunk_epochs):
        branch_model, opt_state_b, loss = branch_train_step(
            branch_model, opt_state_b, RA_target, u_train, t_grid
        )
        if step % 10000 == 0:
            print(f"\r  [branch] Adam step {step}: loss={float(loss):.3e}", end="", flush=True)
    print(f"\n  [branch] final adam loss: {float(loss):.3e}")

    # Evaluate test L2
    def predict_solution(Q_sign, branch_model, u, t_grid):
        b_tk = jax.vmap(lambda t: branch_model(jnp.concatenate([u, jnp.array([t])])))(t_grid)
        return jnp.einsum("txk,tk->tx", Q_sign, b_tk)

    preds = jax.vmap(lambda u: predict_solution(jnp.asarray(Q_sign), branch_model, u, t_grid))(u_test)
    test_l2 = float(jnp.mean(jnp.linalg.norm(preds - s_test, axis=(1, 2))))
    return test_l2


def objective(trial):
    # Architecture search space
    n_trunk_hidden = trial.suggest_int("n_trunk_hidden", 1, 4)
    trunk_hidden_size = trial.suggest_categorical("trunk_hidden_size", [32, 64, 128, 256])
    trunk_arch = [2] + [trunk_hidden_size] * n_trunk_hidden + [num_bases]

    n_branch_hidden = trial.suggest_int("n_branch_hidden", 1, 4)
    branch_hidden_size = trial.suggest_categorical("branch_hidden_size", [32, 64, 128, 256])
    branch_arch = [2] + [branch_hidden_size] * n_branch_hidden + [num_bases]

    # Training parameter search space
    num_trunk_epochs = trial.suggest_int("num_trunk_epochs", 50000, 2000000, step=50000)
    num_LBFGS_epochs = trial.suggest_int("num_LBFGS_epochs", 1000, 50000, step=1000)
    lr_start = trial.suggest_float("lr_start", 1e-4, 1e-1, log=True)
    lr_transition_steps = trial.suggest_int("lr_transition_steps", 1000, 20000, step=1000)
    lr_decay_rate = trial.suggest_float("lr_decay_rate", 0.5, 0.999)
    lr_end = trial.suggest_float("lr_end", 1e-6, 1e-3, log=True)

    lr_start_branch = trial.suggest_float("lr_start_branch", 1e-4, 1e-1, log=True)
    lr_transition_steps_branch = trial.suggest_int("lr_transition_steps_branch", 1000, 20000, step=1000)
    lr_decay_rate_branch = trial.suggest_float("lr_decay_rate_branch", 0.5, 0.999)
    lr_end_branch = trial.suggest_float("lr_end_branch", 1e-6, 1e-3, log=True)

    trial_params = {
        'trunk_arch': trunk_arch,
        'branch_arch': branch_arch,
        'num_trunk_epochs': num_trunk_epochs,
        'num_LBFGS_epochs': num_LBFGS_epochs,
        'lr_start': lr_start,
        'lr_transition_steps': lr_transition_steps,
        'lr_decay_rate': lr_decay_rate,
        'lr_end': lr_end,
        'lr_start_branch': lr_start_branch,
        'lr_transition_steps_branch': lr_transition_steps_branch,
        'lr_decay_rate_branch': lr_decay_rate_branch,
        'lr_end_branch': lr_end_branch,
    }

    print(f"\n--- Trial {trial.number} ---")
    print(json.dumps({k: str(v) for k, v in trial_params.items()}, indent=2))

    key = jax.random.key(seed + trial.number)
    try:
        test_l2 = run_trial(trial_params, key)
    except Exception as e:
        print(f"  Trial failed: {e}")
        raise optuna.exceptions.TrialPruned()

    print(f"  Trial {trial.number} test L2: {test_l2:.6e}")
    return test_l2


def save_callback(study, trial):
    """Save study results to JSON after each trial."""
    out = {
        "best_trial": {
            "number": study.best_trial.number,
            "value": study.best_trial.value,
            "params": study.best_trial.params,
        },
        "all_trials": [
            {
                "number": t.number,
                "value": t.value if t.value is not None else None,
                "state": str(t.state),
                "params": t.params,
            }
            for t in study.trials
        ]
    }
    with open("optuna_study_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"  [saved] study results -> optuna_study_results.json")


if __name__ == "__main__":
    n_trials = config.get("n_optuna_trials", 30)

    # Use a persistent SQLite storage so the study survives crashes
    storage = optuna.storages.RDBStorage("sqlite:///optuna_study.db")
    study = optuna.create_study(
        study_name="orth_td_hparam_search",
        direction="minimize",
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials, callbacks=[save_callback])

    print("\n==============================")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best test L2: {study.best_trial.value:.6e}")
    print(f"Best params: {study.best_trial.params}")
    print("==============================")