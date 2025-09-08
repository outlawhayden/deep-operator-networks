import jax.numpy as jnp
from flax import nnx
import optax
from flax.training import train_state



class SimpleNN(nnx.Module):
    def __init__(self, n_features: int = 64, n_hidden = 100, n_targets: int = 10, *, rngs: nnx.Rngs):
        self.n_features = n_features
        self.layer1 = nnx.Linear(n_features, n_hidden, rngs = rngs)
        self.layer2 = nnx.Linear(n_hidden, n_hidden, rngs = rngs)
        self.layer3 = nnx.Linear(n_hidden, n_targets, rngs = rngs)

    def __call__(self, x):
        x = x.reshape(x.shape[0], self.n_features)
        x = nnx.relu(self.layer1(x))
        x = nnx.relu(self.layer2(x))
        x = self.layer3(x)
        return x

model = SimpleNN(rngs = nnx.Rngs(0))

nnx.display(model)

def l2_loss(model, batch):
    x,y = batch
    preds = model(x)
    return jnp.mean((preds-y) ** 2)

learning_rate = 0.005
momentum = 0.9

optimizer = nnx.Optimizer(
        model, optax.adamw(learning_rate, momentum), wrt = nnx.Param
        )

nnx.display(optimizer)
