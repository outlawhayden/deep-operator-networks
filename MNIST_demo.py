import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

# helper to randomly initialize w,b
def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale*random.normal(w_key, (n,m)), scale * random.normal(b_key, (n,))

# initialize all layers for FCNN, with sizes "sizes"
def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [random_layer_params(m,n,k) for m,n,k in zip(sizes[:-1], sizes[1:], keys)]

# network parameters
layer_sizes = [784, 512, 512, 10]
step_size = 0.01
num_epochs = 8
batch_size = 128
n_targets = 10
params = init_network_params(layer_sizes, random.key(0))

                  
from jax.scipy.special import logsumexp

def relu(x):
    return jnp.maximum(0,x)

def predict(params, image):
    # predict given single image, params
    activations = image
    for w,b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)

    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b
    return logits - logsumexp(logits)

random_flattened_image = random.normal(random.key(1), (28*28,))
preds = predict(params, random_flattened_image)
print(preds.shape)

random_flattened_images = random.normal(random.key(1), (10, 28*28))

batched_predict = vmap(predict, in_axes = (None, 0))

batched_preds = batched_predict(params, random_flattened_images)
print(batched_preds.shape)

def one_hot(x, k, dtype = jnp.float32):
    return jnp.array(x[:, None] == jnp.arange(k), dtype)

def accuracy(params, images, targets):
    target_class = jnp.argmax(targets, axis = 1)
    predicted_class = jnp.argmax(batched_predict(params, images), axis = 1)
    return jnp.mean(predicted_class == target_class)

def loss(params, images, targets):
    preds = batched_predict(params, images)
    return -jnp.mean(preds * targets)

@jit
def update(params, x, y):
    grads = grad(loss)(params, x, y)
    return [(w - step_size * dw, b - step_size * db)
            for (w,b), (dw, db) in zip(params, grads)]

import numpy as np
from jax.tree_util import tree_map
from torch.utils.data import DataLoader, default_collate
from torchvision.datasets import MNIST

def numpy_collate(batch):
    return tree_map(np.asarray, default_collate(batch))

def flatten_and_cast(pic):
    return np.ravel(np.array(pic, dtype = jnp.float32))


mnist_dataset = MNIST('/tmp/mnist/', download = True, transform = flatten_and_cast)
training_generator = DataLoader(mnist_dataset, batch_size = batch_size, collate_fn = numpy_collate)

train_images = np.array(mnist_dataset.train_data).reshape(len(mnist_dataset.train_data), -1)
train_labels = one_hot(np.array(mnist_dataset.train_labels), n_targets)

mnist_dataset_test = MNIST('/tmp/mnist/', download = True, train = False)
test_images = jnp.array(mnist_dataset_test.test_data.numpy().reshape(len(mnist_dataset_test.test_data), -1), dtype = jnp.float32)
test_labels = one_hot(np.array(mnist_dataset_test.test_labels), n_targets)



import time
for epoch in range(num_epochs):
    start_time = time.time()
    for x,y in training_generator:
        y = one_hot(y, n_targets)
        params = update(params, x,y)
    epoch_time = time.time() - start_time

    train_acc = accuracy(params, train_images, train_labels)
    test_acc = accuracy(params, test_images, test_labels)

    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
    print("Training set accuracy {}".format(train_acc))
    print("Test set accuracy {}".format(test_acc))



