import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


loss_data = np.load('data/deeponet_losses.npz')
train_data = loss_data["train"]
test_data = loss_data["test"]


epochs = np.arange(1, len(train_data) + 1)

sns.lineplot(x=epochs, y=train_data, label="Train Loss")
sns.lineplot(x=epochs, y=test_data, label="Test Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale("log")
plt.title("Training vs Test Loss")
plt.legend()
plt.show()
