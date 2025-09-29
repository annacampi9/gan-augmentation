import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import cifar10

# Load CIFAR-10
(x_train, y_train), (_, _) = cifar10.load_data()
y_train = y_train.flatten()

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

n_classes = len(class_names)
n_images = 8  # number of examples per class

# Create grid
fig, axes = plt.subplots(n_classes, n_images, figsize=(12, 10),
                         gridspec_kw={'wspace':0.05, 'hspace':0.05})

for class_idx, class_name in enumerate(class_names):
    idxs = np.where(y_train == class_idx)[0][:n_images]
    for j, idx in enumerate(idxs):
        ax = axes[class_idx, j]
        ax.imshow(x_train[idx])
        ax.axis('off')

    # Add class name on the left margin (bigger & bold)
    axes[class_idx, 0].text(-40, 16, class_name,
                            fontsize=12, fontweight='bold',
                            ha='right', va='center')

plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.05)
plt.show()
