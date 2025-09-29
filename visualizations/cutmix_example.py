import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

# Load Oxford Flowers 102 dataset
ds, info = tfds.load("oxford_flowers102", split="train[:200]", with_info=True, as_supervised=True)
class_names = info.features["label"].names

IMG_SIZE = 128

def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

ds = ds.map(preprocess).batch(8)

import tensorflow as tf
import numpy as np

def get_cutmix_image(image_batch, label_batch, alpha=1.0, num_classes=102, img_size=128):
    """Applies CutMix to a batch of images and labels (NumPy mask version for visualization)."""
    batch_size = image_batch.shape[0]

    # Shuffle images and labels
    indices = np.random.permutation(batch_size)
    shuffled_images = image_batch.numpy()[indices]
    shuffled_labels = label_batch.numpy()[indices]

    # Sample lambda from Beta distribution
    lam = np.random.beta(alpha, alpha)

    # Cutout region
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(img_size * cut_rat)
    cut_h = int(img_size * cut_rat)

    # Random center
    cx = np.random.randint(img_size)
    cy = np.random.randint(img_size)

    # Bounding box
    bbx1 = np.clip(cx - cut_w // 2, 0, img_size)
    bby1 = np.clip(cy - cut_h // 2, 0, img_size)
    bbx2 = np.clip(cx + cut_w // 2, 0, img_size)
    bby2 = np.clip(cy + cut_h // 2, 0, img_size)

    # Build NumPy mask
    mask = np.ones((img_size, img_size, 3), dtype=np.float32)
    mask[bby1:bby2, bbx1:bbx2, :] = 0.0

    # Apply CutMix
    mixed_images = image_batch.numpy() * mask + shuffled_images * (1 - mask)

    # Adjust lambda based on exact area
    lam_adjusted = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img_size * img_size))

    # Mixed labels
    mixed_labels = lam_adjusted * tf.one_hot(label_batch, num_classes).numpy() + \
                   (1 - lam_adjusted) * tf.one_hot(shuffled_labels, num_classes).numpy()

    return tf.convert_to_tensor(mixed_images), tf.convert_to_tensor(mixed_labels)


images, labels = next(iter(ds))
mixed_images, mixed_labels = get_cutmix_image(images, labels, alpha=1.0, num_classes=len(class_names), img_size=IMG_SIZE)

plt.figure(figsize=(10, 10))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(mixed_images[i].numpy())
    label_probs = mixed_labels[i].numpy()
    top2 = label_probs.argsort()[-2:][::-1]
    label_str = f"{label_probs[top2[0]]:.2f} * {class_names[top2[0]]} + {label_probs[top2[1]]:.2f} * {class_names[top2[1]]}"
    plt.title(label_str, fontsize=10)
    plt.axis("off")

plt.suptitle("CutMix Examples on Oxford Flowers 102", fontsize=16)
plt.savefig("cutmix_flowers.png", dpi=300, bbox_inches="tight")
plt.show()
