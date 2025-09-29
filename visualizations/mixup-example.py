import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib

# Download flower dataset
data_dir = tf.keras.utils.get_file(
    origin="https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
    fname="flower_photos", untar=True)
data_dir = pathlib.Path(data_dir)

# Load dataset
img_height, img_width = 180, 180
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names

def sample_beta_distribution(size, concentration=0.4):
    gamma_1 = tf.random.gamma([size], concentration)
    gamma_2 = tf.random.gamma([size], concentration)
    return gamma_1 / (gamma_1 + gamma_2)

def mixup_for_visualization(x, y, lam=0.5):
    """Force mixup with a fixed λ for clear overlays"""
    batch_size = tf.shape(x)[0]
    index = tf.random.shuffle(tf.range(batch_size))
    x1, x2 = x, tf.gather(x, index)
    y1, y2 = y, tf.gather(y, index)

    mixed_x = lam * x1 + (1 - lam) * x2
    mixed_y = lam * y1 + (1 - lam) * y2
    return mixed_x, mixed_y

images, labels = next(iter(train_ds))
mixed_x, mixed_y = mixup_for_visualization(images, tf.one_hot(labels, len(class_names)), lam=0.5)

plt.figure(figsize=(10,10))
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(tf.cast(mixed_x[i]/255.0, tf.float32))  # normalize for display
    label_probs = mixed_y[i].numpy()
    top2 = label_probs.argsort()[-2:][::-1]
    label_str = f"{label_probs[top2[0]]:.2f} * {class_names[top2[0]]} + {label_probs[top2[1]]:.2f} * {class_names[top2[1]]}"
    plt.title(label_str, fontsize=10)
    plt.axis("off")

plt.suptitle("Mixup Examples on Flower Dataset", fontsize=16)
plt.savefig("mixup_flowers.png", dpi=300, bbox_inches="tight")
plt.show()