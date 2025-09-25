"""Dataset loading, normalization, subsampling, tf.data pipelines, and image grids."""

from __future__ import annotations

import os

import numpy as np
import tensorflow as tf

try:
    import tensorflow.keras as keras  # type: ignore
except Exception:  # pragma: no cover
    from tensorflow import keras  # type: ignore


def normalize_images_uint8_to_tanh(x: np.ndarray) -> np.ndarray:
    return ((x.astype(np.float32) - 127.5) / 128.0).astype(np.float32)


def denormalize_tanh_to_uint8(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -1.0, 1.0)
    x = x * 128.0 + 127.5
    return np.clip(x, 0, 255).astype(np.uint8)


def load_dataset(dataset: str, data_dir: str):
    if dataset.lower() == "cifar10":
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        y_train = y_train.flatten().astype(np.int32)
        y_test = y_test.flatten().astype(np.int32)
        x_train = normalize_images_uint8_to_tanh(x_train)
        x_test = normalize_images_uint8_to_tanh(x_test)
        return (x_train, y_train), (x_test, y_test)
    elif dataset.lower() == "svhn":
        from scipy.io import loadmat  # type: ignore

        train_mat = loadmat(os.path.join(data_dir, "svhn", "train_32x32.mat"))
        test_mat = loadmat(os.path.join(data_dir, "svhn", "test_32x32.mat"))
        x_train = train_mat["X"]
        y_train = train_mat["y"].flatten()
        x_test = test_mat["X"]
        y_test = test_mat["y"].flatten()
        y_train[y_train == 10] = 0
        y_test[y_test == 10] = 0
        x_train = np.transpose(x_train, (3, 0, 1, 2))
        x_test = np.transpose(x_test, (3, 0, 1, 2))
        x_train = normalize_images_uint8_to_tanh(x_train)
        x_test = normalize_images_uint8_to_tanh(x_test)
        return (x_train, y_train.astype(np.int32)), (x_test, y_test.astype(np.int32))
    else:
        raise ValueError("Unsupported dataset: " + dataset)


def subsample_per_class(
    x: np.ndarray, y: np.ndarray, num_classes: int, samples_per_class: int, seed: int
):
    rng = np.random.default_rng(seed)
    selected_x = []
    selected_y = []
    for c in range(num_classes):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        idx = idx[:samples_per_class]
        selected_x.append(x[idx])
        selected_y.append(y[idx])
    x_sel = np.concatenate(selected_x, axis=0)
    y_sel = np.concatenate(selected_y, axis=0)
    perm = rng.permutation(x_sel.shape[0])
    return x_sel[perm], y_sel[perm]


def build_datasets(x_train, y_train, x_test, y_test, batch_size: int, augment: bool, cfg):
    autotune = tf.data.AUTOTUNE

    def ds_from(x, y, training: bool):
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if training:
            ds = ds.shuffle(buffer_size=10_000, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size, drop_remainder=True)
        if training and augment:

            def map_aug(img, lab):
                x = img
                if cfg.use_pad_and_crop:
                    pad = cfg.aug_pad
                    x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode="REFLECT")
                    x = tf.image.random_crop(x, size=tf.shape(img))
                if cfg.aug_horizontal_flip:
                    x = tf.image.random_flip_left_right(x)
                if (
                    cfg.aug_width_shift > 0.0
                    or cfg.aug_height_shift > 0.0
                    or cfg.aug_rotation > 0.0
                ):
                    aug_layers = keras.Sequential(
                        [
                            keras.layers.RandomTranslation(
                                cfg.aug_height_shift, cfg.aug_width_shift
                            ),
                            keras.layers.RandomRotation(cfg.aug_rotation / 360.0),
                        ]
                    )
                    x = aug_layers(x, training=True)
                return x, lab

            ds = ds.map(map_aug, num_parallel_calls=autotune)
        ds = ds.prefetch(autotune)
        return ds

    return ds_from(x_train, y_train, True), ds_from(x_test, y_test, False)


def save_sample_grid(images: np.ndarray, rows: int, cols: int, out_path: str):
    b, h, w, c = images.shape
    grid = np.zeros((rows * h, cols * w, c), dtype=images.dtype)
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < b:
                grid[i * h : (i + 1) * h, j * w : (j + 1) * w] = images[idx]
    png = tf.io.encode_png(grid)
    tf.io.write_file(out_path, png)
