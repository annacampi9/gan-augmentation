"""Hybrid batch augmentations (Mixup, CutMix).

All functions preserve the original behavior. Shapes use HWC ordering.
"""

from __future__ import annotations

import tensorflow as tf

from .utils.tfops import compute_cutmix_bbox_params, one_hot_maybe, sample_beta


def mixup_batch(
    x: tf.Tensor, y: tf.Tensor, num_classes: int, alpha: float
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Apply Mixup to a batch.

    Parameters
    ----------
    x : tf.Tensor
        Float tensor of shape [B, H, W, C] in model input range.
    y : tf.Tensor
        Int labels [B] or one-hot labels [B, C].
    num_classes : int
        Number of classes.
    alpha : float
        Beta(alpha, alpha) parameter (>0).

    Returns
    -------
    x_mix : tf.Tensor
        Mixed images, shape [B, H, W, C], dtype float32.
    y_mix : tf.Tensor
        Mixed soft labels, shape [B, C], dtype float32.

    Notes
    -----
    - Uses a single per-example beta coefficient with pairing by shuffle.
    - If `y` is integer, it is one-hot encoded internally.
    """
    batch_size = tf.shape(x)[0]
    beta = sample_beta((batch_size, 1, 1, 1), alpha)
    indices = tf.random.shuffle(tf.range(batch_size))
    x2 = tf.gather(x, indices)
    y2 = tf.gather(y, indices)
    y_one = one_hot_maybe(y, num_classes)
    y2_one = one_hot_maybe(y2, num_classes)
    x_mix = beta * tf.cast(x, tf.float32) + (1.0 - beta) * tf.cast(x2, tf.float32)
    beta_y = tf.squeeze(beta, axis=[1, 2, 3])
    y_mix = tf.expand_dims(beta_y, -1) * y_one + (1.0 - tf.expand_dims(beta_y, -1)) * y2_one
    return x_mix, tf.cast(y_mix, tf.float32)


def cutmix_batch(
    x: tf.Tensor, y: tf.Tensor, num_classes: int, alpha: float
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Apply CutMix to a batch.

    Parameters
    ----------
    x : tf.Tensor
        Float tensor of shape [B, H, W, C].
    y : tf.Tensor
        Int labels [B] or one-hot labels [B, C].
    num_classes : int
        Number of classes.
    alpha : float
        Beta(alpha, alpha) parameter (>0).

    Returns
    -------
    x_mix : tf.Tensor
        Mixed images, shape [B, H, W, C], dtype float32.
    y_mix : tf.Tensor
        Mixed soft labels, shape [B, C], dtype float32.
    """
    x = tf.cast(x, tf.float32)
    batch_size = tf.shape(x)[0]
    height = tf.shape(x)[1]
    width = tf.shape(x)[2]

    lam = sample_beta((batch_size, 1), alpha)
    indices = tf.random.shuffle(tf.range(batch_size))
    x2 = tf.gather(x, indices)
    y2 = tf.gather(y, indices)
    y_one = one_hot_maybe(y, num_classes)
    y2_one = one_hot_maybe(y2, num_classes)

    y1, y2b, x1, x2b = compute_cutmix_bbox_params(height, width, lam)
    ys = tf.reshape(tf.range(height, dtype=tf.int32), (1, -1, 1))
    xs = tf.reshape(tf.range(width, dtype=tf.int32), (1, 1, -1))
    y1e = tf.reshape(y1, (-1, 1, 1))
    y2e = tf.reshape(y2b, (-1, 1, 1))
    x1e = tf.reshape(x1, (-1, 1, 1))
    x2e = tf.reshape(x2b, (-1, 1, 1))
    y_mask = tf.logical_and(ys >= y1e, ys < y2e)
    x_mask = tf.logical_and(xs >= x1e, xs < x2e)
    rect = tf.cast(tf.logical_and(y_mask, x_mask), tf.float32)
    rect = tf.expand_dims(rect, axis=-1)

    x_mix = x * (1.0 - rect) + x2 * rect
    area = tf.cast((y2b - y1) * (x2b - x1), tf.float32)
    lam_eff = 1.0 - (area / tf.cast(height * width, tf.float32))
    y_mix = lam_eff * y_one + (1.0 - lam_eff) * y2_one
    return x_mix, tf.cast(y_mix, tf.float32)


def maybe_apply_hybrid_aug(
    x: tf.Tensor,
    y: tf.Tensor,
    num_classes: int,
    use_mixup: bool,
    use_cutmix: bool,
    alpha: float,
):
    """
    Conditionally apply Mixup or CutMix to the batch.

    Returns a tuple of augmented images and soft labels. If neither augmentation
    is enabled, returns original images and one-hot labels.
    """
    if use_mixup and use_cutmix:
        if tf.random.uniform(()) < 0.5:
            return mixup_batch(x, y, num_classes, alpha)
        else:
            return cutmix_batch(x, y, num_classes, alpha)
    elif use_mixup:
        return mixup_batch(x, y, num_classes, alpha)
    elif use_cutmix:
        return cutmix_batch(x, y, num_classes, alpha)
    else:
        return x, one_hot_maybe(y, num_classes)
