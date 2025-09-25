"""Small TensorFlow tensor helpers shared across modules.

Functions here are pure helpers (no side effects) and are reused by
augmentations and models. Behavior preserved from the original code.
"""

from __future__ import annotations

import tensorflow as tf


def one_hot_maybe(y: tf.Tensor, num_classes: int) -> tf.Tensor:
    """
    Convert labels to one-hot if needed.

    Parameters
    ----------
    y : tf.Tensor
        Tensor of shape [B] (int) or [B, C] (float).
    num_classes : int
        Number of classes C.

    Returns
    -------
    tf.Tensor
        One-hot labels of shape [B, C], dtype float32.
    """
    if y.dtype.is_integer:
        oh = tf.one_hot(tf.cast(y, tf.int32), depth=num_classes)
    else:
        oh = tf.cast(y, tf.float32)
    return tf.cast(oh, tf.float32)


def sample_beta(shape: tuple[int, ...] | tf.TensorShape, alpha: float) -> tf.Tensor:
    """
    Sample from Beta(alpha, alpha) in a numerically stable way.

    Parameters
    ----------
    shape : tuple[int, ...] | tf.TensorShape
        Output shape.
    alpha : float
        Beta parameter (>0).

    Returns
    -------
    tf.Tensor
        Tensor of shape `shape`, dtype float32.
    """
    alpha_t = tf.convert_to_tensor(alpha, dtype=tf.float32)
    alpha_t = tf.maximum(alpha_t, tf.constant(1e-5, dtype=tf.float32))
    x = tf.random.gamma(shape, alpha_t, dtype=tf.float32)
    y = tf.random.gamma(shape, alpha_t, dtype=tf.float32)
    return x / (x + y)


def compute_cutmix_bbox_params(height_t: tf.Tensor, width_t: tf.Tensor, lam_b: tf.Tensor):
    """
    Vectorized CutMix bounding box parameters per example.

    Parameters
    ----------
    height_t : tf.Tensor
        Scalar/int32 tensor H or shape [B, 1].
    width_t : tf.Tensor
        Scalar/int32 tensor W or shape [B, 1].
    lam_b : tf.Tensor
        Lambda values in [0,1], shape [B, 1].

    Returns
    -------
    tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]
        y1, y2, x1, x2 as int32 tensors shaped [B, 1].
    """
    r = tf.sqrt(1.0 - lam_b)  # [B, 1]
    cut_h = tf.cast(r * tf.cast(height_t, tf.float32), tf.int32)
    cut_w = tf.cast(r * tf.cast(width_t, tf.float32), tf.int32)
    cy = tf.random.uniform(tf.shape(cut_h), minval=0, maxval=height_t, dtype=tf.int32)
    cx = tf.random.uniform(tf.shape(cut_w), minval=0, maxval=width_t, dtype=tf.int32)
    y1 = tf.clip_by_value(cy - cut_h // 2, 0, height_t)
    y2 = tf.clip_by_value(cy + cut_h // 2, 0, height_t)
    x1 = tf.clip_by_value(cx - cut_w // 2, 0, width_t)
    x2 = tf.clip_by_value(cx + cut_w // 2, 0, width_t)
    return y1, y2, x1, x2
