"""TensorBoard/file logging helpers (thin wrappers)."""

from __future__ import annotations

import os

import tensorflow as tf


def create_summary_writer(logs_dir: str, run_name: str):
    os.makedirs(logs_dir, exist_ok=True)
    return tf.summary.create_file_writer(os.path.join(logs_dir, run_name))


def log_scalars(writer, step: int, scalars: dict[str, float]):
    with writer.as_default():
        for k, v in scalars.items():
            tf.summary.scalar(k, v, step=step)
