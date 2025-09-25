"""Accuracy/F1 metrics and confusion matrix reporting utilities."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def classification_metrics(disc, ds: tf.data.Dataset, num_classes: int):
    y_true, y_pred = [], []
    for x, y in ds:
        logits = disc(x, training=False)
        logits_2_c = tf.reshape(tf.cast(logits, tf.float32), (-1, 2, num_classes))
        class_logits = tf.reduce_sum(logits_2_c, axis=1)
        preds = tf.argmax(class_logits, axis=1).numpy()
        y_true.extend(y.numpy())
        y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = (y_true == y_pred).mean()
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=3, output_dict=True)
    return acc, macro_f1, cm, report
