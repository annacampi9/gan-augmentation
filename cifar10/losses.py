import tensorflow as tf


def logits_to_views(logits_2c: tf.Tensor, num_classes: int):
    """
    Reshape flat 2*C logits into [B, 2, C] and compute class logits [B, C]
    by summing across the source axis.
    """
    logits_2c = tf.cast(logits_2c, tf.float32)
    logits_2_c = tf.reshape(logits_2c, (-1, 2, num_classes))
    class_logits = tf.reduce_sum(logits_2_c, axis=1)
    return logits_2_c, class_logits


def source_logits_for_labels(logits_2_c: tf.Tensor, y: tf.Tensor, num_classes: int) -> tf.Tensor:
    """
    Select per-example [B, 2] source logits using labels y via one-hot along class axis.
    """
    if y.dtype.is_integer:
        y_one = tf.one_hot(y, depth=num_classes)
    else:
        y_one = tf.cast(y, tf.float32)
    y_one = tf.expand_dims(y_one, axis=-1)  # [B, C, 1]
    # logits_2_c: [B, 2, C] -> [B, 2, 1]
    src = tf.matmul(logits_2_c, y_one)
    src = tf.squeeze(src, axis=-1)  # [B, 2]
    return src


def feature_match(features_real: tf.Tensor, features_fake: tf.Tensor) -> tf.Tensor:
    m1 = tf.reduce_mean(tf.cast(features_real, tf.float32), axis=0)
    m2 = tf.reduce_mean(tf.cast(features_fake, tf.float32), axis=0)
    return tf.reduce_mean(tf.abs(m1 - m2))


def _sparse_ce(logits: tf.Tensor, targets: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(targets, tf.int32), logits=tf.cast(logits, tf.float32)))


def _binary_acc_from_logits(logits_b2: tf.Tensor, target_class: int) -> tf.Tensor:
    preds = tf.argmax(logits_b2, axis=1)
    correct = tf.equal(preds, tf.cast(target_class, preds.dtype))
    return tf.reduce_mean(tf.cast(correct, tf.float32))


def compute_losses(
    logits_real_2c: tf.Tensor,
    logits_fake_2c: tf.Tensor,
    y_real: tf.Tensor,
    y_fake: tf.Tensor,
    features_real: tf.Tensor,
    features_fake: tf.Tensor,
    num_classes: int,
    w: tf.Tensor,
    feat_match_weight: float = 0.5,
):
    """
    Compute DADA losses and metrics following the original Theano math.
    Returns a dict with scalar tensors.
    """
    # Views
    real_2_c, class_lab = logits_to_views(logits_real_2c, num_classes)
    fake_2_c, class_gen = logits_to_views(logits_fake_2c, num_classes)

    # Source logits per labels
    source_lab = source_logits_for_labels(real_2_c, y_real, num_classes)
    source_gen = source_logits_for_labels(fake_2_c, y_fake, num_classes)

    # Loss terms (from logits directly)
    zeros = tf.zeros_like(y_real, dtype=tf.int32)
    ones = tf.ones_like(y_fake, dtype=tf.int32)

    loss_gen_class = _sparse_ce(class_gen, y_fake)
    loss_gen_source = _sparse_ce(source_gen, zeros)
    loss_lab_class = _sparse_ce(class_lab, y_real)
    # Note: includes both real and fake components as in original
    loss_lab_source = _sparse_ce(source_lab, zeros) + _sparse_ce(source_gen, ones)

    # Feature matching
    fm = feature_match(features_real, features_fake)

    # Composite losses with schedule
    one_minus_w = 1.0 - tf.cast(w, tf.float32)
    w_f = tf.cast(w, tf.float32)
    loss_gen = one_minus_w * (loss_gen_source + feat_match_weight * fm)
    loss_disc = one_minus_w * (loss_lab_source) + w_f * (loss_lab_class + loss_gen_class)

    # Metrics
    d_acc_on_real = _binary_acc_from_logits(source_lab, 0)
    d_acc_on_fake = _binary_acc_from_logits(source_gen, 1)
    g_acc_on_fake = _binary_acc_from_logits(source_gen, 0)

    return {
        "loss_gen": loss_gen,
        "loss_disc": loss_disc,
        "feature_loss": fm,
        "loss_gen_source": loss_gen_source,
        "loss_gen_class": loss_gen_class,
        "loss_lab_class": loss_lab_class,
        "loss_lab_source": loss_lab_source,
        "D_acc_on_real": d_acc_on_real,
        "D_acc_on_fake": d_acc_on_fake,
        "G_acc_on_fake": g_acc_on_fake,
    }