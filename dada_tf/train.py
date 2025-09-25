"""Training orchestrator for DADA + optional hybrid augmentations.

High-level flow: config -> data -> models -> train loop -> eval -> save.
Behavior preserved from the original script.
"""

from __future__ import annotations

import math
import os
import time

import numpy as np
import tensorflow as tf

try:
    import tensorflow.keras as keras  # type: ignore
except Exception:  # pragma: no cover
    from tensorflow import keras  # type: ignore

from .augmentations import maybe_apply_hybrid_aug
from .config import cfg
from .discriminator import DADADiscriminator, TransferDisc
from .generator import DADAGenerator
from .losses import compute_losses
from .utils.data import (
    build_datasets,
    denormalize_tanh_to_uint8,
    load_dataset,
    save_sample_grid,
    subsample_per_class,
)
from .utils.logging import create_summary_writer
from .utils.metrics import classification_metrics
from .utils.schedules import lr_from_schedule


class EMAHelper:
    """Simple TF2 EMA for model variables (robust)."""

    def __init__(self, decay=0.999):
        self.decay = decay
        self.shadow = {}

    def update(self, variables):
        for v in variables:
            key = id(v)
            if key not in self.shadow or self.shadow[key].shape != v.shape:
                self.shadow[key] = tf.Variable(v, trainable=False)
            else:
                self.shadow[key].assign(self.decay * self.shadow[key] + (1.0 - self.decay) * v)

    def apply(self, variables, fn):
        originals = [tf.identity(v) for v in variables]
        for v in variables:
            key = id(v)
            if key in self.shadow:
                v.assign(self.shadow[key])
        try:
            return fn()
        finally:
            for v, val in zip(variables, originals):
                v.assign(val)


def set_seed(seed: int):
    tf.random.set_seed(seed)
    np.random.seed(seed)


@tf.function
def train_step(
    gen: DADAGenerator,
    disc: DADADiscriminator,
    opt_g: keras.optimizers.Optimizer,
    opt_d: keras.optimizers.Optimizer,
    x_real: tf.Tensor,
    y_real: tf.Tensor,
    y_gen: tf.Tensor,
    w: tf.Tensor,
    num_classes: int,
    feat_match_weight: float,
    apply_hybrid: tf.Tensor,
    use_mixup: tf.Tensor,
    use_cutmix: tf.Tensor,
    alpha: tf.Tensor,
):
    z = tf.random.normal((tf.shape(x_real)[0], gen.noise_dim), mean=0.0, stddev=1.0)

    with tf.GradientTape() as tape_d:

        def _apply_aug():
            xr, yr = maybe_apply_hybrid_aug(
                x_real,
                y_real,
                num_classes,
                tf.cast(use_mixup, tf.bool),
                tf.cast(use_cutmix, tf.bool),
                tf.cast(alpha, tf.float32),
            )
            return xr, yr

        def _no_aug():
            return x_real, tf.cast(
                tf.one_hot(tf.cast(y_real, tf.int32), depth=num_classes), tf.float32
            )

        x_real_in, y_real_in = tf.cond(tf.cast(apply_hybrid, tf.bool), _apply_aug, _no_aug)

        logits_real, feat_real = disc(x_real_in, training=True, return_features=True)
        x_fake = gen(z, y_gen, training=True)
        x_fake_in = x_fake
        y_fake_in = tf.cast(tf.one_hot(tf.cast(y_gen, tf.int32), depth=num_classes), tf.float32)
        logits_fake, feat_fake = disc(x_fake_in, training=True, return_features=True)
        losses = compute_losses(
            logits_real,
            logits_fake,
            y_real_in,
            y_fake_in,
            feat_real,
            feat_fake,
            num_classes,
            w,
            feat_match_weight,
        )
        loss_d = losses["loss_disc"]
    grads_d = tape_d.gradient(loss_d, disc.trainable_variables)
    opt_d.apply_gradients(zip(grads_d, disc.trainable_variables))

    z2 = tf.random.normal((tf.shape(x_real)[0], gen.noise_dim), mean=0.0, stddev=1.0)
    with tf.GradientTape() as tape_g:
        x_fake2 = gen(z2, y_gen, training=True)
        x_fake2_in = x_fake2
        y_fake2_in = tf.cast(tf.one_hot(tf.cast(y_gen, tf.int32), depth=num_classes), tf.float32)
        logits_fake2, feat_fake2 = disc(x_fake2_in, training=True, return_features=True)
        loss_g = compute_losses(
            logits_real,
            logits_fake2,
            y_real_in,
            y_fake2_in,
            feat_real,
            feat_fake2,
            num_classes,
            w,
            feat_match_weight,
        )["loss_gen"]
    grads_g = tape_g.gradient(loss_g, gen.trainable_variables)
    opt_g.apply_gradients(zip(grads_g, gen.trainable_variables))

    return losses


@tf.function
def gen_step_only(gen, disc, opt_g, x_real, y_real, y_gen, w, num_classes, feat_match_weight):
    with tf.GradientTape() as tape_g:
        z = tf.random.normal((tf.shape(x_real)[0], gen.noise_dim), mean=0.0, stddev=1.0)
        x_fake = gen(z, y_gen, training=True)
        logits_real, feat_real = disc(x_real, training=True, return_features=True)
        logits_fake, feat_fake = disc(x_fake, training=True, return_features=True)
        loss_g = compute_losses(
            logits_real,
            logits_fake,
            y_real,
            y_gen,
            feat_real,
            feat_fake,
            num_classes,
            w,
            feat_match_weight,
        )["loss_gen"]
    grads_g = tape_g.gradient(loss_g, gen.trainable_variables)
    opt_g.apply_gradients(zip(grads_g, gen.trainable_variables))


def main():
    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(cfg.logs_dir, exist_ok=True)
    mode_parts = []
    if cfg.use_dada:
        mode_parts.append("DADA")
    if cfg.use_mixup:
        mode_parts.append("Mixup")
    if cfg.use_cutmix:
        mode_parts.append("CutMix")
    base_mode_label = "+".join(mode_parts) if mode_parts else "Baseline"
    run_dir = os.path.join(cfg.results_dir, f"{cfg.dataset}_{base_mode_label}")
    os.makedirs(run_dir, exist_ok=True)

    set_seed(cfg.seed)

    if cfg.mixed_precision:
        try:
            from tensorflow.keras import mixed_precision as mp

            mp.set_global_policy("mixed_float16")
        except ImportError:
            pass
    if cfg.enable_xla:
        tf.config.optimizer.set_jit(True)
    else:
        try:
            tf.config.optimizer.set_experimental_options({"layout_optimizer": False})
        except Exception:
            pass

    (x_train, y_train), (x_test, y_test) = load_dataset(cfg.dataset, cfg.data_dir)
    spc = cfg.test_samples_per_class if cfg.test_mode else cfg.samples_per_class
    x_train, y_train = subsample_per_class(x_train, y_train, cfg.num_classes, spc, cfg.seed)

    gen = DADAGenerator(num_classes=cfg.num_classes, noise_dim=cfg.noise_dim, weight_norm_all=False)
    if cfg.use_transfer_disc:
        disc = TransferDisc(num_classes=cfg.num_classes, backbone_name=cfg.transfer_backbone)
    else:
        disc = DADADiscriminator(num_classes=cfg.num_classes, use_weight_norm=True)

    lr_g_base = float(
        cfg.g_learning_rate if cfg.g_learning_rate is not None else (cfg.learning_rate * 0.5)
    )
    lr_d_base = float(cfg.d_learning_rate if cfg.d_learning_rate is not None else cfg.learning_rate)
    opt_g = keras.optimizers.Adam(learning_rate=lr_g_base, beta_1=cfg.beta1, beta_2=cfg.beta2)
    opt_d = keras.optimizers.Adam(learning_rate=lr_d_base, beta_1=cfg.beta1, beta_2=cfg.beta2)

    def data_dependent_init(gen_model, disc_model, x_init, y_init):
        bs = tf.minimum(tf.shape(x_init)[0], tf.constant(500))
        x_small = x_init[:bs]
        y_small = tf.cast(y_init[:bs], tf.int32)
        _ = disc_model(x_small, training=True)
        z_small = tf.random.normal((bs, gen_model.noise_dim), mean=0.0, stddev=1.0)
        x_fake_small = gen_model(z_small, y_small, training=True)
        _ = disc_model(x_fake_small, training=True)

    w = tf.Variable(cfg.warmup_w, dtype=tf.float32, trainable=False)
    ckpt = tf.train.Checkpoint(gen=gen, disc=disc, opt_g=opt_g, opt_d=opt_d, w=w)
    manager = tf.train.CheckpointManager(
        ckpt, directory=os.path.join(run_dir, "ckpts"), max_to_keep=5
    )

    batch_size = cfg.test_batch_size if cfg.test_mode else cfg.batch_size
    train_ds_noaug, test_ds = build_datasets(
        x_train, y_train, x_test, y_test, batch_size, augment=False, cfg=cfg
    )
    train_ds_aug, _ = build_datasets(
        x_train, y_train, x_test, y_test, batch_size, augment=True, cfg=cfg
    )

    # Create a writer (kept implicit to avoid unused warning if not used elsewhere)
    _ = create_summary_writer(cfg.logs_dir, f"{cfg.dataset}_{base_mode_label}")

    if cfg.use_transfer_disc and hasattr(disc, "backbone"):
        disc.backbone.trainable = False

    num_batches = math.floor(x_train.shape[0] / batch_size)
    total_epochs = cfg.test_epochs if cfg.test_mode else cfg.total_epochs
    gan_epochs = cfg.test_gan_epochs if cfg.test_mode else cfg.gan_epochs

    ema = None
    if cfg.use_ema:
        ema = EMAHelper(decay=cfg.ema_decay)

    for epoch in range(total_epochs):
        lr_d_epoch = lr_from_schedule(
            epoch, lr_d_base, cfg.lr_step_epochs, cfg.lr_values_d, cfg.lr_schedule_type
        )
        lr_g_epoch = lr_from_schedule(
            epoch, lr_g_base, cfg.lr_step_epochs, cfg.lr_values_g, cfg.lr_schedule_type
        )
        opt_d.learning_rate = lr_d_epoch
        opt_g.learning_rate = lr_g_epoch
        if epoch == gan_epochs:
            w.assign(cfg.post_w)
        if (
            cfg.use_transfer_disc
            and epoch == cfg.transfer_freeze_epochs
            and hasattr(disc, "backbone")
        ):
            disc.backbone.trainable = True

        start = time.time()
        train_ds = (
            train_ds_aug if (cfg.enable_aug_after_gan and epoch >= gan_epochs) else train_ds_noaug
        )

        metrics_accum = {
            "loss_gen": 0.0,
            "loss_disc": 0.0,
            "D_acc_on_real": 0.0,
            "D_acc_on_fake": 0.0,
            "G_acc_on_fake": 0.0,
        }
        max_batches = cfg.test_max_batches_per_epoch if cfg.test_mode else num_batches
        for step, (x_batch, y_batch) in enumerate(train_ds):
            if step >= num_batches:
                break
            if step >= max_batches:
                break
            if epoch == 0 and step == 0:
                data_dependent_init(gen, disc, x_batch, y_batch)
            if cfg.g_label_mode == "random":
                y_gen = tf.random.uniform(
                    shape=(batch_size,), minval=0, maxval=cfg.num_classes, dtype=tf.int32
                )
            else:
                y_gen = tf.cast(y_batch, tf.int32)

            hybrid_on = (
                (cfg.use_mixup or cfg.use_cutmix)
                and (epoch >= gan_epochs)
                and (epoch >= cfg.hybrid_start_epoch)
                and (epoch <= cfg.hybrid_end_epoch)
            )
            apply_hybrid = tf.constant(hybrid_on)
            out = train_step(
                gen,
                disc,
                opt_g,
                opt_d,
                x_batch,
                tf.cast(y_batch, tf.int32),
                y_gen,
                w,
                cfg.num_classes,
                cfg.feat_match_weight,
                apply_hybrid,
                tf.constant(cfg.use_mixup),
                tf.constant(cfg.use_cutmix),
                tf.constant(cfg.alpha),
            )

            if epoch < gan_epochs:
                for _ in range(cfg.g_updates_per_step_warmup - 1):
                    gen_step_only(
                        gen,
                        disc,
                        opt_g,
                        x_batch,
                        tf.cast(y_batch, tf.int32),
                        y_gen,
                        w,
                        cfg.num_classes,
                        cfg.feat_match_weight,
                    )

            if ema is not None:
                ema.update(disc.trainable_variables)

            for k in metrics_accum.keys():
                metrics_accum[k] += float(out[k])

        for k in metrics_accum.keys():
            metrics_accum[k] /= float(num_batches)

        if ema is not None:
            train_acc, train_f1, train_cm, train_report = ema.apply(
                disc.trainable_variables,
                lambda: classification_metrics(disc, train_ds_noaug, cfg.num_classes),
            )
            test_acc, test_f1, test_cm, test_report = ema.apply(
                disc.trainable_variables,
                lambda: classification_metrics(disc, test_ds, cfg.num_classes),
            )
        else:
            train_acc, train_f1, train_cm, train_report = classification_metrics(
                disc, train_ds_noaug, cfg.num_classes
            )
            test_acc, test_f1, test_cm, test_report = classification_metrics(
                disc, test_ds, cfg.num_classes
            )

        if (epoch % cfg.sample_every_epochs) == 0:
            rows = cfg.test_sample_grid_rows if cfg.test_mode else cfg.sample_grid_rows
            cols = cfg.test_sample_grid_cols if cfg.test_mode else cfg.sample_grid_cols
            classes = np.repeat(np.arange(cfg.num_classes, dtype=np.int32), cols)
            z = np.random.normal(loc=0.0, scale=1.0, size=(classes.shape[0], cfg.noise_dim)).astype(
                np.float32
            )
            imgs = gen(
                tf.convert_to_tensor(z), tf.convert_to_tensor(classes), training=False
            ).numpy()
            imgs_uint8 = denormalize_tanh_to_uint8(imgs)
            save_sample_grid(
                imgs_uint8, rows, cols, os.path.join(run_dir, f"samples_{epoch:04d}.png")
            )

        if (epoch % cfg.checkpoint_every) == 0 and not cfg.test_mode:
            manager.save(checkpoint_number=epoch)

        elapsed = time.time() - start
        if (cfg.use_mixup or cfg.use_cutmix) and epoch < gan_epochs and cfg.use_dada:
            phase_label = "DADA (warmup)"
        else:
            phase_label = base_mode_label
        try:
            lr_d_log = float(keras.backend.get_value(opt_d.learning_rate))
        except Exception:
            lr_d_log = (
                float(opt_d.learning_rate.numpy())
                if hasattr(opt_d.learning_rate, "numpy")
                else float(opt_d.learning_rate)
            )
        try:
            lr_g_log = float(keras.backend.get_value(opt_g.learning_rate))
        except Exception:
            lr_g_log = (
                float(opt_g.learning_rate.numpy())
                if hasattr(opt_g.learning_rate, "numpy")
                else float(opt_g.learning_rate)
            )

        print(
            f"[{phase_label}] Epoch {epoch:03d} | "
            f"G={metrics_accum['loss_gen']:.3f} D={metrics_accum['loss_disc']:.3f} | "
            f"train acc={train_acc:.4f} test acc={test_acc:.4f} | "
            f"train F1={train_f1:.4f} test F1={test_f1:.4f} | "
            f"lr_d={lr_d_log:.1e} lr_g={lr_g_log:.1e} | "
            f"{elapsed:.1f}s"
        )

        if epoch == total_epochs - 1:
            print("\nFinal Confusion Matrix (Test):")
            print(test_cm)
            print("\nFinal Per-class F1 (Test):")
            for cls, vals in test_report.items():
                if cls.isdigit():
                    print(
                        f"Class {cls}: F1={vals['f1-score']:.3f}, "
                        f"Precision={vals['precision']:.3f}, "
                        f"Recall={vals['recall']:.3f}"
                    )


if __name__ == "__main__":
    main()
