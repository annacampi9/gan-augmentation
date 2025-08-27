import os
import time
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras

from dada_config import cfg
from generator import DADAGenerator
from discriminator import DADADiscriminator, TransferDisc
from losses import logits_to_views, compute_losses


def set_seed(seed: int):
    tf.random.set_seed(seed)
    np.random.seed(seed)


def normalize_images_uint8_to_tanh(x: np.ndarray) -> np.ndarray:
    # Original: (-127.5 + data)/128 expecting CHW or HWC; here we use HWC uint8
    return ((x.astype(np.float32) - 127.5) / 128.0).astype(np.float32)


def denormalize_tanh_to_uint8(x: np.ndarray) -> np.ndarray:
    # Inverse of above for saving
    x = np.clip(x, -1.0, 1.0)
    x = (x * 128.0 + 127.5)
    return np.clip(x, 0, 255).astype(np.uint8)


def load_dataset(dataset: str):
    if dataset.lower() == "cifar10":
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        y_train = y_train.flatten().astype(np.int32)
        y_test = y_test.flatten().astype(np.int32)
        x_train = normalize_images_uint8_to_tanh(x_train)
        x_test = normalize_images_uint8_to_tanh(x_test)
        return (x_train, y_train), (x_test, y_test)
    elif dataset.lower() == "svhn":
        # Optional: requires scipy. Keep parity with original if available.
        try:
            from scipy.io import loadmat  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("SVHN requires scipy installed.") from e
        train_mat = loadmat(os.path.join(cfg.data_dir, "svhn", "train_32x32.mat"))
        test_mat = loadmat(os.path.join(cfg.data_dir, "svhn", "test_32x32.mat"))
        x_train = train_mat["X"]  # HxWxCxN
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


def subsample_per_class(x: np.ndarray, y: np.ndarray, num_classes: int, samples_per_class: int, seed: int):
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


def build_datasets(x_train, y_train, x_test, y_test, batch_size: int, augment: bool):
    autotune = tf.data.AUTOTUNE

    def ds_from(x, y, training: bool):
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if training:
            ds = ds.shuffle(buffer_size=10_000, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size, drop_remainder=True)
        if training and augment:
            aug_layers = keras.Sequential([
                keras.layers.RandomFlip("horizontal") if cfg.aug_horizontal_flip else keras.layers.Layer(),
                keras.layers.RandomTranslation(cfg.aug_height_shift, cfg.aug_width_shift),
                keras.layers.RandomRotation(cfg.aug_rotation / 360.0),
            ])

            def map_aug(img, lab):
                return aug_layers(img, training=True), lab

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
                grid[i * h:(i + 1) * h, j * w:(j + 1) * w] = images[idx]
    png = tf.io.encode_png(grid)
    tf.io.write_file(out_path, png)


@tf.function
def train_step(gen: DADAGenerator,
               disc: DADADiscriminator,
               opt_g: keras.optimizers.Optimizer,
               opt_d: keras.optimizers.Optimizer,
               x_real: tf.Tensor,
               y_real: tf.Tensor,
               y_gen: tf.Tensor,
               w: tf.Tensor,
               num_classes: int,
               feat_match_weight: float):
    z = tf.random.normal((tf.shape(x_real)[0], gen.noise_dim), mean=0.0, stddev=1.0)

    with tf.GradientTape() as tape_d:
        logits_real, feat_real = disc(x_real, training=True, return_features=True)
        x_fake = gen(z, y_gen, training=True)
        logits_fake, feat_fake = disc(x_fake, training=True, return_features=True)
        losses = compute_losses(logits_real, logits_fake, y_real, y_gen, feat_real, feat_fake, num_classes, w, feat_match_weight)
        loss_d = losses["loss_disc"]
    grads_d = tape_d.gradient(loss_d, disc.trainable_variables)
    opt_d.apply_gradients(zip(grads_d, disc.trainable_variables))

    # Generator update
    z2 = tf.random.normal((tf.shape(x_real)[0], gen.noise_dim), mean=0.0, stddev=1.0)
    with tf.GradientTape() as tape_g:
        x_fake2 = gen(z2, y_gen, training=True)
        logits_fake2, feat_fake2 = disc(x_fake2, training=True, return_features=True)
        # For generator loss, feature_real should be from current real features
        loss_g = compute_losses(logits_real, logits_fake2, y_real, y_gen, feat_real, feat_fake2, num_classes, w, feat_match_weight)["loss_gen"]
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
        loss_g = compute_losses(logits_real, logits_fake, y_real, y_gen,
                                feat_real, feat_fake, num_classes, w, feat_match_weight)["loss_gen"]
    grads_g = tape_g.gradient(loss_g, gen.trainable_variables)
    opt_g.apply_gradients(zip(grads_g, gen.trainable_variables))


def classification_error(disc: DADADiscriminator, ds: tf.data.Dataset, num_classes: int) -> float:
    total = 0
    incorrect = 0
    for x, y in ds:
        logits = disc(x, training=False)
        _, class_logits = logits_to_views(logits, num_classes)
        preds = tf.argmax(class_logits, axis=1)
        total += x.shape[0]
        incorrect += int(tf.reduce_sum(tf.cast(tf.not_equal(preds, tf.cast(y, preds.dtype)), tf.int32)))
    return incorrect / float(total)

def main():
    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(cfg.logs_dir, exist_ok=True)
    run_dir = os.path.join(cfg.results_dir, cfg.dataset)
    os.makedirs(run_dir, exist_ok=True)

    set_seed(cfg.seed)

    if cfg.mixed_precision:
        try:
            from tensorflow.keras import mixed_precision as mp
            mp.set_global_policy('mixed_float16')
        except ImportError:
            pass
    if cfg.enable_xla:
        tf.config.optimizer.set_jit(True)

    (x_train, y_train), (x_test, y_test) = load_dataset(cfg.dataset)
    if cfg.test_mode:
        spc = cfg.test_samples_per_class
    else:
        spc = cfg.samples_per_class
    x_train, y_train = subsample_per_class(x_train, y_train, cfg.num_classes, spc, cfg.seed)

    # Build models
    gen = DADAGenerator(num_classes=cfg.num_classes, noise_dim=cfg.noise_dim, weight_norm_all=False)
    if cfg.use_transfer_disc:
        disc = TransferDisc(num_classes=cfg.num_classes, backbone_name=cfg.transfer_backbone)
    else:
        disc = DADADiscriminator(num_classes=cfg.num_classes, use_weight_norm=True)

    # Optimizers
    opt_g = keras.optimizers.Adam(learning_rate=cfg.learning_rate * 0.5, beta_1=cfg.beta1, beta_2=cfg.beta2)
    opt_d = keras.optimizers.Adam(learning_rate=cfg.learning_rate, beta_1=cfg.beta1, beta_2=cfg.beta2)

    # Data-dependent initialization: run a forward pass on a small real batch
    def data_dependent_init(gen_model, disc_model, x_init, y_init):
        bs = tf.minimum(tf.shape(x_init)[0], tf.constant(500))
        x_small = x_init[:bs]
        y_small = tf.cast(y_init[:bs], tf.int32)
        # One forward through D with real
        _ = disc_model(x_small, training=True)
        # One forward through G and D with fake
        z_small = tf.random.normal((bs, gen_model.noise_dim), mean=0.0, stddev=1.0)
        x_fake_small = gen_model(z_small, y_small, training=True)
        _ = disc_model(x_fake_small, training=True)


    # Variables and ckpt
    w = tf.Variable(cfg.warmup_w, dtype=tf.float32, trainable=False)
    ckpt = tf.train.Checkpoint(gen=gen, disc=disc, opt_g=opt_g, opt_d=opt_d, w=w)
    manager = tf.train.CheckpointManager(ckpt, directory=os.path.join(run_dir, "ckpts"), max_to_keep=5)

    # Datasets; augmentation toggled after gan_epochs
    batch_size = cfg.test_batch_size if cfg.test_mode else cfg.batch_size
    train_ds_noaug, test_ds = build_datasets(x_train, y_train, x_test, y_test, batch_size, augment=False)
    train_ds_aug, _ = build_datasets(x_train, y_train, x_test, y_test, batch_size, augment=True)

    summary_writer = tf.summary.create_file_writer(os.path.join(cfg.logs_dir, cfg.dataset))

    # Freeze backbone if transfer learning is used
    if cfg.use_transfer_disc:
        if hasattr(disc, "backbone"):
            disc.backbone.trainable = False

    num_batches = math.floor(x_train.shape[0] / batch_size)
    total_epochs = cfg.test_epochs if cfg.test_mode else cfg.total_epochs
    gan_epochs = cfg.test_gan_epochs if cfg.test_mode else cfg.gan_epochs
    for epoch in range(total_epochs):
        if epoch == gan_epochs:
            w.assign(cfg.post_w)
        # Unfreeze backbone after configured freeze epochs
        if cfg.use_transfer_disc and epoch == cfg.transfer_freeze_epochs:
            if hasattr(disc, "backbone"):
                disc.backbone.trainable = True

        start = time.time()
        train_ds = train_ds_aug if (cfg.enable_aug_after_gan and epoch >= gan_epochs) else train_ds_noaug

        # Training loop
        metrics_accum = {"loss_gen": 0.0, "loss_disc": 0.0, "D_acc_on_real": 0.0, "D_acc_on_fake": 0.0, "G_acc_on_fake": 0.0}
        max_batches = cfg.test_max_batches_per_epoch if cfg.test_mode else num_batches
        for step, (x_batch, y_batch) in enumerate(train_ds):
            if step >= num_batches:
                break
            if step >= max_batches:
                break
            # Data-dependent init on first batch before training starts
            if epoch == 0 and step == 0:
                data_dependent_init(gen, disc, x_batch, y_batch)
            if cfg.g_label_mode == "random":
                y_gen = tf.random.uniform(shape=(batch_size,), minval=0, maxval=cfg.num_classes, dtype=tf.int32)
            else:
                y_gen = tf.cast(y_batch, tf.int32)

            # D and G updates
            out = train_step(gen, disc, opt_g, opt_d, x_batch, tf.cast(y_batch, tf.int32), y_gen, w, cfg.num_classes, cfg.feat_match_weight)

            # Extra G updates during warmup
            if epoch < gan_epochs:
                for _ in range(cfg.g_updates_per_step_warmup - 1):
                    gen_step_only(gen, disc, opt_g, x_batch, tf.cast(y_batch, tf.int32), y_gen, w, cfg.num_classes, cfg.feat_match_weight)

            for k in metrics_accum.keys():
                metrics_accum[k] += float(out[k])

        for k in metrics_accum.keys():
            metrics_accum[k] /= float(num_batches)

        # Evaluate classification error
        train_err = classification_error(disc, train_ds_noaug, cfg.num_classes)
        test_err = classification_error(disc, test_ds, cfg.num_classes)

        # Sampling grid
        if (epoch % cfg.sample_every_epochs) == 0:
            rows = cfg.test_sample_grid_rows if cfg.test_mode else cfg.sample_grid_rows
            cols = cfg.test_sample_grid_cols if cfg.test_mode else cfg.sample_grid_cols
            classes = np.repeat(np.arange(cfg.num_classes, dtype=np.int32), cols)
            z = np.random.normal(loc=0.0, scale=1.0, size=(classes.shape[0], cfg.noise_dim)).astype(np.float32)
            imgs = gen(tf.convert_to_tensor(z), tf.convert_to_tensor(classes), training=False).numpy()
            imgs_uint8 = denormalize_tanh_to_uint8(imgs)
            save_sample_grid(imgs_uint8, rows, cols, os.path.join(run_dir, f"samples_{epoch:04d}.png"))

        # Checkpoints
        if (epoch % cfg.checkpoint_every) == 0 and not cfg.test_mode:
            manager.save(checkpoint_number=epoch)

        # Logging
        elapsed = time.time() - start
        print(
            f"Epoch {epoch:03d} | G={metrics_accum['loss_gen']:.3f} D={metrics_accum['loss_disc']:.3f} | "
            f"train err={train_err:.4f} test err={test_err:.4f} | "
            f"D(real)={metrics_accum['D_acc_on_real']:.3f} D(fake)={metrics_accum['D_acc_on_fake']:.3f} G(fake)={metrics_accum['G_acc_on_fake']:.3f} | "
            f"{elapsed:.1f}s")
        with summary_writer.as_default():
            tf.summary.scalar("loss_gen", metrics_accum["loss_gen"], step=epoch)
            tf.summary.scalar("loss_disc", metrics_accum["loss_disc"], step=epoch)
            tf.summary.scalar("train_err", train_err, step=epoch)
            tf.summary.scalar("test_err", test_err, step=epoch)
            tf.summary.scalar("D_acc_on_real", metrics_accum["D_acc_on_real"], step=epoch)
            tf.summary.scalar("D_acc_on_fake", metrics_accum["D_acc_on_fake"], step=epoch)
            tf.summary.scalar("G_acc_on_fake", metrics_accum["G_acc_on_fake"], step=epoch)


if __name__ == "__main__":
    main()
