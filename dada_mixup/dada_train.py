# train_dada.py
"""
DADA Trainer (faithful, minimal)

- Wires G, D, and DADA losses correctly
- Two-phase schedule: w=0 for epochs < gan_epochs, else w=1
- Train-only CIFAR-10 data, normalized to [-1, 1]
- WeightNorm data-init warmup for D
- Saves consistent sample grids (PNG)
- Optional light augmentation after epoch >= gan_epochs (matches DADA spirit)
"""

import os
import time
import argparse
import numpy as np
import tensorflow as tf

from dada_config import CONFIG
from dada_generator import create_generator
from dada_discriminator import create_discriminator
from dada_loss import compute_dada_components, combine_dada_losses

AUTOTUNE = tf.data.AUTOTUNE


# ----------------------------- Utils -----------------------------

def get_path(key: str, default: str) -> str:
    """Utility to read CONFIG['paths'][key] with a safe fallback."""
    try:
        path = CONFIG['paths'][key]
    except Exception:
        path = default
    os.makedirs(path, exist_ok=True)
    return path


def save_grid(images: tf.Tensor, labels: tf.Tensor, path: str, nrow: int = 10, dpi: int = 150):
    """
    Save images as a grid PNG.

    Args:
        images: tensor in [-1,1], shape (N,H,W,C)
        labels: int tensor (unused here; kept for future titles)
        path: output PNG path
        nrow: images per row (default 10)
        dpi: PNG DPI
    """
    images = tf.clip_by_value((images + 1.0) / 2.0, 0.0, 1.0)  # [-1,1] -> [0,1]
    images = images.numpy()

    n = images.shape[0]
    H, W, C = images[0].shape
    ncol = min(nrow, n)
    nrow_actual = (n + ncol - 1) // ncol  # ceil

    grid = np.zeros((H * nrow_actual, W * ncol, C), dtype=np.float32)
    for i in range(n):
        r = i // ncol
        c = i % ncol
        grid[r*H:(r+1)*H, c*W:(c+1)*W, :] = images[i]

    import matplotlib.pyplot as plt
    plt.figure(figsize=(ncol * 1.6, nrow_actual * 1.6), dpi=dpi)
    plt.imshow(grid, interpolation='nearest')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"[grid] saved -> {path}")


def create_fixed_sampler():
    """Fixed (z, y) for consistent grids across epochs."""
    K = CONFIG['dataset']['num_classes']
    latent = CONFIG['dada']['latent_dim']
    z_fixed = tf.random.normal([K * 10, latent])
    y_fixed = tf.repeat(tf.range(K, dtype=tf.int32), repeats=10)
    return z_fixed, y_fixed


def to_minus1_1(x_uint8: np.ndarray) -> np.ndarray:
    """CIFAR scaling to [-1,1] to match tanh/disc expectations."""
    x = x_uint8.astype(np.float32)
    return (x - 127.5) / 128.0


def prepare_data(batch_size: int, augment: bool = False) -> tf.data.Dataset:
    """
    Train-only CIFAR-10 dataset, normalized to [-1,1].
    Optional light augmentation (used after epoch >= gan_epochs).

    Args:
        batch_size: batch size (use 100 to match DADA)
        augment: whether to apply light aug (flip/shift/rotate)

    Returns:
        tf.data.Dataset of (x, y)
    """
    (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
    x = to_minus1_1(x_train)
    y = y_train.astype(np.int32).squeeze()

    ds = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10000)

    if augment:
        # light DADA-style augmentation
        def _aug(im, lab):
            # random horizontal flip
            im = tf.image.random_flip_left_right(im)
            # small translations (shift up to ~10% width/height)
            tx = tf.random.uniform([], -0.1, 0.1) * tf.cast(tf.shape(im)[1], tf.float32)
            ty = tf.random.uniform([], -0.1, 0.1) * tf.cast(tf.shape(im)[0], tf.float32)
            im = tfa_image_translate(im, tx, ty)
            # small rotation (~±20 degrees)
            deg = tf.random.uniform([], -20.0, 20.0)
            im = tfa_image_rotate_deg(im, deg)
            return im, lab

        # use map with tf.py_function-free ops (we define helpers below)
        ds = ds.map(_aug, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)
    return ds


def tfa_image_translate(image: tf.Tensor, tx: tf.Tensor, ty: tf.Tensor) -> tf.Tensor:
    """Pure-TF 2D translate (no TFA dependency for simple shift)."""
    # Build translation transforms: [1, 0, -tx, 0, 1, -ty, 0, 0]
    H = tf.cast(tf.shape(image)[0], tf.float32)
    W = tf.cast(tf.shape(image)[1], tf.float32)
    # normalize shifts to [-1,1] space for grid_sample-like ops
    dx = (tx * 2.0) / W
    dy = (ty * 2.0) / H
    # Use affine transform via image projective transform expects 8 params:
    # [a0, a1, a2, a3, a4, a5, a6, a7], last implicit 1.0
    # pure translation matrix:
    # [[1, 0, -tx],
    #  [0, 1, -ty],
    #  [0, 0,  1 ]]
    # TF expects a2/a5 normalized to [-1,1] *some backends differ*. To keep it robust,
    # we fallback to pad+crop shifting which is simpler and consistent.
    pad = tf.pad(image, [[8, 8], [8, 8], [0, 0]], mode='REFLECT')
    # integer shifts for simple demonstration (close enough for light aug)
    tx_i = tf.cast(tf.round(tx), tf.int32)
    ty_i = tf.cast(tf.round(ty), tf.int32)
    start_y = 8 - ty_i
    start_x = 8 - tx_i
    shifted = pad[start_y:start_y+32, start_x:start_x+32, :]
    return shifted


def tfa_image_rotate_deg(image: tf.Tensor, degrees: tf.Tensor) -> tf.Tensor:
    """Small rotation in degrees (approx via scipy-like center rotation)."""
    # For a dependency-free minimal trainer, approximate with small nearest rotate using tf.contrib-like ops unavailable,
    # so we keep it simple: for smoke/baseline, skip rotation if complex. Use identity.
    # (If you have tensorflow_addons installed, you can use tfa.image.rotate.)
    return image  # no-op to keep code self-contained; flip/shift already help


# ----------------------------- Training -----------------------------

@tf.function
def train_step(x_real, y_real, w, G, D, optG, optD):
    """
    Single training step with true DADA losses.

    Args:
        x_real: (B,32,32,3) in [-1,1]
        y_real: (B,) int32
        w: scalar tf.float32 (0.0 or 1.0)
    """
    B = tf.shape(x_real)[0]
    K = CONFIG['dataset']['num_classes']
    latent = CONFIG['dada']['latent_dim']

    y_fake = tf.random.uniform([B], minval=0, maxval=K, dtype=tf.int32)
    z = tf.random.normal([B, latent])

    with tf.GradientTape() as tapeG, tf.GradientTape() as tapeD:
        x_fake = G([z, y_fake], training=True)

        logits_r, f_r = D(x_real, training=True)
        logits_f, f_f = D(x_fake, training=True)

        comps = compute_dada_components(
            logits_2k_real=logits_r,
            logits_2k_fake=logits_f,
            y_real=y_real,
            y_fake=y_fake,
            f_real=f_r,
            f_fake=f_f,
        )
        losses = combine_dada_losses(comps, w=w)

    gradsG = tapeG.gradient(losses['loss_G'], G.trainable_variables)
    gradsD = tapeD.gradient(losses['loss_D'], D.trainable_variables)
    optG.apply_gradients(zip(gradsG, G.trainable_variables))
    optD.apply_gradients(zip(gradsD, D.trainable_variables))

    return {**{k: tf.cast(v, tf.float32) for k, v in comps.items()},
            **{k: tf.cast(v, tf.float32) for k, v in losses.items()}}


def smoke_test():
    """1-batch overfit smoke test with sample grid output."""
    results_dir = get_path('results_dir', 'results')
    models_dir = get_path('models_dir', 'models')

    print("[smoke] building models ...")
    G = create_generator(CONFIG).build_model()
    D = create_discriminator(CONFIG).build_model()

    # WeightNorm data-init warmup for D (one inert forward)
    _ = D(tf.zeros([4, 32, 32, 3], tf.float32), training=False)

    optG = tf.keras.optimizers.Adam(
        learning_rate=CONFIG['dada']['learning_rate'],
        beta_1=CONFIG['dada']['beta1'],
        beta_2=CONFIG['dada']['beta2']
    )
    optD = tf.keras.optimizers.Adam(
        learning_rate=CONFIG['dada']['learning_rate'],
        beta_1=CONFIG['dada']['beta1'],
        beta_2=CONFIG['dada']['beta2']
    )

    print("[smoke] loading data ...")
    ds = prepare_data(CONFIG['training']['batch_size'], augment=False)
    x_real, y_real = next(iter(ds))

    z_fixed, y_fixed = create_fixed_sampler()

    print("[smoke] training ~200 steps (w=0) ...")
    w = tf.constant(0.0, tf.float32)
    for step in range(200):
        out = train_step(x_real, y_real, w, G, D, optG, optD)
        if step % 10 == 0:
            print(f"step {step:03d} | G={float(out['loss_G']):.4f} "
                  f"D={float(out['loss_D']):.4f} "
                  f"gen_src={float(out['loss_gen_source']):.4f} "
                  f"lab_src={float(out['loss_lab_source']):.4f} "
                  f"fm={float(out['feature_match']):.4f}")

    print("[smoke] saving sample grid ...")
    fake_images = G([z_fixed, y_fixed], training=False)
    save_grid(fake_images, y_fixed, os.path.join(results_dir, 'overfit_grid.png'))
    print("[smoke] done.")


def full_training():
    """Full two-phase training with per-epoch grids and periodic checkpoints."""
    results_dir = get_path('results_dir', 'results')
    models_dir = get_path('models_dir', 'models')

    print("[train] building models ...")
    G = create_generator(CONFIG).build_model()
    D = create_discriminator(CONFIG).build_model()

    # WeightNorm data-init warmup
    _ = D(tf.zeros([4, 32, 32, 3], tf.float32), training=False)

    optG = tf.keras.optimizers.Adam(
        learning_rate=CONFIG['dada']['learning_rate'],
        beta_1=CONFIG['dada']['beta1'],
        beta_2=CONFIG['dada']['beta2']
    )
    optD = tf.keras.optimizers.Adam(
        learning_rate=CONFIG['dada']['learning_rate'],
        beta_1=CONFIG['dada']['beta1'],
        beta_2=CONFIG['dada']['beta2']
    )

    gan_epochs = CONFIG['dada']['gan_epochs']
    total_epochs = CONFIG['dada']['total_epochs']

    z_fixed, y_fixed = create_fixed_sampler()

    for epoch in range(total_epochs):
        t0 = time.time()
        w = tf.constant(0.0, tf.float32) if epoch < gan_epochs else tf.constant(1.0, tf.float32)
        phase = "GAN" if epoch < gan_epochs else "CLS"

        # Turn on light augmentation only after GAN phase (optional)
        ds = prepare_data(CONFIG['training']['batch_size'], augment=(epoch >= gan_epochs))

        loss_G_vals, loss_D_vals = [], []

        for i, (x_real, y_real) in enumerate(ds):
            out = train_step(x_real, y_real, w, G, D, optG, optD)
            # convert to floats for stable logging
            loss_G_vals.append(float(out['loss_G'].numpy()))
            loss_D_vals.append(float(out['loss_D'].numpy()))

            if (i + 1) % 50 == 0:
                print(f"[{phase}] epoch {epoch:04d} batch {i+1:04d} "
                      f"G={np.mean(loss_G_vals[-50:]):.4f} "
                      f"D={np.mean(loss_D_vals[-50:]):.4f}")

        # epoch summary
        dt = time.time() - t0
        print(f"[{phase}] epoch {epoch:04d} | G={np.mean(loss_G_vals):.4f} "
              f"D={np.mean(loss_D_vals):.4f} | {dt:.1f}s")

        # save sample grid
        fake_images = G([z_fixed, y_fixed], training=False)
        save_grid(fake_images, y_fixed, os.path.join(results_dir, f'epoch_{epoch:04d}.png'))

        # periodic checkpoints
        if (epoch + 1) % 50 == 0:
            G.save_weights(os.path.join(models_dir, f'generator_epoch_{epoch:04d}.h5'))
            D.save_weights(os.path.join(models_dir, f'discriminator_epoch_{epoch:04d}.h5'))

    print("[train] complete.")


# ----------------------------- Main -----------------------------

def main():
    parser = argparse.ArgumentParser(description="DADA Trainer")
    parser.add_argument("--mode", choices=["smoke", "train"], default="smoke",
                        help="Run a quick 1-batch overfit (smoke) or full training loop.")
    args = parser.parse_args()

    # sanity: encourage batch size 100 for parity
    if CONFIG['training']['batch_size'] != 100:
        print(f"[warn] batch_size={CONFIG['training']['batch_size']} (original DADA uses 100).")

    if args.mode == "smoke":
        smoke_test()
    else:
        full_training()


if __name__ == "__main__":
    main()
