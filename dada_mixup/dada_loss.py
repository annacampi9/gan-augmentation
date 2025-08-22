"""
DADA Loss Utilities

Pure, reusable loss helpers that reproduce DADA's logic for training
the discriminator and generator with source/class classification.
"""

import tensorflow as tf


def reshape_logits_2k(logits_2k, num_classes):
    """
    Reshape discriminator logits from (B, 2*K) to (B, 2, K) format.
    
    Args:
        logits_2k: tf.Tensor, shape (B, 2*K)
        num_classes: int
    
    Returns:
        tf.Tensor, shape (B, 2, K)  # (source_bit, class)
    """
    batch_size = tf.shape(logits_2k)[0]
    return tf.reshape(logits_2k, [batch_size, 2, num_classes])


def heads_from_logits(logits_2k_real, logits_2k_fake, y_real, y_fake, num_classes):
    """
    Extract source and class heads from discriminator logits.
    
    Args:
        logits_2k_real: (B, 2*K) raw logits from D on real images
        logits_2k_fake: (B, 2*K) raw logits from D on fake images
        y_real: int32 tensor (B,) true labels of real batch
        y_fake: int32 tensor (B,) conditioning labels used for generator batch
        num_classes: int
    
    Returns:
        dict with:
          'src_real_2': (B,2)  # picked per-sample class channel for real
          'src_fake_2': (B,2)  # picked per-sample class channel for fake
          'cls_real_k': (B,K)  # class logits (sum over source bit) for real
          'cls_fake_k': (B,K)  # class logits (sum over source bit) for fake
    """
    # Reshape to (B, 2, K)
    logits_real_reshaped = reshape_logits_2k(logits_2k_real, num_classes)
    logits_fake_reshaped = reshape_logits_2k(logits_2k_fake, num_classes)
    
    # Create one-hot encodings for class selection
    y_real_onehot = tf.one_hot(y_real, num_classes, dtype=tf.float32)  # (B, K)
    y_fake_onehot = tf.one_hot(y_fake, num_classes, dtype=tf.float32)  # (B, K)
    
    # Select per-sample class channel with einsum
    # (B, 2, K) @ (B, K) -> (B, 2)
    src_real_2 = tf.einsum('bsk,bk->bs', logits_real_reshaped, y_real_onehot)
    src_fake_2 = tf.einsum('bsk,bk->bs', logits_fake_reshaped, y_fake_onehot)
    
    # Class logits are sum over source bit
    cls_real_k = tf.reduce_sum(logits_real_reshaped, axis=1)  # (B, K)
    cls_fake_k = tf.reduce_sum(logits_fake_reshaped, axis=1)  # (B, K)
    
    return {
        'src_real_2': src_real_2,
        'src_fake_2': src_fake_2,
        'cls_real_k': cls_real_k,
        'cls_fake_k': cls_fake_k
    }


def compute_dada_components(
    logits_2k_real,
    logits_2k_fake,
    y_real,
    y_fake,
    f_real,
    f_fake,
    fm_weight=0.5,
):
    """
    Compute all DADA loss components.
    
    Args:
        logits_2k_real: (B, 2*K) raw logits from D on real images
        logits_2k_fake: (B, 2*K) raw logits from D on fake images
        y_real: int32 tensor (B,) true labels of real batch
        y_fake: int32 tensor (B,) conditioning labels used for generator batch
        f_real: (B, 192) penultimate features from D on real images
        f_fake: (B, 192) penultimate features from D on fake images
        fm_weight: float, weight for feature matching loss
    
    Returns:
        dict of scalar losses:
          'loss_gen_source'   # generator wants fake->REAL (source target=0) on its class
          'loss_lab_source'   # discriminator source CE: real->0 on true class & fake->1 on its class
          'loss_lab_class'    # discriminator class CE on real images
          'loss_gen_class'    # generator class CE on fake images
          'feature_match'     # L1 between batch means of penultimate features
    """
    num_classes = tf.shape(logits_2k_real)[1] // 2
    
    # Extract heads
    heads = heads_from_logits(logits_2k_real, logits_2k_fake, y_real, y_fake, num_classes)
    
    # Initialize loss functions
    source_ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    class_ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # Source classification targets
    # Real images should be classified as real (0), fake as fake (1)
    source_real_targets = tf.zeros(tf.shape(y_real), dtype=tf.int32)  # (B,) all zeros
    source_fake_targets = tf.ones(tf.shape(y_fake), dtype=tf.int32)   # (B,) all ones
    
    # Generator source loss: wants fake images to be classified as real (0)
    loss_gen_source = source_ce(source_real_targets, heads['src_fake_2'])
    
    # Discriminator source loss: real->0, fake->1
    loss_lab_source_real = source_ce(source_real_targets, heads['src_real_2'])
    loss_lab_source_fake = source_ce(source_fake_targets, heads['src_fake_2'])
    loss_lab_source = loss_lab_source_real + loss_lab_source_fake
    
    # Class classification losses
    loss_lab_class = class_ce(y_real, heads['cls_real_k'])  # D on real
    loss_gen_class = class_ce(y_fake, heads['cls_fake_k'])  # G on fake
    
    # Feature matching: L1 between batch means
    f_real_mean = tf.reduce_mean(f_real, axis=0)  # (192,)
    f_fake_mean = tf.reduce_mean(f_fake, axis=0)  # (192,)
    feature_match = tf.reduce_mean(tf.abs(f_real_mean - f_fake_mean))
    
    return {
        'loss_gen_source': loss_gen_source,
        'loss_lab_source': loss_lab_source,
        'loss_lab_class': loss_lab_class,
        'loss_gen_class': loss_gen_class,
        'feature_match': feature_match
    }


def combine_dada_losses(components, w):
    """
    Combine DADA loss components with epoch weighting.
    
    Args:
        components: dict from compute_dada_components
        w: float tensor scalar in {0.0, 1.0} (0 for GAN phase <200 epochs, 1 for classification phase >=200)
    
    Returns:
        dict with:
          'loss_G' = (1-w) * (loss_gen_source + 0.5*feature_match) + w * (loss_gen_class)
          'loss_D' = (1-w) * (loss_lab_source)                     + w * (loss_lab_class)
    """
    # GAN phase losses (w=0)
    gan_loss_g = components['loss_gen_source'] + 0.5 * components['feature_match']
    gan_loss_d = components['loss_lab_source']
    
    # Classification phase losses (w=1)
    cls_loss_g = components['loss_gen_class']
    cls_loss_d = components['loss_lab_class']
    
    # Combine with weighting
    loss_G = (1 - w) * gan_loss_g + w * cls_loss_g
    loss_D = (1 - w) * gan_loss_d + w * cls_loss_d
    
    return {
        'loss_G': loss_G,
        'loss_D': loss_D
    }


if __name__ == "__main__":
    # Self-test with dummy data
    B = 8
    K = 10
    
    # Create dummy tensors
    logits_real = tf.random.normal([B, 2*K])
    logits_fake = tf.random.normal([B, 2*K])
    y_real = tf.random.uniform([B], 0, K, dtype=tf.int32)
    y_fake = tf.random.uniform([B], 0, K, dtype=tf.int32)
    f_real = tf.random.normal([B, 192])
    f_fake = tf.random.normal([B, 192])
    
    print("Testing DADA loss utilities...")
    
    # Test reshape function
    reshaped = reshape_logits_2k(logits_real, K)
    print(f"Reshape test: {logits_real.shape} -> {reshaped.shape}")
    assert reshaped.shape == (B, 2, K), f"Expected (B,2,K), got {reshaped.shape}"
    
    # Test head extraction
    heads = heads_from_logits(logits_real, logits_fake, y_real, y_fake, K)
    print(f"Heads extracted: {list(heads.keys())}")
    assert heads['src_real_2'].shape == (B, 2), f"Expected (B,2), got {heads['src_real_2'].shape}"
    assert heads['cls_real_k'].shape == (B, K), f"Expected (B,K), got {heads['cls_real_k'].shape}"
    
    # Test component computation
    components = compute_dada_components(
        logits_real, logits_fake, y_real, y_fake, f_real, f_fake
    )
    print("Components computed:")
    for name, value in components.items():
        print(f"  {name}: {value.numpy():.4f}")
        assert tf.math.is_finite(value), f"{name} is not finite: {value}"
    
    # Test loss combination
    losses_w0 = combine_dada_losses(components, w=0.0)
    losses_w1 = combine_dada_losses(components, w=1.0)
    
    print(f"Combined losses (w=0.0): G={losses_w0['loss_G'].numpy():.4f}, D={losses_w0['loss_D'].numpy():.4f}")
    print(f"Combined losses (w=1.0): G={losses_w1['loss_G'].numpy():.4f}, D={losses_w1['loss_D'].numpy():.4f}")
    
    assert tf.math.is_finite(losses_w0['loss_G']), "loss_G (w=0) is not finite"
    assert tf.math.is_finite(losses_w0['loss_D']), "loss_D (w=0) is not finite"
    assert tf.math.is_finite(losses_w1['loss_G']), "loss_G (w=1) is not finite"
    assert tf.math.is_finite(losses_w1['loss_D']), "loss_D (w=1) is not finite"
    
    print("✅ All tests passed!")