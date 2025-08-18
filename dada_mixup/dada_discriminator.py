"""
DADA Discriminator reimplementation in TensorFlow/Keras.

- Input: (None, 32, 32, 3) images scaled to [-1, 1]
- Output: logits_2k (B, 2*K) for class-conditional discrimination and features for feature matching
- Architecture: Conv blocks + NIN layers + 2K head (real/fake per class)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow_addons as tfa


class DadaDiscriminator(Model):
    """
    DADA Discriminator with class-conditional discrimination.
    
    The 2K head outputs 2 logits per class: one for real/fake discrimination
    and one for class prediction, enabling class-conditional GAN training.
    Features are exported for feature matching loss in the original DADA.
    """
    
    def __init__(self, config, **kwargs):
        super(DadaDiscriminator, self).__init__(**kwargs)
        self.num_classes = config['dataset']['num_classes']
        self.use_weight_norm = config['dada']['use_weight_norm']
        self.dropout_rate = config['dada']['dropout_rate']
        
        # Gaussian noise for regularization
        self.gaussian_noise = layers.GaussianNoise(0.2, name='gaussian_noise')
        
        # Conv block 1: 96 channels
        self.conv1_1 = self._make_conv(96, 3, padding='same', name='conv1_1')
        self.conv1_2 = self._make_conv(96, 3, padding='same', name='conv1_2')
        self.conv1_3 = self._make_conv(96, 3, strides=2, name='conv1_3')
        self.dropout1 = layers.Dropout(self.dropout_rate, name='dropout1')
        
        # Conv block 2: 192 channels
        self.conv2_1 = self._make_conv(192, 3, padding='same', name='conv2_1')
        self.conv2_2 = self._make_conv(192, 3, padding='same', name='conv2_2')
        self.conv2_3 = self._make_conv(192, 3, strides=2, name='conv2_3')
        self.dropout2 = layers.Dropout(self.dropout_rate, name='dropout2')
        
        # Final conv: 192 channels, valid padding
        self.conv3 = self._make_conv(192, 3, padding='valid', name='conv3')
        
        # NIN layers (1x1 conv as pointwise conv)
        self.nin1 = self._make_conv(192, 1, name='nin1')
        self.nin2 = self._make_conv(192, 1, name='nin2')
        
        # Global average pooling
        self.global_pool = layers.GlobalAveragePooling2D(name='global_pool')
        
        # Final dense layer: 2*K outputs (no activation)
        self.final_dense = self._make_dense(2 * self.num_classes, name='final_dense')
        
    def _make_conv(self, filters, kernel_size, strides=1, padding='same', name=None):
        """Helper to create Conv2D with optional WeightNorm and LeakyReLU."""
        conv = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=True,
            name=name
        )
        
        if self.use_weight_norm:
            conv = tfa.layers.WeightNormalization(conv, data_init=False)
        
        return conv
    
    def _make_dense(self, units, name=None):
        """Helper to create Dense layer with optional WeightNorm."""
        dense = layers.Dense(units, use_bias=True, name=name)
        
        if self.use_weight_norm:
            dense = tfa.layers.WeightNormalization(dense, data_init=False)
        
        return dense
    
    def call(self, inputs, training=None):
        x = inputs
        
        # Gaussian noise
        x = self.gaussian_noise(x, training=training)
        
        # Conv block 1: 96 channels
        x = self.conv1_1(x)
        x = layers.LeakyReLU(0.2)(x)
        x = self.conv1_2(x)
        x = layers.LeakyReLU(0.2)(x)
        x = self.conv1_3(x)
        x = layers.LeakyReLU(0.2)(x)
        x = self.dropout1(x, training=training)
        
        # Conv block 2: 192 channels
        x = self.conv2_1(x)
        x = layers.LeakyReLU(0.2)(x)
        x = self.conv2_2(x)
        x = layers.LeakyReLU(0.2)(x)
        x = self.conv2_3(x)
        x = layers.LeakyReLU(0.2)(x)
        x = self.dropout2(x, training=training)
        
        # Final conv: 192 channels, valid padding (pad=0 in original)
        x = self.conv3(x)
        x = layers.LeakyReLU(0.2)(x)
        
        # NIN layers (1x1 conv as pointwise conv)
        x = self.nin1(x)
        x = layers.LeakyReLU(0.2)(x)
        x = self.nin2(x)
        x = layers.LeakyReLU(0.2)(x)
        
        # Global pooling
        x = self.global_pool(x)
        
        # Store features for feature matching (penultimate layer before final dense)
        features = x
        
        # Final dense layer: 2*K outputs (no activation)
        logits_2k = self.final_dense(x)
        
        return logits_2k, features
    
    def build_model(self):
        """Build a functional Keras model with fresh Inputs for the graph."""
        inputs = layers.Input(shape=(32, 32, 3), name='discriminator_input')
        logits_2k, features = self.call(inputs)
        return Model(inputs=inputs, outputs=[logits_2k, features], name='dada_discriminator')


def create_discriminator(config):
    """
    Factory function to create a DADA discriminator.
    
    Args:
        config: Configuration dictionary containing dataset and DADA settings
    
    Returns:
        DadaDiscriminator: Configured discriminator model
    """
    return DadaDiscriminator(config)


if __name__ == "__main__":
    from dada_config import CONFIG
    
    print("Testing DADA Discriminator...")
    
    # Create discriminator
    discriminator = create_discriminator(CONFIG)
    discriminator_model = discriminator.build_model()
    
    print("\nDADA Discriminator Architecture:")
    discriminator_model.summary()
    
    # Test forward pass
    batch_size = 2
    test_input = tf.random.uniform([batch_size, 32, 32, 3], minval=-1.0, maxval=1.0)
    
    logits_2k, features = discriminator_model(test_input)
    
    print(f"\nInput shape: {test_input.shape}")
    print(f"logits_2k shape: {logits_2k.shape}")
    print(f"features shape: {features.shape}")
    print(f"Expected logits_2k shape: (batch_size, 2*{CONFIG['dataset']['num_classes']}) = ({batch_size}, {2 * CONFIG['dataset']['num_classes']})")
    
    print("\n✅ DADA Discriminator implementation complete!") 