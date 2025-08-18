"""
DADA Generator reimplementation in TensorFlow/Keras.

- Input: Noise vector + class label
- Output: 32x32x3 synthetic image
- Architecture: Progressive upsampling + DADA-style class conditioning
"""

import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from dada_config import CONFIG


class ClassConditioningLayer(layers.Layer):
    """
    DADA-style Class Conditioning Layer.
    Concatenates a one-hot encoded class map to the feature map along channels.
    """
    def __init__(self, num_classes, **kwargs):
        super(ClassConditioningLayer, self).__init__(**kwargs)
        self.num_classes = num_classes

    def call(self, inputs):
        feature_map, class_labels = inputs
        class_one_hot = tf.one_hot(class_labels, depth=self.num_classes)
        batch_size = tf.shape(feature_map)[0]
        height = tf.shape(feature_map)[1]
        width = tf.shape(feature_map)[2]
        class_map = tf.reshape(class_one_hot, (batch_size, 1, 1, self.num_classes))
        class_map = tf.tile(class_map, (1, height, width, 1))
        return tf.concat([feature_map, class_map], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config


class DadaGenerator(Model):
    def __init__(self, config, **kwargs):
        super(DadaGenerator, self).__init__(**kwargs)
        self.latent_dim = config['dada']['latent_dim']
        self.num_classes = config['dataset']['num_classes']
        self.use_weight_norm = config['dada']['use_weight_norm']
        self.use_batch_norm = config['dada']['use_batch_norm']

        # Initial dense to spatial feature map
        self.initial_dense = layers.Dense(4 * 4 * 512, activation='relu', name='initial_dense')
        self.reshape_layer = layers.Reshape((4, 4, 512), name='reshape')

        # Class conditioning layers
        self.class_conditioning_1 = ClassConditioningLayer(self.num_classes, name='class_conditioning_1')
        self.class_conditioning_2 = ClassConditioningLayer(self.num_classes, name='class_conditioning_2')
        self.class_conditioning_3 = ClassConditioningLayer(self.num_classes, name='class_conditioning_3')

        # Upsample block 1: 4x4 -> 8x8
        self.upsample_1 = self._make_conv_t(256, name='upsample_1')
        if self.use_batch_norm:
            self.batch_norm_1 = layers.BatchNormalization(name='batch_norm_1')

        # Upsample block 2: 8x8 -> 16x16
        self.upsample_2 = self._make_conv_t(128, name='upsample_2')
        if self.use_batch_norm:
            self.batch_norm_2 = layers.BatchNormalization(name='batch_norm_2')

        # Final conv: 16x16 -> 32x32
        self.final_conv = self._make_conv_t(3, activation='tanh', name='final_conv')

    def _make_conv_t(self, filters, kernel_size=5, strides=2, padding='same',
                     activation='relu', name=None):
        """Helper to create Conv2DTranspose with optional WeightNorm."""
        conv = layers.Conv2DTranspose(filters=filters,
                                      kernel_size=kernel_size,
                                      strides=strides,
                                      padding=padding,
                                      activation=activation,
                                      name=name)
        if self.use_weight_norm:
            return tfa.layers.WeightNormalization(conv, data_init=False)
        else:
            return conv

    def call(self, inputs, training=None):
        noise, class_labels = inputs

        x = self.initial_dense(noise)
        x = self.reshape_layer(x)

        x = self.class_conditioning_1([x, class_labels])
        x = self.upsample_1(x)
        if self.use_batch_norm:
            x = self.batch_norm_1(x, training=training)

        x = self.class_conditioning_2([x, class_labels])
        x = self.upsample_2(x)
        if self.use_batch_norm:
            x = self.batch_norm_2(x, training=training)

        x = self.class_conditioning_3([x, class_labels])
        x = self.final_conv(x)

        return x

    def build_model(self):
        """Build a functional Keras model with fresh Inputs for the graph."""
        noise_in = layers.Input(shape=(self.latent_dim,), name='noise_input')
        class_in = layers.Input(shape=(), dtype=tf.int32, name='class_input')
        outputs = self.call([noise_in, class_in])
        return Model(inputs=[noise_in, class_in], outputs=outputs, name='dada_generator')


def create_generator(config):
    return DadaGenerator(config)


def test_generator(generator_model, config, batch_size=4):
    latent_dim = config['dada']['latent_dim']
    num_classes = config['dataset']['num_classes']
    sample_noise = tf.random.normal([batch_size, latent_dim])
    sample_labels = tf.random.uniform([batch_size], minval=0, maxval=num_classes, dtype=tf.int32)
    sample_images = generator_model([sample_noise, sample_labels])
    return sample_images, sample_labels


def save_generator(generator_model, config, model_name='dada_generator'):
    """
    Save the generator model.
    If weight normalization is enabled, only save weights to avoid TFA serialization issues.
    """
    model_path = os.path.join(config['paths']['models_dir'], model_name)

    if config['dada']['use_weight_norm']:
        weights_path = model_path + '.weights.h5'
        generator_model.save_weights(weights_path)
        print(f"⚠️  WeightNormalization active — saved weights only to: {weights_path}")
        return weights_path
    else:
        generator_model.save(model_path)
        print(f"✅ Full model saved to: {model_path}")
        return model_path


def load_generator(config, model_path):
    """
    Load a generator model from either a full model path or a weights file.
    """
    generator = create_generator(config)
    generator_model = generator.build_model()

    if model_path.endswith('.weights.h5'):
        generator_model.load_weights(model_path)
        print(f"✅ Loaded generator weights from: {model_path}")
    else:
        generator_model = tf.keras.models.load_model(
            model_path,
            custom_objects={'ClassConditioningLayer': ClassConditioningLayer}
        )
        print(f"✅ Loaded full generator model from: {model_path}")

    return generator_model


if __name__ == "__main__":
    print("Testing DADA Generator...")

    generator = create_generator(CONFIG)
    generator_model = generator.build_model()

    print("\nDADA Generator Architecture:")
    generator_model.summary()

    sample_images, sample_labels = test_generator(generator_model, CONFIG)
    print(f"\nGenerated images shape: {sample_images.shape}")
    print(f"Image value range: [{tf.reduce_min(sample_images):.3f}, {tf.reduce_max(sample_images):.3f}]")

    model_path = save_generator(generator_model, CONFIG)
    print(f"\nGenerator model saved to: {model_path}")

    def visualize_samples(images, labels, num=4):
        images = (images + 1) / 2  # Convert [-1, 1] to [0, 1]
        plt.figure(figsize=(10, 2))
        for i in range(num):
            plt.subplot(1, num, i + 1)
            plt.imshow(images[i])
            plt.title(f"Class {labels[i].numpy()}")
            plt.axis("off")
        plt.suptitle("Generated Samples")
        plt.tight_layout()
        plt.show()

    visualize_samples(sample_images, sample_labels)
    print("\n✅ DADA Generator implementation complete!")
