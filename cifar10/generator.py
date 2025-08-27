import tensorflow as tf
from weightnorm_wrapper import WeightNorm


def broadcast_labels_to_feature_maps(one_hot_labels: tf.Tensor, height: int, width: int) -> tf.Tensor:
    one_hot_labels = tf.cast(one_hot_labels, tf.float32)
    y = tf.reshape(one_hot_labels, (-1, one_hot_labels.shape[-1], 1, 1))
    y = tf.tile(y, multiples=(1, 1, height, width))
    y = tf.transpose(y, perm=[0, 2, 3, 1])
    return y


class DADAGenerator(tf.keras.Model):
    def __init__(self, num_classes: int, noise_dim: int = 100, weight_norm_all: bool = False, name: str = "DADAGenerator"):
        super().__init__(name=name)
        self.num_classes = num_classes
        self.noise_dim = noise_dim
        self.weight_norm_all = weight_norm_all

        dense = tf.keras.layers.Dense(4 * 4 * 512, activation=None)
        self.dense = WeightNorm(dense) if weight_norm_all else dense
        self.bn0 = tf.keras.layers.BatchNormalization()

        def maybe_wn(layer):
            return WeightNorm(layer) if weight_norm_all else layer

        # 4x4x512 -> 8x8x256
        self.deconv1 = maybe_wn(tf.keras.layers.Conv2DTranspose(256, kernel_size=5, strides=2, padding="same", activation=None, use_bias=False))
        self.bn1 = tf.keras.layers.BatchNormalization()

        # 8x8x(256+C) -> 16x16x128
        self.deconv2 = maybe_wn(tf.keras.layers.Conv2DTranspose(128, kernel_size=5, strides=2, padding="same", activation=None, use_bias=False))
        self.bn2 = tf.keras.layers.BatchNormalization()

        # 16x16x(128+C) -> 32x32x3
        last_conv = tf.keras.layers.Conv2DTranspose(3, kernel_size=5, strides=2, padding="same", activation=None)
        # Apply weight norm on last layer as in original
        self.deconv3 = WeightNorm(last_conv)

        self.act_relu = tf.keras.layers.ReLU()
        self.act_tanh = tf.keras.layers.Activation("tanh")

    def call(self, z: tf.Tensor, y: tf.Tensor, training: bool = False) -> tf.Tensor:
        # y: int labels or one-hot; ensure one-hot
        if y.dtype.is_integer:
            y_one = tf.one_hot(y, depth=self.num_classes)
        else:
            y_one = y
        y_one = tf.cast(y_one, tf.float32)

        # MLP concat
        z = tf.cast(z, tf.float32)
        z_concat = tf.concat([z, y_one], axis=-1)
        x = self.dense(z_concat)
        x = self.bn0(x, training=training)
        x = self.act_relu(x)
        x = tf.reshape(x, (-1, 4, 4, 512))

        # Conv concat
        y_map = broadcast_labels_to_feature_maps(y_one, height=4, width=4)
        x = tf.concat([x, y_map], axis=-1)

        x = self.deconv1(x, training=training)
        x = self.bn1(x, training=training)
        x = self.act_relu(x)

        # concat again
        y_map = broadcast_labels_to_feature_maps(y_one, height=8, width=8)
        x = tf.concat([x, y_map], axis=-1)

        x = self.deconv2(x, training=training)
        x = self.bn2(x, training=training)
        x = self.act_relu(x)

        # concat again before last
        y_map = broadcast_labels_to_feature_maps(y_one, height=16, width=16)
        x = tf.concat([x, y_map], axis=-1)

        x = self.deconv3(x, training=training)
        x = self.act_tanh(x)
        return x

    @tf.function
    def sample(self, y: tf.Tensor, z: tf.Tensor | None = None, batch_size: int | None = None) -> tf.Tensor:
        if z is None:
            if batch_size is None:
                batch_size = tf.shape(y)[0]
            z = tf.random.normal((batch_size, self.noise_dim), mean=0.0, stddev=1.0)
        return self.call(z, y, training=False)