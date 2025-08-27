import tensorflow as tf
from weightnorm_wrapper import WeightNorm


class DADADiscriminator(tf.keras.Model):
    def __init__(self, num_classes: int, use_weight_norm: bool = True, name: str = "DADADiscriminator"):
        super().__init__(name=name)
        self.num_classes = num_classes
        self.use_weight_norm = use_weight_norm

        def maybe_wn(layer):
            return WeightNorm(layer) if use_weight_norm else layer

        self.gn = tf.keras.layers.GaussianNoise(0.2)

        # Block 1: 96, 96, 96 (stride 2) + Dropout
        self.conv1_1 = maybe_wn(tf.keras.layers.Conv2D(96, 3, padding="same", activation=None))
        self.conv1_2 = maybe_wn(tf.keras.layers.Conv2D(96, 3, padding="same", activation=None))
        self.conv1_3 = maybe_wn(tf.keras.layers.Conv2D(96, 3, strides=2, padding="same", activation=None))
        self.drop1 = tf.keras.layers.Dropout(0.5)

        # Block 2: 192, 192, 192 (stride 2) + Dropout
        self.conv2_1 = maybe_wn(tf.keras.layers.Conv2D(192, 3, padding="same", activation=None))
        self.conv2_2 = maybe_wn(tf.keras.layers.Conv2D(192, 3, padding="same", activation=None))
        self.conv2_3 = maybe_wn(tf.keras.layers.Conv2D(192, 3, strides=2, padding="same", activation=None))
        self.drop2 = tf.keras.layers.Dropout(0.5)

        # Extra conv with pad=0 (valid) like original
        self.conv3 = maybe_wn(tf.keras.layers.Conv2D(192, 3, padding="valid", activation=None))

        # NIN layers (1x1 convs)
        self.nin1 = maybe_wn(tf.keras.layers.Conv2D(192, 1, padding="valid", activation=None))
        self.nin2 = maybe_wn(tf.keras.layers.Conv2D(192, 1, padding="valid", activation=None))

        self.gap = tf.keras.layers.GlobalAveragePooling2D()

        # Penultimate features kept here
        self.dropout_final = tf.keras.layers.Dropout(0.0)

        # Final logits: 2 * C (no activation)
        self.fc_logits = maybe_wn(tf.keras.layers.Dense(2 * num_classes, activation=None))

        self.act_lrelu = tf.keras.layers.LeakyReLU(0.2)

    def call(self, x: tf.Tensor, training: bool = False, return_features: bool = False):
        h = tf.cast(x, tf.float32)
        h = self.gn(h, training=training)

        # Block 1
        h = self.conv1_1(h)
        h = self.act_lrelu(h)
        h = self.conv1_2(h)
        h = self.act_lrelu(h)
        h = self.conv1_3(h)
        h = self.act_lrelu(h)
        h = self.drop1(h, training=training)

        # Block 2
        h = self.conv2_1(h)
        h = self.act_lrelu(h)
        h = self.conv2_2(h)
        h = self.act_lrelu(h)
        h = self.conv2_3(h)
        h = self.act_lrelu(h)
        h = self.drop2(h, training=training)

        # Conv + NINs
        h = self.conv3(h)
        h = self.act_lrelu(h)
        h = self.nin1(h)
        h = self.act_lrelu(h)
        h = self.nin2(h)
        h = self.act_lrelu(h)

        # Global average pool
        features = self.gap(h)
        features = self.dropout_final(features, training=training)

        logits = self.fc_logits(features)

        if return_features:
            return logits, features
        return logits


class TransferDisc(tf.keras.Model):
    def __init__(self, num_classes: int, backbone_name: str = "ResNet50", name: str = "TransferDisc"):
        super().__init__(name=name)
        self.num_classes = num_classes
        self.backbone_name = backbone_name

        self.gn = tf.keras.layers.GaussianNoise(0.2)

        backbone = self._build_backbone(backbone_name)
        self.backbone = backbone
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.dropout_final = tf.keras.layers.Dropout(0.0)
        self.fc_logits = tf.keras.layers.Dense(2 * num_classes, activation=None)

    def _build_backbone(self, name: str) -> tf.keras.Model:
        name = name.lower()
        if name == "resnet50":
            base = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=(32, 32, 3))
            self.preprocess = tf.keras.applications.resnet50.preprocess_input
        elif name == "mobilenetv2":
            base = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(32, 32, 3))
            self.preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
        elif name == "efficientnetb0" or name == "efficientnet":
            base = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(32, 32, 3))
            self.preprocess = tf.keras.applications.efficientnet.preprocess_input
        else:
            raise ValueError("Unsupported transfer backbone: " + name)
        return base

    def call(self, x: tf.Tensor, training: bool = False, return_features: bool = False):
        h = tf.cast(x, tf.float32)
        h = self.gn(h, training=training)
        # Use backbone-specific preprocessing to match ImageNet training distribution
        h = self.preprocess(h)
        h = self.backbone(h, training=training)
        features = self.gap(h)
        features = self.dropout_final(features, training=training)
        logits = self.fc_logits(features)
        if return_features:
            return logits, features
        return logits