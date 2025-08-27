import tensorflow as tf

class WeightNorm(tf.keras.layers.Wrapper):
    """
    Weight Normalization (Salimans & Kingma, 2016).
    Works with Dense, Conv2D, Conv2DTranspose.
    Reparameterizes: w = g * v / ||v||, optimizing v and g.
    """

    def __init__(self, layer: tf.keras.layers.Layer, eps: float = 1e-8, **kwargs):
        if not isinstance(layer, tf.keras.layers.Layer):
            raise ValueError("WeightNorm must wrap a Keras layer.")
        super().__init__(layer, **kwargs)
        self.eps = eps
        self.built_wn = False

    def build(self, input_shape):
        # Build inner layer if needed
        if not self.layer.built:
            self.layer.build(input_shape)
        if not hasattr(self.layer, "kernel"):
            raise ValueError(f"Wrapped layer {self.layer.name} has no 'kernel'.")

        k = self.layer.kernel
        k_shape = k.shape

        # Determine axes
        if len(k_shape) == 2:        # Dense: (in, out)
            out_axis = 1
            reduce_axes = (0,)
        elif len(k_shape) == 4:      # Conv2D: (kh, kw, in, out) or Conv2DTranspose
            out_axis = 3
            reduce_axes = (0, 1, 2)
            filters = getattr(self.layer, "filters", None)
            if filters is not None and k_shape[2] == filters:  # heuristic for transpose
                out_axis = 2
                reduce_axes = (0, 1, 3)
        else:
            raise ValueError("Unsupported kernel rank for WeightNorm.")

        self._out_axis = out_axis
        self._reduce_axes = reduce_axes

        # Trainable params: v (copy of kernel) and g (scaling)
        self.v = self.add_weight(
            name="wn_v",
            shape=k_shape,
            initializer=lambda shape, dtype=None: tf.identity(k),
            trainable=True,
            dtype=k.dtype,
        )
        g_shape = (k_shape[out_axis],)
        self.g = self.add_weight(
            name="wn_g",
            shape=g_shape,
            initializer="ones",
            trainable=True,
            dtype=k.dtype,
        )

        self.built_wn = True
        super().build(input_shape)

    def _compute_w(self):
        v = self.v
        v_sq = tf.square(v)
        v_sum = tf.reduce_sum(v_sq, axis=self._reduce_axes, keepdims=True)
        v_norm = tf.sqrt(tf.maximum(v_sum, tf.cast(self.eps, v.dtype)))
        bshape = [1] * len(v.shape)
        bshape[self._out_axis] = v.shape[self._out_axis]
        g_b = tf.reshape(self.g, bshape)
        return g_b * (v / v_norm)

    def call(self, inputs, training=None):
        if not self.built_wn:
            raise RuntimeError("WeightNorm wrapper not built.")
        w = self._compute_w()

        # Dense
        if isinstance(self.layer, tf.keras.layers.Dense):
            outputs = tf.matmul(inputs, w)
            if getattr(self.layer, "use_bias", False) and self.layer.bias is not None:
                outputs = tf.nn.bias_add(outputs, self.layer.bias)

        # Conv2D
        elif isinstance(self.layer, tf.keras.layers.Conv2D):
            strides = (1,) + self.layer.strides + (1,)
            outputs = tf.nn.conv2d(
                inputs,
                filters=w,
                strides=strides,
                padding=self.layer.padding.upper(),
                dilations=self.layer.dilation_rate,
            )
            if getattr(self.layer, "use_bias", False) and self.layer.bias is not None:
                outputs = tf.nn.bias_add(outputs, self.layer.bias)

        # Conv2DTranspose
        elif isinstance(self.layer, tf.keras.layers.Conv2DTranspose):
            strides = (1,) + self.layer.strides + (1,)
            input_shape = tf.shape(inputs)
            batch = input_shape[0]
            in_h = input_shape[1]
            in_w = input_shape[2]
            kernel_h, kernel_w = self.layer.kernel_size
            stride_h, stride_w = self.layer.strides
            filters = self.layer.filters

            if self.layer.padding.upper() == "SAME":
                out_h = in_h * stride_h
                out_w = in_w * stride_w
            else:  # VALID
                out_h = (in_h - 1) * stride_h + kernel_h
                out_w = (in_w - 1) * stride_w + kernel_w

            output_shape = tf.stack([batch, out_h, out_w, filters])
            outputs = tf.nn.conv2d_transpose(
                inputs,
                filters=w,
                output_shape=output_shape,
                strides=strides,
                padding=self.layer.padding.upper(),
                dilations=self.layer.dilation_rate,
            )
            if getattr(self.layer, "use_bias", False) and self.layer.bias is not None:
                outputs = tf.nn.bias_add(outputs, self.layer.bias)

        else:
            raise ValueError("Unsupported layer type for WeightNorm.")

        # Apply activation if present
        if self.layer.activation is not None:
            outputs = self.layer.activation(outputs)

        return outputs
