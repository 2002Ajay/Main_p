# CBAM Module
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Conv2D, GlobalMaxPooling2D, GlobalAveragePooling2D, Reshape, Concatenate, Multiply, Activation 

class CBAMLayer(Layer):
    def __init__(self, ratio=8, **kwargs):
        super(CBAMLayer, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        # Define variables and layers that create variables in build
        channel = input_shape[-1]
        self.shared_dense_one = Dense(channel // self.ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
        self.shared_dense_two = Dense(channel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
        self.conv2d = Conv2D(1, (7, 7), padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)

    def call(self, inputs):
        # Channel attention
        avg_pool = GlobalAveragePooling2D()(inputs)
        avg_pool = Reshape((1, 1, -1))(avg_pool)
        avg_pool = self.shared_dense_one(avg_pool)
        avg_pool = self.shared_dense_two(avg_pool)

        max_pool = GlobalMaxPooling2D()(inputs)
        max_pool = Reshape((1, 1, -1))(max_pool)
        max_pool = self.shared_dense_one(max_pool)
        max_pool = self.shared_dense_two(max_pool)

        channel_attention = Activation('sigmoid')(avg_pool + max_pool)
        channel_refined_feature = Multiply()([inputs, channel_attention])

        # Spatial attention
        avg_pool = tf.reduce_mean(channel_refined_feature, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(channel_refined_feature, axis=-1, keepdims=True)
        concat = Concatenate(axis=-1)([avg_pool, max_pool])

        # Apply convolution layer defined in build
        spatial_attention = self.conv2d(concat)
        refined_feature = Multiply()([channel_refined_feature, spatial_attention])

        return refined_feature

    def get_config(self):
        config = super(CBAMLayer, self).get_config()
        config.update({
            'ratio': self.ratio,
        })
        return config
