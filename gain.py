# GAIN
# Generative Adversarial Imputation Nets
# Tensorflow Implementation

import tensorflow as tf

class GAIN(tf.keras.models.Model):
    def __init__(self, num_features, hidden_size, drop_rate):
        super(GAIN, self).__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.drop_rate = drop_rate

        self.generator = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(num_features*2, )),
                tf.keras.layers.Dense(units=hidden_size, activation='relu'),
                tf.keras.layers.Dropout(rate=drop_rate),
                tf.keras.layers.Dense(units=num_features, activation='sigmoid')
            ]
        )

        self.discriminator = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(num_features*2, )),
                tf.keras.layers.Dense(units=hidden_size, activation='relu'),
                tf.keras.layers.Dropout(rate=drop_rate),
                tf.keras.layers.Dense(units=num_features, activation='sigmoid')
            ]
        )

    @tf.function
    def generate(self, data_with_random_mat, M):
        inputs = tf.concat([data_with_random_mat, M], axis=1)
        X_bar = self.generator(inputs)
        return X_bar

    @tf.function
    def discriminate(self, X_hat, hint_mat):
        inputs = tf.concat([X_hat, hint_mat], axis=1)
        estimated_mask_mat = self.discriminator(inputs)
        return estimated_mask_mat

