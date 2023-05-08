import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt


def positional_encoding(length, depth):
    depth = depth/2

    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, d_model, mask_zero=True)
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x


if __name__ == '__main__':
    pos_encoding = positional_encoding(length=2048, depth=512)

    # Check the shape.
    print(pos_encoding.shape)

    # # Plot the dimensions.
    # plt.pcolormesh(pos_encoding.numpy().T, cmap='RdBu')
    # plt.ylabel('Depth')
    # plt.xlabel('Position')
    # plt.colorbar()
    # plt.show()

    pos_encoding /= tf.norm(pos_encoding, axis=1, keepdims=True)
    p = pos_encoding[1000]
    dots = tf.einsum('pd,d -> p', pos_encoding, p)
    plt.subplot(2, 1, 1)
    plt.plot(dots)
    plt.ylim([0, 1])
    plt.plot([950, 950, float('nan'), 1050, 1050],
             [0, 1, float('nan'), 0, 1], color='k', label='Zoom')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(dots)
    plt.xlim([950, 1050])
    plt.ylim([0, 1])
