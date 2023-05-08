import tensorflow as tf

from global_self_attention import GlobalSelfAttention
from feed_forward import FeedForward


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        """
        Args:
            d_model (integer): self attention: size of each attention head for query and key
            num_heads (integer): self attention: number of attention heads
            dff (integer): internal dimensionality of the FeedForward layer
            dropout_rate (float, optional): self attention: dropout probability. Defaults to 0.1.
        """
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x
