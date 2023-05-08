import tensorflow as tf

from causal_self_attention import CausalSelfAttention
from cross_attention import CrossAttention
from feed_forward import FeedForward


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        """
        Args:
            d_model (integer): self attention: size of each attention head for query and key
            num_heads (integer): self attention: number of attention heads
            dff (integer): internal dimensionality of the FeedForward layer
            dropout_rate (float, optional): self attention: dropout probability. Defaults to 0.1.
        """
        super(DecoderLayer, self).__init__()

        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def call(self, x, context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
        return x
