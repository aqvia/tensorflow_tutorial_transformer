import tensorflow as tf

from encoder import Encoder
from decoder import Decoder


class Transformer(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff,
                 input_vocab_size, target_vocab_size, dropout_rate=0.1):
        """
        Args:
            num_layers (integer): num of layers
            d_model (integer): dimension of the embedding
            num_heads (integer): number of attention heads
            dff (integer): dimension of the feedforward
            input_vocab_size (integer): size of the vocabulary: for encoder
            target_vocab_size (integer): size of the vocabulary: for decoder
            dropout_rate (float, optional): dropout probability. Defaults to 0.1.
        """
        super().__init__()
        self.encoder = Encoder(num_layers=num_layers,
                               d_model=d_model,
                               num_heads=num_heads,
                               dff=dff,
                               vocab_size=input_vocab_size,
                               dropout_rate=dropout_rate)

        self.decoder = Decoder(num_layers=num_layers,
                               d_model=d_model,
                               num_heads=num_heads,
                               dff=dff,
                               vocab_size=target_vocab_size,
                               dropout_rate=dropout_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs):
        # Kerasモデルの`fit`メソッドを使用するには、`inputs`にすべての入力を渡すこと
        context, x = inputs

        context = self.encoder(context)  # (batch_size, context_len, d_model)

        x = self.decoder(x, context)  # (batch_size, target_len, d_model)

        # 最終の linear layer の出力
        # (batch_size, target_len, target_vocab_size)
        logits = self.final_layer(x)

        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            # b/250038731
            del logits._keras_mask
        except AttributeError:
            pass

        # Return the final output and the attention weights.
        return logits
