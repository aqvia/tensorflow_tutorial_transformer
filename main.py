import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text

from . import MAX_TOKENS
from transformer import Transformer
from custom_schedule import CustomSchedule
from loss_and_metrics import masked_accuracy, masked_loss
from translator import Translator, print_translation
from create_attention_plots import plot_attention_head, plot_attention_weights
from export_translator import ExportTranslator

examples, metadata = tfds.load(
    'ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']

# for pt_examples, en_examples in train_examples.batch(3).take(1):
#   print('> Examples in Portuguese:')
#   for pt in pt_examples.numpy():
#     print(pt.decode('utf-8'))
#   print()

#   print('> Examples in English:')
#   for en in en_examples.numpy():
#     print(en.decode('utf-8'))

# set up the tokenizer
model_name = 'ted_hrlr_translate_pt_en_converter'
tf.keras.utils.get_file(
    f'{model_name}.zip',
    f'https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip',
    cache_dir='.',
    cache_subdir='',
    extract=True
)
tokenizers = tf.saved_model.load(model_name)
# print([item for item in dir(tokenizers.en) if not item.startswith('_')])

# set up a data pipeline with tf.data


def prepare_batch(pt, en):
    """テキストのバッチを入力として、訓練データの形式で返す
    """
    pt = tokenizers.pt.tokenize(pt)
    pt = pt[:, :MAX_TOKENS]
    pt = pt.to_tensor()

    en = tokenizers.en.tokenize(en)
    en = en[:, :(MAX_TOKENS+1)]
    en_inputs = en[:, :-1].to_tensor()
    en_labels = en[:, 1:].to_tensor()

    return (pt, en_inputs), en_labels


BUFFER_SIZE = 20000
BATCH_SIZE = 64


def make_batches(ds):
    return (
        ds
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .map(prepare_batch, tf.data.AUTOTUNE)
        .prefetch(buffer_size=tf.data.AUTOTUNE))


# test the dataset
train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)

for (pt, en), en_labels in train_batches.take(1):
    break

# print(pt.shape)
# print(en.shape)
# print(en_labels.shape)
# print(en[0][:10])
# print(en_labels[0][:10])


# Transformer
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1

transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
    target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
    dropout_rate=dropout_rate)
output = transformer((pt, en))

print(en.shape)
print(pt.shape)
print(output.shape)

attn_scores = transformer.decoder.dec_layers[-1].last_attn_scores
print(attn_scores.shape)  # (batch, heads, target_seq, input_seq)

transformer.summary()


# Training
# set up the optimizer
learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

transformer.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy]
)
transformer.fit(train_batches, epochs=20, validation_data=val_batches)

# run
translator = Translator(tokenizers, transformer)

# example
sentence = 'este é um problema que temos que resolver.'
ground_truth = 'this is a problem we have to solve .'

translated_text, translated_tokens, attention_weights = translator(
    tf.constant(sentence))
print_translation(sentence, translated_text, ground_truth)

# create attention plots
head = 0
# Shape: `(batch=1, num_heads, seq_len_q, seq_len_k)`.
attention_heads = tf.squeeze(attention_weights, 0)
attention = attention_heads[head]

in_tokens = tf.convert_to_tensor([sentence])
in_tokens = tokenizers.pt.tokenize(in_tokens).to_tensor()
in_tokens = tokenizers.pt.lookup(in_tokens)[0]

plot_attention_head(in_tokens, translated_tokens, attention)
plot_attention_weights(sentence, translated_tokens,
                       attention_weights[0], tokenizers)

# Export the model
translator = ExportTranslator(translator)
tf.saved_model.save(translator, export_dir='translator')
reloaded = tf.saved_model.load('translator')
