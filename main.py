import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text

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
MAX_TOKENS = 128


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
