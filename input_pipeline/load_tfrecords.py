import gin
import tensorflow as tf
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE

# parse function for S2L task
@gin.configurable
def _parse_function(example_proto):
    feature_description = {'data': tf.io.FixedLenFeature((), tf.string),
                           'label': tf.io.FixedLenFeature((), tf.int64)}
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    data = tf.io.decode_raw(parsed_features['data'], tf.float64)
    data = tf.reshape(data, (-1, 6))
    return data, parsed_features['label']


# parse function for S2S task
def _parse_function_S2S(example_proto):
    feature_description = {'data': tf.io.FixedLenFeature((), tf.string),
                           'label': tf.io.FixedLenFeature((), tf.int64)}
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    data = tf.io.decode_raw(parsed_features['data'], tf.float64)
    label = parsed_features['label']
    data = tf.reshape(data, (-1, 6))
    label = tf.reshape(label, (-1, 1))

    return data, label


def squeeze(x, y):
    x = tf.squeeze(x)
    y = tf.squeeze(y)
    return x, y

@gin.configurable
def load_from_tfrecords(data_dir, batch_size):
    tftrain_path = os.path.join(data_dir, 'train.tfrecords')
    tftest_path = os.path.join(data_dir, 'test.tfrecords')
    tfval_path = os.path.join(data_dir, 'validation.tfrecords')

    raw_train_ds = tf.data.TFRecordDataset(tftrain_path)
    raw_test_ds = tf.data.TFRecordDataset(tftest_path)
    raw_val_ds = tf.data.TFRecordDataset(tfval_path)

    parsed_train_ds = raw_train_ds.map(_parse_function_S2S, num_parallel_calls=AUTOTUNE).cache(os.path.join(data_dir, "train"))
    parsed_val_ds = raw_val_ds.map(_parse_function_S2S, num_parallel_calls=AUTOTUNE).cache(os.path.join(data_dir, "validation"))
    parsed_test_ds = raw_test_ds.map(_parse_function_S2S, num_parallel_calls=AUTOTUNE).cache(os.path.join(data_dir, "test"))

    parsed_train_ds = parsed_train_ds.batch(250, drop_remainder=True)
    parsed_val_ds = parsed_val_ds.batch(250, drop_remainder=True)
    parsed_test_ds = parsed_test_ds.batch(250, drop_remainder=True)
    parsed_train_ds = parsed_train_ds.map(squeeze)
    parsed_val_ds = parsed_val_ds.map(squeeze)
    parsed_test_ds = parsed_test_ds.map(squeeze)

    parsed_train_ds = parsed_train_ds.shuffle(3000).repeat()

    parsed_train_ds = parsed_train_ds.batch(batch_size)
    parsed_val_ds = parsed_val_ds.batch(batch_size)
    parsed_test_ds = parsed_test_ds.batch(batch_size)

    return parsed_train_ds.prefetch(buffer_size=AUTOTUNE), \
           parsed_val_ds.prefetch(buffer_size=AUTOTUNE), \
           parsed_test_ds.prefetch(buffer_size=AUTOTUNE)
