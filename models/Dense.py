from tensorflow.keras import Model
from tensorflow.keras.layers import Input, ZeroPadding1D, concatenate
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import AveragePooling1D, GlobalAveragePooling1D, MaxPooling1D
from tensorflow.keras.layers import BatchNormalization
import gin

@gin.configurable
def DenseNet(nb_dense_block, nb_layers, growth_rate, nb_filter, reduction, dropout_rate, classes=12):

    eps = 1.1e-5
    # compute compression factor
    compression = 1.0 - reduction
    global concat_axis
    concat_axis = 2
    img_input = Input(shape=(250, 6), name='data')


    # Initial convolution
    x = ZeroPadding1D(3)(img_input)
    x = Conv1D(nb_filter, 7, 2, use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis)(x)
    x = Activation('relu')(x)
    x = ZeroPadding1D(1)(x)
    x = MaxPooling1D(3, strides=2)(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx+2
        x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate)

        # Add transition_block
        x = transition_block(x, nb_filter, compression=compression, dropout_rate=dropout_rate)
        nb_filter = int(nb_filter * compression)

    x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate)

    x = BatchNormalization(epsilon=eps, axis=concat_axis)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(classes)(x)
    x = Activation('softmax')(x)

    model = Model(img_input, x)

    return model


def conv_block(x, nb_filter, dropout_rate=None):

    eps = 1.1e-5
    # 1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4
    x = BatchNormalization(epsilon=eps, axis=concat_axis)(x)
    x = Activation('relu')(x)
    x = Conv1D(inter_channel, 1, use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3 Convolution
    x = BatchNormalization(epsilon=eps, axis=concat_axis)(x)
    x = Activation('relu')(x)
    x = ZeroPadding1D(1)(x)
    x = Conv1D(nb_filter, 3, use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(x, nb_filter, compression=1.0, dropout_rate=None):
    eps = 1.1e-5

    x = BatchNormalization(epsilon=eps, axis=concat_axis)(x)
    x = Activation('relu')(x)
    x = Conv1D(int(nb_filter * compression), 1, use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling1D(2, strides=2)(x)

    return x


def dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=None, grow_nb_filters=True):

    concat_feat = x

    for i in range(nb_layers):

        x = conv_block(concat_feat, growth_rate, dropout_rate)
        concat_feat = concatenate([concat_feat, x], axis=concat_axis)

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter