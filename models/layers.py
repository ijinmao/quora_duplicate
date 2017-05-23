from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Lambda, Dense, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import Bidirectional
from keras.legacy.layers import Highway
from keras.layers.convolutional import Conv1D
from keras.layers import TimeDistributed
import keras.backend as K
from keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate


class WordRepresLayer(object):
    """Word embedding representation layer
    """
    def __init__(self, sequence_length, nb_words,
                 word_embedding_dim, embedding_matrix):
        self.model = Sequential()
        self.model.add(Embedding(nb_words,
                                 word_embedding_dim,
                                 weights=[embedding_matrix],
                                 input_length=sequence_length,
                                 trainable=False))

    def __call__(self, inputs):
        return self.model(inputs)


class CharRepresLayer(object):
    """Char embedding representation layer
    """
    def __init__(self, sequence_length, nb_chars, nb_per_word,
                 embedding_dim, rnn_dim, rnn_unit='gru', dropout=0.0):
        def _collapse_input(x, nb_per_word=0):
            x = K.reshape(x, (-1, nb_per_word))
            return x

        def _unroll_input(x, sequence_length=0, rnn_dim=0):
            x = K.reshape(x, (-1, sequence_length, rnn_dim))
            return x

        if rnn_unit == 'gru':
            rnn = GRU
        else:
            rnn = LSTM
        self.model = Sequential()
        self.model.add(Lambda(_collapse_input,
                              arguments={'nb_per_word': nb_per_word},
                              output_shape=(nb_per_word,),
                              input_shape=(sequence_length, nb_per_word,)))
        self.model.add(Embedding(nb_chars,
                                 embedding_dim,
                                 input_length=nb_per_word,
                                 trainable=True))
        self.model.add(rnn(rnn_dim,
                           dropout=dropout,
                           recurrent_dropout=dropout))
        self.model.add(Lambda(_unroll_input,
                              arguments={'sequence_length': sequence_length,
                                         'rnn_dim': rnn_dim},
                              output_shape=(sequence_length, rnn_dim)))
        
    def __call__(self, inputs):
        return self.model(inputs)


class ContextLayer(object):
    """Word context layer
    """
    def __init__(self, rnn_dim, rnn_unit='gru', input_shape=(0,),
                 dropout=0.0, highway=False, return_sequences=False,
                 dense_dim=0):
        if rnn_unit == 'gru':
            rnn = GRU
        else:
            rnn = LSTM
        self.model = Sequential()
        self.model.add(
            Bidirectional(rnn(rnn_dim,
                              dropout=dropout,
                              recurrent_dropout=dropout,
                              return_sequences=return_sequences),
                          input_shape=input_shape))
        # self.model.add(rnn(rnn_dim,
        #                    dropout=dropout,
        #                    recurrent_dropout=dropout,
        #                    return_sequences=return_sequences,
        #                    input_shape=input_shape))
        if highway:
            if return_sequences:
                self.model.add(TimeDistributed(Highway(activation='tanh')))
            else:
                self.model.add(Highway(activation='tanh'))

        if dense_dim > 0:
            self.model.add(TimeDistributed(Dense(dense_dim,
                                                 activation='relu')))
            self.model.add(TimeDistributed(Dropout(dropout)))
            self.model.add(TimeDistributed(BatchNormalization()))

    def __call__(self, inputs):
        return self.model(inputs)


class AttentionLayer(object):
    """Decomposable attention layer

    # References
        A Decomposable Attention Model for Natural Language Inference
    """
    def __init__(self, dense_dim, sequence_length=0,
                 input_dim=0, dropout=0.0):
        self.dense_dim = dense_dim
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        model = Sequential()
        model.add(Dense(dense_dim,
                        activation='relu',
                        input_shape=(input_dim,)))
        model.add(Dropout(dropout))
        model.add(BatchNormalization())
        self.model = TimeDistributed(model)

    def __call__(self, x1, x2):
        def _dot_product(args):
            x = args[0]
            y = args[1]
            return K.batch_dot(x, K.permute_dimensions(y, (0, 2, 1)))

        def _normalize(args, transpose=False):
            att_w = args[0]
            x = args[1]
            if transpose:
                att_w = K.permute_dimensions(att_w, (0, 2, 1))
            e = K.exp(att_w - K.max(att_w, axis=-1, keepdims=True))
            sum_e = K.sum(e, axis=-1, keepdims=True)
            nor_e = e / sum_e
            return K.batch_dot(nor_e, x)

        # (batch_size, timesteps1, dim)
        f1 = self.model(x1)
        # (batch_size, timesteps2, dim)
        f2 = self.model(x2)
        output_shape = (self.sequence_length, self.sequence_length,)
        # attention weights, (batch_size, timesteps1, timesteps2)
        att_w = Lambda(
            _dot_product,
            output_shape=output_shape)([f1, f2])
        output_shape = (self.sequence_length, self.input_dim,)
        # (batch_size, timesteps1, dim)
        att1 = Lambda(
            _normalize, arguments={'transpose': False},
            output_shape=output_shape)([att_w, x2])
        # (batch_size, timestep2, dim)
        att2 = Lambda(
            _normalize, arguments={'transpose': True},
            output_shape=output_shape)([att_w, x1])
        return att1, att2


class NNCompareLayer(object):
    """Compare operation using feed forward neural network with ReLu.

    # References
        A Decomposable Attention Model for Natural Language Inference

    """

    def __init__(self, dense_dim, sequence_length=0,
                 input_dim=0, dropout=0.0):
        model = Sequential()
        model.add(Dense(dense_dim,
                        activation='relu',
                        input_shape=(input_dim,)))
        model.add(Dropout(dropout))
        model.add(BatchNormalization())
        self.model = TimeDistributed(model, input_shape=(sequence_length, input_dim,))

    def __call__(self, inputs):
        x = self.model(inputs)
        avg_x = GlobalAveragePooling1D()(x)
        max_x = GlobalMaxPooling1D()(x)
        x = concatenate([avg_x, max_x])
        x = BatchNormalization()(x)
        return x


class SubMultCompareLayer(object):
    """Compare operation using subtraction and multiplication followed by an NN layer.

    # Input shape
        two tensors with shape: (batch_size, timesteps, dim)

    # Output shape
        (batch_size, timesteps, dim * 2)

    # References
        A compare-aggregate model for matching text sequences
    """

    def __init__(self, dense_dim, sequence_length=0, input_dim=0, dropout=0):
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.dense_dim = dense_dim
        model = Sequential()
        model.add(Dense(dense_dim,
                  activation='relu',
                  input_shape=(input_dim,)))
        model.add(Dropout(dropout))
        model.add(BatchNormalization())
        self.model = TimeDistributed(model, input_shape=(sequence_length, input_dim,))

    def __call__(self, x1, x2):
        def _sub_ops(args):
            x1 = args[0]
            x2 = args[1]
            x = K.abs(x1 - x2)
            return x

        def _mult_ops(args):
            x1 = args[0]
            x2 = args[1]
            return x1 * x2

        output_shape = (self.sequence_length, self.input_dim,)
        sub = Lambda(_sub_ops, output_shape=output_shape)([x1, x2])
        mult = Lambda(_mult_ops, output_shape=output_shape)([x1, x2])
        sub = self.model(sub)
        mult = self.model(mult)
        return concatenate([sub, mult])


class CNNAggregationLayer(object):
    """CNN aggregation layer.

    """
    def __init__(self, filters=64):
        self.filters = filters

    def __call__(self, inputs):
        x = Conv1D(self.filters, 3, activation='relu')(inputs)
        return GlobalMaxPooling1D()(x)


class PredictLayer(object):
    """Prediction layer.

    """
    def __init__(self, dense_dim, input_dim=0,
                 dropout=0.0):
        self.model = Sequential()
        self.model.add(Dense(dense_dim,
                             activation='relu',
                             input_shape=(input_dim,)))
        self.model.add(Dropout(dropout))
        self.model.add(BatchNormalization())
        self.model.add(Dense(1, activation='sigmoid'))

    def __call__(self, inputs):
        return self.model(inputs)
