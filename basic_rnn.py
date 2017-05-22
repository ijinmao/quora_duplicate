# -*- coding: utf-8 -*-
"""Model graph of basic rnn model.

"""
import numpy as np
from keras.layers import Input
import keras.backend as K
from keras.layers.merge import concatenate
from keras.models import Model
from config import (
    BasicRnnConfig, TrainConfig
)
from layers import (
    WordRepresLayer, CharRepresLayer, ContextLayer,
    PredictLayer
)
np.random.seed(BasicRnnConfig.SEED)


def build_model(embedding_matrix, word_index, char_index):
    print('--- Building model...')
    # Params
    nb_words = min(TrainConfig.MAX_NB_WORDS, len(word_index)) + 1
    sequence_length = TrainConfig.MAX_SEQUENCE_LENGTH
    context_rnn_dim = BasicRnnConfig.RNN_DIM
    word_embedding_dim = TrainConfig.WORD_EMBEDDING_DIM
    rnn_unit = BasicRnnConfig.RNN_UNIT
    nb_per_word = TrainConfig.MAX_CHAR_PER_WORD
    dropout = BasicRnnConfig.DROP_RATE
    dense_dim = BasicRnnConfig.DENSE_DIM

    if TrainConfig.USE_CHAR:
        nb_chars = min(TrainConfig.MAX_NB_CHARS, len(char_index)) + 1
        char_embedding_dim = TrainConfig.CHAR_EMBEDDING_DIM
        char_rnn_dim = TrainConfig.CHAR_LSTM_DIM

    # define inputs
    w1 = Input(shape=(sequence_length,), dtype='int32')
    w2 = Input(shape=(sequence_length,), dtype='int32')
    if TrainConfig.USE_CHAR:
        c1 = Input(shape=(sequence_length, nb_per_word), dtype='int32')
        c2 = Input(shape=(sequence_length, nb_per_word), dtype='int32')

    # define word embedding representation
    word_layer = WordRepresLayer(
        sequence_length, nb_words, word_embedding_dim, embedding_matrix)
    w_res1 = word_layer(w1)
    w_res2 = word_layer(w2)

    # define char embedding representation
    if TrainConfig.USE_CHAR:
        char_layer = CharRepresLayer(
            sequence_length, nb_chars, nb_per_word, char_embedding_dim,
            char_rnn_dim, rnn_unit=rnn_unit, dropout=dropout)
        c_res1 = char_layer(c1)
        c_res2 = char_layer(c2)
        sequence1 = concatenate([w_res1, c_res1])
        sequence2 = concatenate([w_res2, c_res2])
    else:
        sequence1 = w_res1
        sequence2 = w_res2

    # define stack lstm layers
    for i in range(BasicRnnConfig.RNN_DIM_LAYER):
        if i == BasicRnnConfig.RNN_DIM_LAYER - 1:
            return_q = False
        else:
            return_q = True
        context_layer = ContextLayer(
            context_rnn_dim, rnn_unit=rnn_unit, dropout=dropout,
            input_shape=(sequence_length, K.int_shape(sequence1)[-1],),
            return_sequences=return_q)
        context1 = context_layer(sequence1)
        context2 = context_layer(sequence2)
        sequence1 = context1
        sequence2 = context2

    final_repres = concatenate([sequence1, sequence2])

    # Build predition layer
    preds = PredictLayer(dense_dim,
                         input_dim=K.int_shape(final_repres)[-1],
                         dropout=dropout)(final_repres)

    if TrainConfig.USE_CHAR:
        inputs = [w1, w2, c1, c2]
    else:
        inputs = [w1, w2]

    # Build model graph
    model = Model(inputs=inputs, outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    return model
