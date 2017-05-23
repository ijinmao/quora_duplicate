# -*- coding: utf-8 -*-
"""Model graph of Decomposable Attention.

References:
    A Decomposable Attention Model for Natural Language Inference
"""
import keras.backend as K
from keras.layers import Input
from keras.models import Model
from keras.layers.merge import concatenate
from config import AttentionConfig, TrainConfig
from models.layers import (
    WordRepresLayer, CharRepresLayer, ContextLayer,
    AttentionLayer, NNCompareLayer, PredictLayer
)


def build_model(embedding_matrix, word_index, char_index):
    print('--- Building model...')
    # Params
    nb_words = min(TrainConfig.MAX_NB_WORDS, len(word_index)) + 1
    sequence_length = TrainConfig.MAX_SEQUENCE_LENGTH
    word_embedding_dim = TrainConfig.WORD_EMBEDDING_DIM
    rnn_unit = AttentionConfig.RNN_UNIT
    dropout = AttentionConfig.DROP_RATE
    context_rnn_dim = AttentionConfig.CONTEXT_LSTM_DIM
    dense_dim = AttentionConfig.DENSE_DIM
    if TrainConfig.USE_CHAR:
        nb_chars = min(TrainConfig.MAX_NB_CHARS, len(char_index)) + 1
        char_embedding_dim = TrainConfig.CHAR_EMBEDDING_DIM
        char_rnn_dim = TrainConfig.CHAR_LSTM_DIM
        nb_per_word = TrainConfig.MAX_CHAR_PER_WORD

    # Build words input
    w1 = Input(shape=(sequence_length,), dtype='int32')
    w2 = Input(shape=(sequence_length,), dtype='int32')
    if TrainConfig.USE_CHAR:
        c1 = Input(shape=(sequence_length, nb_per_word), dtype='int32')
        c2 = Input(shape=(sequence_length, nb_per_word), dtype='int32')

    # Build word representation layer
    word_layer = WordRepresLayer(
        sequence_length, nb_words, word_embedding_dim, embedding_matrix)
    w_res1 = word_layer(w1)
    w_res2 = word_layer(w2)

    # Build chars input
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

    # Build context representation layer, (batch_size, timesteps, context_rnn_dim * 2)
    context_layer = ContextLayer(
        context_rnn_dim, rnn_unit=rnn_unit, dropout=dropout,
        input_shape=(sequence_length, K.int_shape(sequence1)[-1],),
        return_sequences=True)
    context1 = context_layer(sequence1)
    context2 = context_layer(sequence2)

    # Build attention layer, (batch_size, timesteps, dense_dim)
    att_layer = AttentionLayer(dense_dim,
                               sequence_length=sequence_length,
                               input_dim=K.int_shape(context1)[-1],
                               dropout=dropout)
    # attention1, (batch_size, timesteps1, dim)
    # attention2, (batch_size, timesteps2, dim)
    attention1, attention2 = att_layer(context1, context2)

    # # Build compare layer
    aggregation1 = concatenate([context1, attention1])
    aggregation2 = concatenate([context2, attention2])
    compare_layer = NNCompareLayer(dense_dim,
                                   sequence_length=sequence_length,
                                   input_dim=K.int_shape(aggregation1)[-1],
                                   dropout=dropout)
    compare1 = compare_layer(aggregation1)
    compare2 = compare_layer(aggregation2)

    final_repres = concatenate([compare1, compare2])

    # Build predition layer
    pred = PredictLayer(dense_dim,
                        input_dim=K.int_shape(final_repres)[-1],
                        dropout=dropout)(final_repres)

    # Build model
    if TrainConfig.USE_CHAR:
        inputs = (w1, w2, c1, c2)
    else:
        inputs = (w1, w2)
    model = Model(inputs=inputs,
                  outputs=pred)
    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    return model
