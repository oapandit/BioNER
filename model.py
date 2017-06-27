from __future__ import print_function
import numpy as np
import sys

np.random.seed(1337)  # for reproducibility

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Embedding, merge, Reshape
from keras.layers import LSTM, SimpleRNN, GRU, TimeDistributedDense, TimeDistributed, Bidirectional


class model:
    '''
    This class is used to generate keras models.
    '''

    def __init__(self, word_vector_size, word_context_length,nb_unique_chars, char_vector_size, max_word_length, output_dim, max_features,
                 tag_vector_size, char_feature_output, hidden_size, add_feat_vec_size, pos_tag_vector_size):
        '''
        :param word_vector_size: Vector dimention of word2vec word embeddings e.g.200
        :param word_context_length: General sentence length. BLSTM will have context of these many words
        :param char_vector_size:  Number of unique chars in language, e.g. English - 26 lowercase letters + 1 for digit + 1 for special symbols
        :param max_word_length: Maximum length of word in data. In general character matrix dimention in the word.
        :param output_dim: Embedding layer output dimention
        :param tag_vector_size: Number of classes for classification.
        :param char_feature_output: Number of neurons in character feature extraction network.
        :param hidden_size: Number of neurons in final classification network.
        :param add_feat_vec_size: Vector dimention of additional features
        :param pos_tag_vector_size:  Vector dimention of pos tag.
        '''

        self.add_feat_vec_size = add_feat_vec_size
        self.pos_tag_vector_size = pos_tag_vector_size
        self.char_feature_output = char_feature_output
        self.nb_unique_chars = nb_unique_chars
        self.char_vector_size = char_vector_size
        self.hidden_size = hidden_size
        self.max_word_length = max_word_length
        self.output_dim = output_dim
        self.word_vector_size = word_vector_size
        self.tag_vector_size = tag_vector_size
        self.word_context_length = word_context_length
        self.max_features = max_features

    def get_model(self, nn="BLSTM", opt = "adam",embedding=False, add_feat=False, pos_tag=False, output_act = "softmax"):
        '''
        :param nn: Name of the Network with which classification is done. Default is BLSTM
        :param embedding: Input is char one-hot encoded or character numeric value. True if character in numeric value.
        :param add_feat: True if additional features are provided in input
        :param pos_tag: True if POS information is provided in input
        :return: model object is returned
        '''

        if embedding:
            char_input = Input(shape=(self.word_context_length, self.char_vector_size,), dtype='float32',
                               name='char_input')
            x = TimeDistributed(
                Embedding(self.max_features, self.output_dim, input_length=self.char_vector_size, dropout=0.2))(
                char_input)
        else:
            char_input = Input(shape=(self.word_context_length, self.max_word_length, self.nb_unique_chars,),
                               dtype='float32',
                               name='char_input')
            x = Dropout(0.2)(char_input)
        lstm_out = TimeDistributed(LSTM(self.char_feature_output, dropout_W=0.2, dropout_U=0.2))(x)

        if add_feat and pos_tag:
            word_feat_vector_size = self.word_vector_size + self.pos_tag_vector_size + self.add_feat_vec_size
        elif add_feat:
            word_feat_vector_size = self.word_vector_size + self.add_feat_vec_size
        elif pos_tag:
            word_feat_vector_size = self.word_vector_size + self.pos_tag_vector_size
        else:
            word_feat_vector_size = self.word_vector_size

        word_input = Input(shape=(self.word_context_length, word_feat_vector_size,), name='word_input')
        merged = merge([lstm_out, word_input], mode='concat', concat_axis=2)
        if (nn == "BLSTM"):
            x = Bidirectional(LSTM(self.hidden_size, return_sequences=True, dropout_W=0.2, dropout_U=0.2))(merged)
        elif (nn == "BGRU"):
            x = Bidirectional(GRU(self.hidden_size, return_sequences=True, dropout_W=0.2, dropout_U=0.2))(merged)
        elif (nn == "BRNN"):
            x = Bidirectional(SimpleRNN(self.hidden_size, return_sequences=True, dropout_W=0.2, dropout_U=0.2))(merged)
        elif (nn == "RNN"):
            x = SimpleRNN(self.hidden_size, return_sequences=True, dropout_W=0.2, dropout_U=0.2)(merged)
        elif (nn == "GRU"):
            x = GRU(self.hidden_size, return_sequences=True, dropout_W=0.2, dropout_U=0.2)(merged)
        elif (nn == "LSTM"):
            x = LSTM(self.hidden_size, return_sequences=True, dropout_W=0.2, dropout_U=0.2)(merged)
        else:
            print("Incorrect Neural network name passed. Please check")
            sys.exit(0)
        # Additional features are concatenated at the output of the BLSTM. This code is commented as we did not get good results after doing this.
        # if (add_feat):
        #     add_feat_input = Input(shape=(self.word_context_length, self.add_feat_vec_size,), name='add_feat_input')
        #     x = merge([x, add_feat_input], mode='concat', concat_axis=2)
        #     main_loss = TimeDistributed(Dense(self.tag_vector_size, activation='softmax'))(x)
        #     model = Model(input=[char_input, word_input, add_feat_input], output=[main_loss])
        # else:
        if output_act == "softplus":
            main_loss1 = TimeDistributed(Dense(self.tag_vector_size, activation='softplus'))(x)
            main_loss = Activation("softplus")(main_loss1)
        else:
            main_loss = TimeDistributed(Dense(self.tag_vector_size, activation='softmax'))(x)
        model = Model(input=[char_input, word_input], output=[main_loss])
        # optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam']
        # if opt == 'SGD':
        #     model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
        # elif opt == 'RMSprop':
        #     model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])
        # elif opt == 'Adagrad':
        #     model.compile(optimizer='Adagrad', loss='categorical_crossentropy', metrics=['accuracy'])
        # elif opt == 'Adadelta':
        #     model.compile(optimizer='Adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
        # elif opt == 'Adamax':
        #     model.compile(optimizer='Adamax', loss='categorical_crossentropy', metrics=['accuracy'])
        # else:
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())
        return model
