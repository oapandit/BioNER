'''
Laods all of data for bio-ner training task.
'''

from __future__ import print_function
import pickle
import numpy as np
import os


class data:
    def __init__(self, RAW_DATA_PATH, PROCESSED_DATA_PATH, WORD_VEC_FILE, CHAR_VEC_FILE, POS_VEC_FILE,
                 ADD_FEAT_VEC_FILE, CLASS_VEC_FILE, ADD_FEAT_FILE_SUBSTRING, CHAR_ONE_HOT_FILE_SUBSTRING,
                 WORD_CONTEXT_LENGTH,NB_UNIQUE_CHARS, CHAR_VECTOR_SIZE, WORD_VECTOR_SIZE, NB_TREE_CLASSES, WORD_LENGTH,
                 ADD_FEAT_VEC_SIZE,LIMITED_ADD_FEAT_VEC_SIZE, POS_TAG_VECTOR_SIZE):
        self.RAW_DATA_PATH = RAW_DATA_PATH
        self.PROCESSED_DATA_PATH = PROCESSED_DATA_PATH

        self.WORD_VEC_FILE = WORD_VEC_FILE
        self.CHAR_VEC_FILE = CHAR_VEC_FILE
        self.POS_VEC_FILE = POS_VEC_FILE
        self.ADD_FEAT_VEC_FILE = ADD_FEAT_VEC_FILE
        self.CLASS_VEC_FILE = CLASS_VEC_FILE

        self.ADD_FEAT_FILE_SUBSTRING = ADD_FEAT_FILE_SUBSTRING
        self.CHAR_ONE_HOT_FILE_SUBSTRING = CHAR_ONE_HOT_FILE_SUBSTRING
        self.WORD_CONTEXT_LENGTH = WORD_CONTEXT_LENGTH
        self.NB_UNIQUE_CHARS = NB_UNIQUE_CHARS
        self.CHAR_VECTOR_SIZE = CHAR_VECTOR_SIZE
        self.WORD_VECTOR_SIZE = WORD_VECTOR_SIZE
        self.NB_TREE_CLASSES = NB_TREE_CLASSES
        self.WORD_LENGTH = WORD_LENGTH
        self.ADD_FEAT_VEC_SIZE = ADD_FEAT_VEC_SIZE
        self.LIMITED_ADD_FEAT_VEC_SIZE = LIMITED_ADD_FEAT_VEC_SIZE
        self.POS_TAG_VECTOR_SIZE = POS_TAG_VECTOR_SIZE


    '''
    Searches files with given substring. Loads data from all those files.
    Data is concatenated and reshaped as given by reshape_size and returned.
    '''

    def read_parts_from_file(self, file_name_substring, reshape_size):
        filelist = []
        for root, dirs, files in os.walk(self.PROCESSED_DATA_PATH):
            for filen in files:
                if filen.startswith(file_name_substring):
                    filelist.append(filen)
        f_number = 1
        filelist_in_order = []
        for f in filelist:
            fname = file_name_substring + "_" + str(f_number)
            filelist_in_order.append(fname)
            f_number += 1
        complete_array = np.array([])
        first_access = True
        for fname in filelist_in_order:
            # print(fname)
            f = open(os.path.join(self.PROCESSED_DATA_PATH, fname), 'r')
            complete_array_part = pickle.load(f)
            complete_array_part = complete_array_part.reshape(-1, reshape_size)
            if (first_access):
                complete_array = complete_array_part
                first_access = False
            else:
                complete_array = np.concatenate((complete_array, complete_array_part), axis=0)
        return complete_array

    '''
    Reads data from files and saves it in desired shape.
    Data is deivded with given deviding factor
    '''

    def get_data_from_file_and_reshape(self, f_name, word_context_length, vector_size, max_word_length=None,
                                       data_path=None):
        if max_word_length is None:
            max_word_length = self.WORD_LENGTH
        if data_path is None:
            data_path = self.PROCESSED_DATA_PATH
        if f_name.endswith(".txt"):  # Loading text files
            x = np.genfromtxt(os.path.join(data_path, f_name))
            x = np.array(x)
            x = x.reshape(-1, vector_size)
        elif f_name.endswith(".pickle"): # Loading pickle files
            f = open(os.path.join(data_path, f_name), 'r')
            x = pickle.load(f)
            x = x.reshape(-1, vector_size)
        else:   # Loading files in parts...
            if f_name == self.CHAR_ONE_HOT_FILE_SUBSTRING:  # character data is reshaped deifferently
                x = self.read_parts_from_file(f_name, max_word_length * vector_size)
            else:
                x = self.read_parts_from_file(f_name, vector_size)
        samples = int(x.shape[0] / word_context_length)
        if f_name == self.CHAR_ONE_HOT_FILE_SUBSTRING:  # character data is reshaped to 4d array
            x = x[0:samples * word_context_length].reshape(samples, word_context_length, max_word_length, vector_size)
        else:
            x = x[0:samples * word_context_length].reshape(samples, word_context_length, vector_size)
        return x

    def devide_train_test_data(self, x, devide_factor):
        training_size = int(x.shape[0] * devide_factor)
        x_train = x[0:training_size, :]
        x_test = x[training_size:, :]
        return x_train.astype('float32'), x_test.astype('float32')


    def print_data_shape_details(self, data_name, x1, x2=None):
        if x2 is None:
            print(data_name + " : shape : " + str(x1.shape))
        else:
            print(data_name + " : train shape : " + str(x1.shape))
            print(data_name + " : test shape : " + str(x2.shape))



    def get_train_test_data(self,char_one_hot=True, read_add_feat=False, read_pos=False, devide_factor=0.70):

        X_word, X_char, Y = self.get_data_without_devide(char_one_hot, read_add_feat, read_pos)

        X_word_train, X_word_test = self.devide_train_test_data(X_word, devide_factor)
        self.print_data_shape_details( "Word data", X_word_train, X_word_test)

        X_char_train, X_char_test = self.devide_train_test_data(X_char, devide_factor)
        self.print_data_shape_details( "Char data", X_char_train, X_char_test)

        Y_train, Y_test = self.devide_train_test_data(Y, devide_factor)
        self.print_data_shape_details( "Class data",Y_train, Y_test)

        return X_word_train, X_word_test, X_char_train, X_char_test, Y_train, Y_test

    def get_data_without_devide(self, char_one_hot=True, read_add_feat=False, read_pos=False):

        print('Loading word data...')
        X_word = self.get_data_from_file_and_reshape(self.WORD_VEC_FILE, self.WORD_CONTEXT_LENGTH,
                                                     self.WORD_VECTOR_SIZE)

        if char_one_hot:
            print('Loading one hot character data...')
            X_char = self.get_data_from_file_and_reshape(self.CHAR_ONE_HOT_FILE_SUBSTRING, self.WORD_CONTEXT_LENGTH,
                                                         self.NB_UNIQUE_CHARS)
        else:
            print('Loading numeric char data...')
            X_char = self.get_data_from_file_and_reshape(self.CHAR_VEC_FILE, self.WORD_CONTEXT_LENGTH,
                                                         self.CHAR_VECTOR_SIZE)

        if read_add_feat:
            print('Loading additional feature data...')
            X_add_feat = self.get_data_from_file_and_reshape(self.ADD_FEAT_FILE_SUBSTRING, self.WORD_CONTEXT_LENGTH,
                                                             self.ADD_FEAT_VEC_SIZE)
            X_word = np.concatenate((X_word, X_add_feat), axis=2)

        if read_pos:
            print('Loading POS data...')
            X_pos = self.get_data_from_file_and_reshape(self.POS_VEC_FILE, self.WORD_CONTEXT_LENGTH,
                                                        self.POS_TAG_VECTOR_SIZE)
            X_word = np.concatenate((X_word, X_pos), axis=2)

        print('Loading tree data...')
        Y = self.get_data_from_file_and_reshape(self.CLASS_VEC_FILE, self.WORD_CONTEXT_LENGTH, self.NB_TREE_CLASSES)

        self.print_data_shape_details("Char data", X_char)
        self.print_data_shape_details("Word data", X_word)
        self.print_data_shape_details("Class data", Y)
        return X_word, X_char, Y

    if __name__ == "__main__":
        get_data_without_devide()
