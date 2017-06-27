from __future__ import print_function
import numpy as np
import os
np.random.seed(1337)  # for reproducibility
from sklearn.metrics import classification_report
import logging

import pickle

import GENIA_data_preporcess as data_preproc

class report:
    type_tag_dict_file = "TYPE_ENTITY_DICT.pickle"


    def __init__(self,models_path,log_file_name):
        logging.basicConfig(filename=os.path.join(models_path,log_file_name), level=logging.DEBUG)

    def get_reshape_index(self,x):
        x = self.get_reshape(x)
        x = np.argmax(x, axis=1)
        return x


    def get_reshape(self,x):
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        return x

    def save_y_vector(self,Y_pred,fname):
        with open(os.path.join(data_preproc.processed_data_path,fname),'wb') as handle:
            pickle.dump(Y_pred, handle)

    def load_y_vector(self,fname):
        with open(os.path.join(data_preproc.processed_data_path,fname), 'rb') as handle:
            y_loaded = pickle.load(handle)
        print("Y loaded from file" + " : shape : " + str(y_loaded.shape))
        return y_loaded


    def get_report(self,Y_pred, Y_expected, m_file_name):

        logging.info(15 * "=====")
        logging.info("Model : " + m_file_name)

        print(classification_report(self.get_reshape_index(Y_expected),self.get_reshape_index(Y_pred),digits = 4))
        logging.info(classification_report(self.get_reshape_index(Y_expected),self.get_reshape_index(Y_pred),digits = 4))

        logging.info(15 * "=====")


    def generate_final_output(self,Y_pred):

        evaluation_script_path = data_preproc.raw_data_path
        gold_annotated_file = "Genia4EReval1_word_length_normalized_x_y.txt"
        output_file = data_preproc.TEST_data_file[:-5]+".ans"

        #
        # with open(os.path.join(data_preproc.processed_data_path, data_preproc.tag_dict_file), 'rb') as handle:
        #     worddict = pickle.load(handle)
        #
        # # for key, value in worddict.iteritems():
        # #     print(key + "  --->  " + str(value))
        #
        # rev_dict = dict((v, k) for k, v in worddict.items())
        #
        # # for key, value in rev_dict.iteritems():
        # #     print(str(key) + "  --->  " + str(value))
        #
        # with open(os.path.join(data_preproc.processed_data_path,report.type_tag_dict_file),'wb') as handle:
        #     pickle.dump(rev_dict, handle)

        with open(os.path.join(data_preproc.processed_data_path, report.type_tag_dict_file), 'wb') as handle:
            rev_dict = pickle.load(handle)

        output_fh = open(os.path.join(data_preproc.raw_data_path, output_file), "w")

        with open(os.path.join(data_preproc.raw_data_path,data_preproc.TEST_data_words_file)) as f:
            content = f.readlines()

        i=0
        Y_pred = self.get_reshape_index(Y_pred)
        print(len(content))
        print(Y_pred.shape[0])

        while i < Y_pred.shape[0]:
            if content[i] == '\n':
                i+=1
                continue
            word = content[i].split('\t')[0].strip()
            tag = rev_dict[Y_pred[i]+1]
            output_fh.write(word + "\t" +tag+"\n")
            i+=1

        output_fh.close()

        command = "cd " + evaluation_script_path
        os.system(command)
        command = "perl evalIOB2.pl " + gold_annotated_file + " " + output_file
        os.system(command)

