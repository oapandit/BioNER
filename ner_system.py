
from data import data
from model import model
from report import report
import os

RAW_DATA_PATH = "../raw_data"
PROCESSED_DATA_PATH = "../processed_data"
MODELS_PATH = "../models"
REPORT_PATH = "../reports"

WORD_VEC_FILE = "word_vectors_file_word_length_normalized.pickle"
CHAR_VEC_FILE = "char_vectors_file_word_length_normalized.txt"
POS_VEC_FILE = "genia_text_data_POS_Tag_vectors.pickle"
ADD_FEAT_VEC_FILE = "limited_feature_vec_len_normalized.pickle"
CLASS_VEC_FILE = "tag_vec_file_word_normalized.txt"

TEST_word_vec_file_substring = "TEST_word_vec"
TEST_char_vec_file_substring = "TEST_word_char_one_hot_encoded"
TEST_add_feat_vec_file_substring = "TEST_add_feature_vec"
TEST_entity_vec_file_substring = "TEST_entitys_vec"

ADD_FEAT_FILE_SUBSTRING = "add_feature_vec"
CHAR_ONE_HOT_FILE_SUBSTRING = "word_char_one_hot_encoded"

WORD_CONTEXT_LENGTH = 30
NB_UNIQUE_CHARS = 28
CHAR_VECTOR_SIZE = 30
WORD_VECTOR_SIZE = 200
NB_CLASSES = 11
MAX_WORD_LENGTH = 30
ADD_FEAT_VEC_SIZE = 10
LIMITED_ADD_FEAT_VEC_SIZE = 4
POS_TAG_VECTOR_SIZE = 47

CHAR_FEATURE_OUTPUT = 64
HIDDEN_SIZE = 64
NB_EPOCH = 50
BATCH_SIZE = 1
EMBEDDING_OP_DIM = 10
MAX_FEATURES = 31


bio_ner_train_data = None
bio_ner_test_data = None
bio_ner_model = None
bio_ner_report = None


def param_init(log_filename):

    global bio_ner_train_data, bio_ner_test_data,bio_ner_model, bio_ner_report

    bio_ner_train_data = data(RAW_DATA_PATH, PROCESSED_DATA_PATH, WORD_VEC_FILE, CHAR_VEC_FILE, POS_VEC_FILE,
                        ADD_FEAT_VEC_FILE, CLASS_VEC_FILE, ADD_FEAT_FILE_SUBSTRING, CHAR_ONE_HOT_FILE_SUBSTRING,
                        WORD_CONTEXT_LENGTH, NB_UNIQUE_CHARS, CHAR_VECTOR_SIZE, WORD_VECTOR_SIZE, NB_CLASSES,
                        MAX_WORD_LENGTH,
                        ADD_FEAT_VEC_SIZE, LIMITED_ADD_FEAT_VEC_SIZE, POS_TAG_VECTOR_SIZE)

    bio_ner_test_data = data(RAW_DATA_PATH, PROCESSED_DATA_PATH, TEST_word_vec_file_substring, CHAR_VEC_FILE, POS_VEC_FILE,
                        ADD_FEAT_VEC_FILE, TEST_entity_vec_file_substring, TEST_add_feat_vec_file_substring, TEST_char_vec_file_substring,
                        WORD_CONTEXT_LENGTH, NB_UNIQUE_CHARS, CHAR_VECTOR_SIZE, WORD_VECTOR_SIZE, NB_CLASSES,
                        MAX_WORD_LENGTH,
                        ADD_FEAT_VEC_SIZE, LIMITED_ADD_FEAT_VEC_SIZE, POS_TAG_VECTOR_SIZE)

    bio_ner_model = model(
        WORD_VECTOR_SIZE, WORD_CONTEXT_LENGTH, NB_UNIQUE_CHARS, CHAR_VECTOR_SIZE, MAX_WORD_LENGTH, EMBEDDING_OP_DIM,
        MAX_FEATURES,
        NB_CLASSES, CHAR_FEATURE_OUTPUT, HIDDEN_SIZE, ADD_FEAT_VEC_SIZE, POS_TAG_VECTOR_SIZE)

    bio_ner_report = report(REPORT_PATH, log_filename)

def load_train_save(model_file_name,train_model,X_word_train, X_word_test, X_char_train, X_char_test, Y_train, Y_test,fit=True):
    model_loaded = False
    if os.path.isfile(os.path.join(MODELS_PATH, model_file_name)):
        print("Model is already trained.", model_file_name)
        train_model.load_weights(os.path.join(MODELS_PATH, model_file_name))
        print("Model loaded.")
        model_loaded = True
    if not model_loaded and fit:
        print("Training the model", model_file_name)
        train_model.fit([X_char_train, X_word_train], Y_train, nb_epoch=NB_EPOCH, batch_size=BATCH_SIZE,
                        validation_data=([X_char_test, X_word_test], Y_test))
        print('training completed')
        train_model.save_weights(os.path.join(MODELS_PATH, model_file_name))
        print("Saved model to file : " + model_file_name)
    if model_loaded or fit:
        Y_pred = train_model.predict([X_char_test, X_word_test])
        bio_ner_report.get_report(Y_pred, Y_test, model_file_name)
        bio_ner_report.generate_final_output(Y_pred)
    else:
        print("Model neither loaded nor trained")



def optimier_exp():
    log_filename = "bio_ner_optimizers.log"
    param_init(log_filename)
    X_word_train, X_word_test, X_char_train, X_char_test, Y_train, Y_test = bio_ner_train_data.get_train_test_data()
    # nns = ["LSTM", "RNN", "GRU", "BRNN", "BGRU", "BLSTM"]
    nns = ["RNN", "GRU", "BRNN", "BGRU", "BLSTM"]
    optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta']
    for NN in nns:
        for opt in optimizers:
            model_file_name = "LSTM_" + NN + "_bs_" + str(BATCH_SIZE) + "_wcl_" + str(
                WORD_CONTEXT_LENGTH) + "_char_nuron_" + str(CHAR_FEATURE_OUTPUT) + "_er_nuron_" + str(
                HIDDEN_SIZE) + "_opt_" + opt + ".h5"
            train_model = bio_ner_model.get_model(NN, opt)
            load_train_save(model_file_name, train_model, X_word_train, X_word_test, X_char_train, X_char_test, Y_train,
                            Y_test) # if file exists, dont train again

def pos_exp():
    log_filename = "bio_ner_POS.log"
    param_init(log_filename)
    X_word_train, X_char_train, Y_train = [],[],[]
    X_word_test, X_char_test, Y_test = bio_ner_test_data.get_data_without_devide()
    # X_word_train, X_word_test, X_char_train, X_char_test, Y_train, Y_test = bio_ner_train_data.get_train_test_data(read_pos=True)
    # nns = ["LSTM", "RNN", "GRU", "BRNN", "BGRU", "BLSTM"]
    nns = ["BLSTM"]
    for NN in nns:
        model_file_name = "LSTM_" + NN + "_bs_" + str(BATCH_SIZE) + "_wcl_" + str(
            WORD_CONTEXT_LENGTH) + "_char_nuron_" + str(CHAR_FEATURE_OUTPUT) + "_er_nuron_" + str(
            HIDDEN_SIZE) + "_POS.h5"
        train_model = bio_ner_model.get_model(NN,pos_tag=True)
        load_train_save(model_file_name, train_model, X_word_train, X_word_test, X_char_train, X_char_test,
                        Y_train,Y_test,False)  # if file exists, don't train again


def add_pos_exp():
    log_filename = "bio_ner_add_feat_POS.log"
    param_init(log_filename)
    X_word_train, X_char_train, Y_train = [],[],[]
    X_word_test, X_char_test, Y_test = bio_ner_test_data.get_data_without_devide()
    # X_word_train, X_word_test, X_char_train, X_char_test, Y_train, Y_test = bio_ner_train_data.get_train_test_data(read_pos=True,read_add_feat=True)
    # nns = ["LSTM", "RNN", "GRU", "BRNN", "BGRU", "BLSTM"]
    nns = ["BLSTM"]
    for NN in nns:
        model_file_name = "LSTM_" + NN + "_bs_" + str(BATCH_SIZE) + "_wcl_" + str(
            WORD_CONTEXT_LENGTH) + "_char_nuron_" + str(CHAR_FEATURE_OUTPUT) + "_er_nuron_" + str(
            HIDDEN_SIZE) + "_POS_add_feat.h5"
        train_model = bio_ner_model.get_model(NN,pos_tag=True,add_feat=True)
        load_train_save(model_file_name, train_model, X_word_train, X_word_test, X_char_train, X_char_test,
                        Y_train,Y_test,False)  # if file exists don't train again

def train_test_complete():
    log_filename = "bio_ner_full_data_test_softplus.log"
    param_init(log_filename)
    X_word_train,X_char_train, Y_train = bio_ner_train_data.get_data_without_devide()
    X_word_test, X_char_test,Y_test = bio_ner_test_data.get_data_without_devide()
    nns = ["BLSTM" , "BGRU", "LSTM", "RNN", "GRU", "BRNN"]
    for NN in nns:
        train_model = bio_ner_model.get_model(NN,output_act ="softplus")
        model_file_name = "LSTM"+NN+"bs_1_wcl_30_char_nuron_64_er_nuron_64_softplus.h5"
        load_train_save(model_file_name, train_model, X_word_train, X_word_test, X_char_train, X_char_test, Y_train,
                        Y_test, fit=True)

def evaluate_system():
    log_filename = "system_evaluation.log"
    model_file_name = "LSTMBLSTMbs_1_wcl_30_char_nuron_64_er_nuron_64.h5"
    param_init(log_filename)
    X_word_test, X_char_test, Y_test = bio_ner_test_data.get_data_without_devide()
    # train_model = bio_ner_model.get_model()
    # train_model.load_weights(os.path.join(MODELS_PATH, model_file_name))
    # Y_pred = train_model.predict([X_char_test, X_word_test])
    # bio_ner_report.save_y_vector(Y_pred,"output_y.pkl")
    Y_pred = bio_ner_report.load_y_vector("output_y.pkl")
    # bio_ner_report.get_report(Y_pred, Y_test, model_file_name)
    bio_ner_report.generate_final_output(Y_pred)

if __name__ == "__main__":
    pos_exp()
    add_pos_exp()
    # optimier_exp()
    # train_test_complete()
    # evaluate_system()