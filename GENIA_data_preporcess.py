from __future__ import print_function
import os
import numpy as np
from gensim.models import Word2Vec
import nltk
from nltk.corpus import gutenberg
##from nltk.tokenize.punkt import sentences_from_tokens
import collections
import time
import operator
from collections import OrderedDict
import re
import pickle
from string import ascii_lowercase

vec_random = np.random.rand(1, 200)

TRAIN_data_file = "Genia4ERtask1.iob2"
TRAIN_data_words_file = "Genia4ERtask1_word_length_normalized_x.txt"
TRAIN_data_tag_file = "Genia4ERtask1_word_length_normalized_y.txt"
TRAIN_word_vec_file_substring = "TRAIN_word_vec"
TRAIN_char_vec_file_substring = "TRAIN_word_char_one_hot_encoded"
TRAIN_add_feat_vec_file_substring = "TRAIN_add_feature_vec"
TRAIN_entity_vec_file_substring = "TRAIN_entitys_vec"


TEST_data_file = "Genia4EReval1.iob2"
TEST_data_words_file = "Genia4EReval1_word_length_normalized_x.txt"
TEST_data_tag_file = "Genia4EReval1_word_length_normalized_y.txt"

TEST_word_vec_file_substring = "TEST_word_vec"
TEST_char_vec_file_substring = "TEST_word_char_one_hot_encoded"
TEST_add_feat_vec_file_substring = "TEST_add_feature_vec"
TEST_entity_vec_file_substring = "TEST_entitys_vec"

tag_dict_file = "ENTITY_TYPE_DICT.pickle"
processed_data_path = "../processed_data"
word2vec_model_path = "../../../../krishanu/"
word2vec_model_name = "pubmed_Bio_Nlp_word2vec.bin"
devide_files_factor = 2000
raw_data_path = "../raw_data"


###### Helper functions ==>
def get_file_and_reset_array(file_substring,sr_num,all_vec_array):
    sr_num +=1
    f_name=file_substring+"_"+str(sr_num)
    dump_to_file = open(os.path.join(processed_data_path,f_name),"wb")
    pickle.dump(all_vec_array, dump_to_file)
    dump_to_file.close()
    print('dumped arrays to file : ',f_name)
    return np.array([]),sr_num

def get_char_vec_for_word(word, character_dict, max_vectors_in_word):
    word_length = 0
    char_array = np.array([])
    char_num_array = np.array([])
    vec_dim = len(character_dict.keys())
    for ch in word:
        word_length = word_length + 1
        char_vec = np.zeros(vec_dim)
        if (word_length > max_vectors_in_word):
            break
        if ch.isdigit():
            ch = '1'
        elif not ch.isalpha():
            ch = '_'
        vec = int(character_dict[ch]) - 1
        char_vec[vec] = 1
        char_num_array = np.append(char_num_array, int(character_dict[ch]))
        char_array = np.append(char_array, char_vec)
    if (word_length < max_vectors_in_word):
        char_vec = np.zeros(vec_dim)
        while (word_length != max_vectors_in_word):
            char_num_array = np.append(char_num_array, 0)
            char_array = np.append(char_array, char_vec)
            word_length = word_length + 1
    return (char_num_array, char_array)

def get_unique_tag_dict(y_file,tag_dict_file,load_from_file = False):
    if not load_from_file:
        print("Dictionary does not exist. Creating ....")
        with open(os.path.join(raw_data_path,y_file)) as f:
            content = f.readlines()
        i = 0
        worddict = {}
        k = 0
        while i < len(content):
            if worddict.get(content[i].strip("\n"), 0) == 0:
                worddict[content[i].strip("\n")] = k
                k = k + 1
            i = i + 1
        with open(os.path.join(processed_data_path,tag_dict_file),'wb') as handle:
            pickle.dump(worddict, handle)
    else:
        print("Dictionary exists. Loading ....")
        with open(os.path.join(processed_data_path,tag_dict_file), 'rb') as handle:
            worddict = pickle.load(handle)
    for key, value in worddict.iteritems():
        print(key + "  --->  " + str(value))
    return worddict


def get_vec_for_word(model, word):
    try:
        vec = model[word]
        return vec
    except:
        return vec_random

def get_add_feat_of_word(word):
    featureVector = np.zeros(10)
    if (word.isdigit()):
        featureVector[0] = 1
    if (word.isalpha()):
        featureVector[1] = 1
    if (word.isupper()):
        featureVector[2] = 1
    if (word.istitle()):
        featureVector[3] = 1
    if (len(word) > 25):
        featureVector[4] = 1
    elif (len(word) > 20):
        featureVector[5] = 1
    elif (len(word) > 15):
        featureVector[6] = 1
    elif (len(word) > 10):
        featureVector[7] = 1
    elif (len(word) > 5):
        featureVector[8] = 1
    elif (len(word) > 0):
        featureVector[9] = 1
    return (featureVector)

#############Acutal Functions ===>

def separate_word_tag(search_for_files=True,file=None):
    filelist = []
    if search_for_files:
        for root, dirs, files in os.walk("."):
            for filen in files:
                if filen.endswith(".iob2"):
                    filelist.append(filen)
    else:
        filelist.append(file)
    for fname in filelist:
        x_file = fname[:-5] + '_word_length_normalized_x.txt'
        y_file = fname[:-5] + '_word_length_normalized_y.txt'
        with open(os.path.join(raw_data_path,fname)) as f:
            content = f.readlines()
        i = 0
        dataFile = open(os.path.join(raw_data_path,x_file), "w")
        dataFile1 = open(os.path.join(raw_data_path,y_file), "w")
        while i < len(content):
            while content[i] != '\n':
                if "MEDLINE" in content[i]:
                    break
                x = content[i].split('\t')[0].strip()
                y = content[i].split('\t')[1].strip()
                if (len(x) > 30):
                    print(x)
                    print(i)
                    list_of_words = re.split(r'([-/])+', x)
                    l = 0
                    while l < len(list_of_words):
                        if (len(list_of_words[l]) > 0):
                            dataFile.write(list_of_words[l] + "\n")
                            if (l == 0):
                                dataFile1.write(y + "\n")
                            else:
                                if (y == "O"):
                                    dataFile1.write(y + "\n")
                                else:
                                    dataFile1.write("I-" + y[2:] + "\n")
                        l = l + 1
                else:
                    dataFile.write(x + "\n")
                    dataFile1.write(y + "\n")
                i = i + 1
            i = i + 1



def generate_word_tag_same_as_output(search_for_files=True, file=None):
    filelist = []
    if search_for_files:
        for root, dirs, files in os.walk("."):
            for filen in files:
                if filen.endswith(".iob2"):
                    filelist.append(filen)
    else:
        filelist.append(file)
    for fname in filelist:
        x_file = fname[:-5] + '_word_length_normalized_x_y.txt'
        with open(os.path.join(raw_data_path, fname)) as f:
            content = f.readlines()
        i = 0
        dataFile = open(os.path.join(raw_data_path, x_file), "w")
        while i < len(content):
            while content[i] != '\n':
                if "MEDLINE" in content[i]:
                    break
                x = content[i].split('\t')[0].strip()
                y = content[i].split('\t')[1].strip()
                if (len(x) > 30):
                    # print(x)
                    # print(i)
                    list_of_words = re.split(r'([-/])+', x)
                    l = 0
                    while l < len(list_of_words):
                        if (len(list_of_words[l]) > 0):
                            if (l == 0):
                                dataFile.write(list_of_words[l] +"\t"+y + "\n")
                            else:
                                if (y == "O"):
                                    dataFile.write(list_of_words[l] +"\t"+y + "\n")
                                else:
                                    dataFile.write(list_of_words[l] +"\t"+"I-" + y[2:] + "\n")
                        l = l + 1
                else:
                    dataFile.write(x +"\t"+y+"\n")
                i = i + 1
            i = i + 1


def generate_word_vectors(words_file):
    model = Word2Vec.load_word2vec_format(os.path.join(word2vec_model_path,word2vec_model_name), binary=True)
    with open(os.path.join(raw_data_path,words_file)) as f:
        content = f.readlines()
    i = 0
    all_vec_array = np.array([])
    k=0
    while i < len(content):
        word = content[i].strip("\n")
        vec = get_vec_for_word(model, word)
        all_vec_array = np.append(all_vec_array, vec)
        if (i % devide_files_factor == 0 and i != 0 or i == len(content) - 1):
            all_vec_array,k = get_file_and_reset_array(TEST_word_vec_file_substring,k,all_vec_array)
        i = i + 1

def generate_enity_one_hot_vectors(y_file,tag_dict_file,vec_file_substring):
    with open(os.path.join(raw_data_path,y_file)) as f:
        content = f.readlines()
    laod_dict_from_file = True
    worddict = get_unique_tag_dict(y_file,tag_dict_file,laod_dict_from_file)
    vec_dim =  len(worddict.keys())
    i = 0
    k=0
    all_vec_array = np.array([])
    while i < len(content):
        vec = np.zeros(vec_dim)
        vec[worddict[content[i].strip("\n")] - 1] = 1
        all_vec_array = np.append(all_vec_array, vec)
        if (i % devide_files_factor == 0 and i != 0 or i == len(content) - 1):
            all_vec_array,k = get_file_and_reset_array(vec_file_substring,k,all_vec_array)
        i = i + 1

def generate_add_feat_one_hot_vectors(x_file,vec_file_substring):
    with open(os.path.join(raw_data_path,x_file)) as f:
        content = f.readlines()
    i = 0
    k = 0
    all_vec_array = np.array([])
    while i < len(content):
        word = content[i].strip("\n")
        vec = get_add_feat_of_word(word)
        all_vec_array = np.append(all_vec_array, vec)
        if (i % devide_files_factor == 0 and i != 0 or i == len(content) - 1):
            all_vec_array,k = get_file_and_reset_array(vec_file_substring,k,all_vec_array)
        i = i + 1

def generate_char_one_hot_vectors(x_file,vec_file_substring):
    i = 1
    character_dict = {}
    for c in ascii_lowercase:
        character_dict[c] = i
        i = i + 1
    character_dict['1'] = i
    character_dict['_'] = i + 1
    max_vectors_in_word = 30
    all_vec_array = np.array([])
    i = 0
    k = 0
    print("Number of unique chars ", len(character_dict.keys()))
    for key, value in character_dict.iteritems():
        print(repr(key) + "  --->  " + str(value))
    with open(os.path.join(raw_data_path,x_file)) as f:
        content = f.readlines()
    while i < len(content):
        word = content[i].strip()
        char_num_array, char_array = get_char_vec_for_word(word.lower(), character_dict, max_vectors_in_word)
        all_vec_array = np.append(all_vec_array, char_array)
        if (i % devide_files_factor == 0 and i != 0 or i == len(content) - 1):
            all_vec_array,k = get_file_and_reset_array(vec_file_substring,k,all_vec_array)
        i = i + 1


def getPosTagsForGeniaData(GENIA_gtag_file,GENIA_words_file,pos_tag_file):
    with open(GENIA_gtag_file) as f:
        pos_tag_content = f.readlines()
    with open(GENIA_words_file) as f1:
        word_content = f1.readlines()
    dataFile = open(pos_tag_file,"w")
    dataFile2 = open("diff_file_x.txt","w")
    i=0
    l=len(pos_tag_content)
    k=0
    while k<len(word_content):
        while i<l:
            word = pos_tag_content[i].strip('\n').split("\t")
	    if(len(word)<5):
                i=i+1
                continue
            i=i+1
            actualWord = str(word[2])
            wordPosTag = str(word[4])
            wordChunk = str(word[5])
            word_from_genia_file = word_content[k].strip('\n')
            if(word_from_genia_file==actualWord):
                dataFile.write(word_from_genia_file+"\t"+wordPosTag+"\t"+wordChunk+"\n")
                break
            if(actualWord in word_from_genia_file):
                dataFile.write(word_from_genia_file+"\t"+wordPosTag+"\t"+wordChunk+"\n")
                dataFile2.write(word_from_genia_file+"\t"+wordPosTag+"\t"+actualWord+"\n")
                break
        k=k+1
    dataFile.close()
    dataFile2.close()

if __name__ == "__main__":
    generate_word_tag_same_as_output(False,TEST_data_file)
    # separate_word_tag(False,TEST_data_file)
    # generate_word_vectors(TEST_data_words_file)
    # generate_add_feat_one_hot_vectors(TEST_data_words_file,TEST_add_feat_vec_file_substring)
    # generate_char_one_hot_vectors(TEST_data_words_file,TEST_char_vec_file_substring)
    # get_unique_tag_dict(TRAIN_data_tag_file,tag_dict_file)
    # generate_enity_one_hot_vectors(TEST_data_tag_file,tag_dict_file,TEST_entity_vec_file_substring)