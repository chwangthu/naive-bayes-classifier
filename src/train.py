import os
import re
import sys
import random
import config
import _pickle as pickle
from math import log, inf

def get_set(fold, num_folds=5):
    '''
    get trainset and testset from 5 folds, 
    args fold to specify the testset, the rest
    is trainset
    '''
    testset = []
    trainset = []
    for i in range(num_folds):
        tep_fold = []
        f =  open(config.fold_data_dir + "/" + str(i))
        lines = f.readlines()
        for line in lines:
            line = line[:-1]
            line = line.split(" ")
            tep_fold.append((line[0], line[1]))
        f.close()
        if i != fold:
            trainset.extend(tep_fold)
        else:
            testset = tep_fold
    
    return trainset, testset

def write_res(num_per_words, total_words):
    if not os.path.exists(config.train_data_dir):
        os.makedirs(config.train_data_dir)
    words_num = open(config.train_data_dir+"/num_per_words", 'wb')
    pickle.dump(num_per_words, words_num)

    total_words_file = open(config.train_data_dir+"/total_words", 'wb')
    pickle.dump(total_words, total_words_file)

def split_email(content):
    re_words = re.compile(u"[\u4e00-\u9fa5]+")
    return re.findall(re_words, content)

def train(trainset, train_per=1):
    # The nums of each word in spam or ham emails
    random.seed(config.train_seed)
    words_num = { "spam": {}, "ham": {}} # for each category, the num of every kinds of words
    total_words = { "spam": 0, "ham": 0} # total words for each category
    print("Total " + str(int(len(trainset)*train_per)) + " to train.")
    num_trained = 0
    spam_words = 0
    ham_words = 0
    for item in trainset: # item[0] is path, item[1] is label
        if (random.random() > train_per):
            continue
        try:
            f = open(item[0])
            content = f.read() # read as an string
            f.close()
        except:
            print("File " + item[0] + " not found")
            continue

        res = split_email(content)

        if item[1] == "spam":
            total_words["spam"] += len(res)
        else:
            total_words["ham"] += len(res)

        for word in res:
            if words_num[item[1]].__contains__(word):
                words_num[item[1]][word] += 1
            else:
                words_num[item[1]][word] = 1
        num_trained += 1
        # if(num_trained == 10):
        #     break
        if num_trained % 5000 == 0:
            print("Training process: " + str(num_trained) + "/" + str(len(trainset)))
    print("Training finished.")
    # print(total_words)
    write_res(words_num, total_words)
    return words_num, total_words