#-*- coding:utf-8 -*-
import os
import re
import sys
import random
import json
import config
import _pickle as pickle
from math import log, inf
from spamfilter import BayesSpamFilter
from parser import get_label, divide_into_folds
from train import get_set, train
# The percent of trainset to use
train_per = [0.05, 0.5, 1]

if __name__ == "__main__":

    acc = {}
    label_dic = get_label()
    for seed in config.fold_seed:
        divide_into_folds(label_dic, seed)
        print("Divide finished.")
        tep_acc = {}
        for i in range(5):
            for train_per in config.train_per:
                print("--------------------------------------------")
                print("Using fold: %d , training percent: %2d%%" %(i, train_per*100))
                (trainset, testset) = get_set(i)
                (words_num, total_words) = train(trainset, train_per)
            # with open("../train_data/num_per_words", 'rb') as f:
            #     words_num = pickle.load(f)
            # with open("../train_data/total_words", 'rb') as f:
            #     total_words = pickle.load(f)    
            
                spam_filter = BayesSpamFilter(words_num, total_words, testset)
                acc_fold = spam_filter.classify()
                # sum += acc_fold
                if tep_acc.__contains__(str(train_per)):
                    tep_acc[str(train_per)] += acc_fold
                else:
                    tep_acc[str(train_per)] = acc_fold
                # print(tep_acc)
        print(tep_acc)
        for item in tep_acc.keys():
            tep_acc[item] = tep_acc[item] / 5
        print("Acc for seed " + str(seed))
        print(tep_acc)
        acc[str(seed)] = tep_acc
        print(acc)
    # print("avr_acc= " + str(sum/5))