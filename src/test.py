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

def change_trainset():
    acc = {}
    prec = {}
    recall = {}
    label_dic = get_label()
    for seed in config.fold_seed:
        divide_into_folds(label_dic, seed)
        print("Divide finished.")
        (tep_acc, tep_prec, tep_recall) = ({},{},{})
        for i in range(5):
            for train_per in config.train_per:
                print("--------------------------------------------")
                print("Using fold: %d , training percent: %2d%%" %(i, train_per*100))
                (trainset, testset) = get_set(i)
                (words_num, total_words) = train(trainset, train_per)
                spam_filter = BayesSpamFilter(words_num, total_words, testset)
                acc_fold, prec_fold, recall_fold, f1 = spam_filter.classify()
                # sum += acc_fold
                if tep_acc.__contains__(str(train_per)):
                    tep_acc[str(train_per)] += acc_fold
                else:
                    tep_acc[str(train_per)] = acc_fold
                
                if tep_prec.__contains__(str(train_per)):
                    tep_prec[str(train_per)] += prec_fold
                else:
                    tep_prec[str(train_per)] = prec_fold
                
                if tep_recall.__contains__(str(train_per)):
                    tep_recall[str(train_per)] += recall_fold
                else:
                    tep_recall[str(train_per)] = recall_fold
                # print(tep_acc)
        print(tep_acc)
        for a, b, c in zip(tep_acc.keys(), tep_prec.keys(), tep_recall.keys()):
            tep_acc[a] /= 5
            tep_prec[b] /= 5
            tep_recall[c] /= 5
        # for item in tep_acc.keys():
        #     tep_acc[item] = tep_acc[item] / 5
        # for item in tep_prec.keys():
        #     tep_prec[item] = tep_prec[item] / 5
        # for item in tep_recall.keys():
        #     tep_recall[item] = tep_recall[item] / 5

        print(tep_acc)
        print(tep_prec)
        print(tep_recall)

        acc[str(seed)] = tep_acc
        prec[str(seed)] = tep_prec
        recall[str(seed)] = tep_recall

        print("acc", acc)
        print("precision", prec)
        print("recall", recall)
        (aver_acc, aver_recall, aver_prec, aver_f1) = ({},{},{},{})
        for item in train_per:
            item = str(item)
            (tep_acc, tep_recall, tep_prec) = (0, 0, 0)
            for seed in config.fold_seed:
                seed = str(seed)
                tep_acc += acc[seed][item]
                tep_recall += recall[seed][item]
                tep_prec += prec[seed][item]
            (aver_acc[item], aver_recall[item], aver_prec[item]) = (tep_acc/5, tep_recall/5, tep_prec/5)
            aver_f1[item] = 2*aver_prec[item]*aver_recall[item]/(aver_prec[item]+aver_recall[item])
        print("acc", aver_acc)
        print("prec", aver_prec)
        print("recall", aver_recall)
        print("f1", aver_f1)

def change_lambda():
    lam = 1e-100
    lam_list = []
    res_list = []
    for _ in range(20):
        label_dic = get_label()
        divide_into_folds(label_dic)
        (acc, prec, recall, f1) = (0, 0, 0, 0)
        for i in range(5):
           (trainset, testset) = get_set(i)
           (words_num, total_words) = train(trainset) 
           spam_filter = BayesSpamFilter(words_num, total_words, testset, lam=lam)
           acc_fold, prec_fold, recall_fold, f1_fold = spam_filter.classify()
           acc += acc_fold
           prec += prec_fold
           recall += recall_fold
           f1 += f1_fold
        #    print(acc, prec, recall, f1)
        lam_list.append(lam)
        res_list.append(acc/5)
        (acc, prec, recall, f1) = (acc/5*100, prec/5*100, recall/5*100, f1/5*100)
        print("lam="+str(lam), end=" ")
        print("acc=%.6f%%, prec=%.6f%%, recall=%.6f%%, f1=%.6f%%" %(acc, prec, recall, f1))
        print("-----------------")
        lam *= 1e10
    print(lam_list)
    print(res_list)

def add_features():
    label_dic = get_label()
    divide_into_folds(label_dic)
    (acc, prec, recall, f1) = (0, 0, 0, 0)
    for i in range(5):
        (trainset, testset) = get_set(i)
        (words_num, total_words, mailer_dict) = train(trainset=trainset, use_mailer=True, use_from=True)
        # print(mailer_dict) 
        spam_filter = BayesSpamFilter(x_num=words_num, total_words=total_words, testset=testset, lam=1e-50, use_mailer=False, mailer_dict=mailer_dict, use_from=True)
        acc_fold, prec_fold, recall_fold, f1_fold = spam_filter.classify()
        acc += acc_fold
        prec += prec_fold
        recall += recall_fold
        f1 += f1_fold
    (acc, prec, recall, f1) = (acc/5*100, prec/5*100, recall/5*100, f1/5*100)
    print("acc=%.6f%%, prec=%.6f%%, recall=%.6f%%, f1=%.6f%%" %(acc, prec, recall, f1))

if __name__ == "__main__":
    # issue1()
    # change_lambda()
    add_features()