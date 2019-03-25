#-*- coding:utf-8 -*-
import os
import config
import random
import _pickle as pickle

def get_label():
    f = open(config.label_file)
    lines = f.readlines()
    f.close()

    label_dic = {}
    for line in lines:
        line = line[:-1] # remove line escape
        line = line.split(' ')
        label = line[0]
        file = config.cut_data_dir + line[1][7:] 
        label_dic[file] = label
        # print(label_dic[label])
    # print(label_dic)
    return label_dic

def divide_into_folds(label_dic, seed=1999, num_folds=5):
    dataset = []
    random.seed(seed)
    for _ in range(num_folds):
        dataset.append([])
    for item in label_dic:
        # print(item)
        t = random.randint(0, 4)
        dataset[t].append(item + " " + label_dic[item] + "\n")
    if not os.path.exists(config.fold_data_dir):
        os.makedirs(config.fold_data_dir)
    for i in range(num_folds):
        f = open(config.fold_data_dir + "/" + str(i), 'w')
        f.writelines(dataset[i])
        f.close()


if __name__ == "__main__":
    label_dic = get_label()
    divide_into_folds(label_dic)
    pass
