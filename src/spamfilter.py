#-*- coding:utf-8 -*-
import sys
from abc import abstractmethod, ABCMeta
from math import log
import re

class BayesClassifier():
    __metaclass__ = ABCMeta
    def __init__(self, hypothesis, laplace=True, lam=1.0, min_prob=1e-50):
        self.hypothesis = hypothesis
        self.laplace = laplace
        self.lam = lam
        pass

    @abstractmethod
    def cal_py(self):
        pass

    @abstractmethod
    def cal_p_xi_y(self):
        pass

    @abstractmethod
    def classify(self):
        pass

class BayesSpamFilter(BayesClassifier):
    def __init__(self, x_num, total_words, testset, hypothesis=["spam", "ham"], laplace=True, lam=1.0, min_prob=1e-50):
        BayesClassifier.__init__(self, hypothesis, laplace, lam, min_prob)
        self.x_num = x_num
        self.total_words = total_words
        self.testset = testset
        self.num_words = 0
        for item in total_words.keys():
            self.num_words += total_words[item]
    
    def classify(self):
        num_tested = 0
        num_correct = 0
        print("Total " + str(len(self.testset)) + " to test")
        for line in self.testset:
            try:
                f = open(line[0])
                content = f.read() # read as an string
                f.close()
            except:
                print("Test file " + line[0] + " not found")
                continue
            res = self.split_email(content)
            pred = self.hypothesis[0]
            max_p = float('-Inf')
            for y in self.hypothesis:
                prob = self.cal_py(y)
                # print(prob)
                for item in res:
                    prob += self.cal_p_xi_y(item, y)

                if prob > max_p:
                    # print(h)
                    max_p = prob
                    pred = y

            num_tested += 1
            if line[1] == pred:
                num_correct += 1
            if(num_tested % 1000 == 0):
                print("Tested: "+str(num_tested)+", correct "+str(num_correct)+" acc="+str(float(num_correct/num_tested)))
        print("Final tested: "+str(num_tested)+", correct "+str(num_correct)+" acc="+str(float(num_correct/num_tested)))
        return float(num_correct/num_tested)

    def cal_py(self, y):
        return log(float(self.total_words[y])/self.num_words)
        
    def cal_p_xi_y(self, xi, y):
        if not self.x_num[y].__contains__(xi):
            if self.laplace:
                return log(1.0*self.lam/(self.total_words[y]+self.lam*len(xi)))
            else:
                return self.min_prob
        else:
            return log(float(self.x_num[y][xi])/float(self.total_words[y]))
    
    def split_email(self, content):
        re_words = re.compile(u"[\u4e00-\u9fa5]+")
        return re.findall(re_words, content)

