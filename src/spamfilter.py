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
    def __init__(self, x_num, total_words, testset, hypothesis=["spam", "ham"], laplace=True, lam=1.0, min_prob=1e-50, use_mailer=False, mailer_dict={}, use_from=False):
        BayesClassifier.__init__(self, hypothesis, laplace, lam, min_prob)
        self.x_num = x_num
        self.total_words = total_words
        self.testset = testset
        self.num_words = 0
        self.use_mailer = use_mailer
        self.mailer_dict = mailer_dict
        self.use_from = use_from
        for item in total_words.keys():
            self.num_words += total_words[item]
    
    def classify(self):
        tot_mailer = {"spam": 0, "ham": 0} 
        if self.use_mailer:
            for a in self.mailer_dict["spam"].keys():
                tot_mailer["spam"] += self.mailer_dict["spam"][a]
            for b in self.mailer_dict["ham"].keys():
                tot_mailer["ham"] += self.mailer_dict["ham"][b]
            
        num_tested = 0
        num_correct = 0
        num_spam_real = 0
        num_spam_pred = 0
        num_spam_right_pred = 0
        # print("Total " + str(len(self.testset)) + " to test")
        for line in self.testset:
            try:
                f = open(line[0])
                content = f.read() # read as an string
                f.close()
            except:
                print("Test file " + line[0] + " not found")
                continue
            res = self.split_email(content)

            mailer_pattern = re.compile(u"X-Mailer.*")
            mailer = re.findall(mailer_pattern, content)
            
            from_pattern = re.compile(u"From.*@.*", re.IGNORECASE)
            recv_from = re.findall(from_pattern, content)
            if len(recv_from) != 0 and self.use_from:
                recv_from = recv_from[0].split('@')[-1]
                email = recv_from.split('>')[0].split(')')[0].split(' ')[0]
                res.append(email)

            pred = self.hypothesis[0]
            max_p = float('-Inf')
            for y in self.hypothesis:
                prob = self.cal_py(y)

                if self.use_mailer & len(mailer) == 1:
                    if self.mailer_dict[y].__contains__(mailer[0]):
                        mailer_prob = log(float(self.mailer_dict[y][mailer[0]])/float(tot_mailer[y]))
                    elif self.laplace:
                        mailer_prob = log(1.0*self.lam/(tot_mailer[y]+self.lam*tot_mailer[y]))
                    else:
                        mailer_prob = self.min_prob
                    prob += mailer_prob

                for item in res:
                    prob += self.cal_p_xi_y(item, y)

                if prob > max_p:
                    max_p = prob
                    pred = y

            num_tested += 1
            if line[1] == 'spam':
                num_spam_real += 1
            if pred == 'spam':
                num_spam_pred += 1
            if line[1] == pred:
                num_correct += 1
                if pred == 'spam':
                    num_spam_right_pred += 1
            if(num_tested % 1000 == 0):
                print("\r Tested: %5d/%d, correct: %5d, acc: %.8f%%, prec: %.8f%%, rec: %.8f%%" %(num_tested, len(self.testset), num_correct, float(num_correct/num_tested)*100, float(num_spam_right_pred/num_spam_pred)*100, float(num_spam_right_pred/num_spam_real)*100), end=" ")

        acc = float(num_correct/num_tested)
        prec = float(num_spam_right_pred/num_spam_pred)
        recall = float(num_spam_right_pred/num_spam_real)
        f1 = 2*(prec*recall)/(prec+recall)
        print("\r Final tested: %5d, correct: %5d, acc: %.8f%%, prec: %.8f%%, rec: %.8f%%" %(num_tested, num_correct, acc*100, prec*100, recall*100))
        return acc, prec, recall, f1

    def cal_py(self, y):
        return log(float(self.total_words[y])/self.num_words)
        
    def cal_p_xi_y(self, xi, y):
        if not self.x_num[y].__contains__(xi):
            if self.laplace:
                return log(1.0*self.lam/(self.total_words[y]+self.lam*self.total_words[y]))
            else:
                return self.min_prob
        else:
            return log(float(self.x_num[y][xi])/float(self.total_words[y]))
    
    def split_email(self, content):
        re_words = re.compile(u"[\u4e00-\u9fa5]+")
        return re.findall(re_words, content)

