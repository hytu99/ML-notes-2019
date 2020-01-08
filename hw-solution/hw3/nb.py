# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 20:42:13 2019

@author: Zhanqiu Zhang
"""

import os
from collections import defaultdict
from math import log

class Classifier:
    
    def __init__(self,train_data_path):
        
        # scan train data and count words
        self.msg_dict,self.sp_dict,self.msg_count,self.sp_count,self.words_count=self.train(train_data_path)
        
        # take log for probabilities
        self.take_log()
        
    def train(self,path):
        
        mail_names=os.listdir(path)
        
        # dictionaries to count words in spam mails and non-spam mails
        msg_dict=defaultdict(int)
        sp_dict=defaultdict(int)
        
        # count spam mails and non-spam mails
        msg_count=0
        sp_count=0
        
        # count total words in all mails
        word_set=set()
        
        # scan all training mails
        content=None
        for name in mail_names:
            with open(path+"\\"+name) as file:
                for content in file:
                    pass
            content_list=content.strip().split()
            
            if name[0]=="s":
                sp_count+=1
                for word in content_list:
                    if word.isalpha():
                        word_set.add(word)
                        sp_dict[word]+=1
            else:
                msg_count+=1
                for word in content_list:
                    if word.isalpha():
                        word_set.add(word)
                        msg_dict[word]+=1
                
        return msg_dict,sp_dict,msg_count,sp_count,len(word_set)
    
    def take_log(self):
        
        # log( P(spam) ) and log( P(non-spam) )
        self.log_sp_freq=log(self.sp_count/(self.sp_count+self.msg_count))
        self.log_msg_freq=log(self.msg_count/(self.sp_count+self.msg_count))

        self.log_sp_total_word=log(sum(self.sp_dict.values())+self.words_count)
        self.log_msg_total_word=log(sum(self.msg_dict.values())+self.words_count)
        
        # dictionaries for log( P(word|spam) ) and log( P(word|non-spam) )
        self.log_sp_prob_dict=defaultdict(lambda :-self.log_sp_total_word)
        self.log_msg_prob_dict=defaultdict(lambda :-self.log_msg_total_word)
        
        for word in self.sp_dict:
            self.log_sp_prob_dict[word]+=log(self.sp_dict[word]+1)
        for word in self.msg_dict:
            self.log_msg_prob_dict[word]+=log(self.msg_dict[word]+1)
        
        
    def test(self,mail_path):
        
        # scan test mail
        with open(mail_path) as file:
            for content in file:
                pass
        content_list=content.strip().split()
        
        # calculate probabilities for spam and non-spam
        log_p_sp=self.log_sp_freq
        log_p_msg=self.log_msg_freq
        for word in content_list:
            if word.isalpha():
                log_p_sp+=self.log_sp_prob_dict[word]
                log_p_msg+=self.log_msg_prob_dict[word]
        
        # prediction
        if log_p_sp>=log_p_msg:
            return 0 # 1 is spam
        else:
            return 1 # 0 is non-spam
        
def main():
    
    # get paths for train data and test data
    # the code should be in the same path as README.md
    work_path=os.getcwd()
    train_path=work_path+"\\train-mails"
    test_path=work_path+"\\test-mails"
    
    # train classifier
    classifier=Classifier(train_path)
    confusion=[[0,0],[0,0]] # true negtive, false negative, false positive, true positive

    # test classifier
    label=None
    test_mail_names=os.listdir(test_path)
    for name in test_mail_names:
        prediction=classifier.test(test_path+"\\"+name)
        if name[0]=="s":
            label=0
        else:
            label=1
        confusion[prediction][label]+=1
        
    # result
    print("The confusion matrix:")
    print(confusion)
    print("Precision:")
    Precision = confusion[0][0]/(confusion[0][0]+confusion[0][1])
    print(Precision)
    print("Recall:")
    Recall = confusion[0][0]/(confusion[0][0]+confusion[1][0])
    print(Recall)
    print("F1:")
    print(2/(1/Precision+1/Recall))
    
if __name__ == "__main__":
    main()
