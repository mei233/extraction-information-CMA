#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 15:34:23 2020

@author: mei
"""
import re
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer  
lemmatizer = WordNetLemmatizer()  
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.svm import SVC,LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
iris = datasets.load_iris()


def clean(elem) : 

    elem = elem.replace('"','')
    elem = elem.replace('(','')
    elem = elem.replace(')','')
    elem = elem.replace(',','')
    elem = elem.replace('.','')
    elem = elem.replace('[','')
    elem = elem.replace(']','')
    elem = elem.replace('  ',' ')
    elem = elem.replace('\t',' ')
    elem = elem.replace('<','')
    elem = elem.replace('>','')
    elem = elem.replace(" '",'')
    elem = elem.replace("' ",'')
    elem = elem.replace("�_�",'')
    elem = elem.replace("��_",'')
    # elem = elem.replace("'s",' is')
    elem = elem.replace('-','')
    elem = elem.replace('/br','')
    elem = elem.replace('?','')
    elem = elem.replace(':','')
    elem = elem.replace('...','')
    elem = elem.replace('!','')
    elem = elem.replace('/','')
    elem = elem.replace('\\','')
    elem = elem.replace('>','')
    elem = elem.replace(';',' ')
    elem = elem.replace('/>','')
    elem = elem.replace('[link]','')
    elem = elem.replace('link','')
    elem = elem.replace('|',' ')
    elem = elem.replace('..',' ')
    elem = elem.replace('rt','')
    elem = re.sub(r'RT','',elem)
    elem = re.sub(r'@[^ ]+','',elem)
    elem = re.sub(r'#[^ ]+','',elem)
    elem = re.sub(r'http[^ ]+','',elem)
    elem = elem.replace('@','')
    

    return ' '.join([word for word in elem.split() if word != ''])

def SupprimerDoublons(CSV_File):
    df = pd.read_csv(CSV_File,sep=',',engine='python',error_bad_lines=False)
    df_sorted = df.sort_values('existence.confidence', ascending=False).reset_index(drop=True)
    non_duplicate=df_sorted.groupby('tweet', as_index=False).first()
    
    return non_duplicate


dict_y = {
    'A': 2,
    'Y': 1,
    'N': 0
    }


def tokens(elem) :
    tokens = word_tokenize(elem)
    return ' '.join(tokens)


def token_swds(elem) :
    tokens = word_tokenize(elem)
    return ' '.join(token for token in tokens if token not in stop_words and token != '' )

def tokenlemme(elem):
    tokens = word_tokenize(elem)
    return ' '.join(lemmatizer.lemmatize(token)for token in tokens if token != '' )

def stopslemme(elem):
    tokens = word_tokenize(elem)
    return ' '.join(lemmatizer.lemmatize(token)for token in tokens if token not in stop_words and token != '' )
    
def random_tirer(Corpus,nb_sample):
    corpus_yes = Corpus[Corpus['existence']=='Y' ]
    # corpus_yes = Corpus[Corpus['existence.confidence']==1.0]
    corpus_random_yes =  corpus_yes.sample(n=nb_sample, frac=None, replace=False,  weights=None, random_state=None, axis=0)
    corpus_no = Corpus[Corpus['existence']=='N']
    corpus_random_no =  corpus_no.sample(n=nb_sample, frac=None, replace=False,  weights=None, random_state=None, axis=0)
    corpus_sait_pas = Corpus[Corpus['existence']=='A']
    corpus_random_sait_pas =  corpus_sait_pas.sample(n=nb_sample, frac=None, replace=False,  weights=None, random_state=None, axis=0)
    Corpus = pd.concat([corpus_random_yes,corpus_random_no,corpus_random_sait_pas],axis=0)
    
    return Corpus

def pretraitement(file_in, train_file, test_file): 

    
    # Step - a : supprimer les doublons
    Corpus = SupprimerDoublons(file_in)
    # if you don't want to delet those repetitions, do this one
    # Corpus = pd.read_csv(file_in, sep=',',engine='python',error_bad_lines=False)
     
    # Step - b : Remove blank rows if any.
    Corpus.dropna(axis=0, how='any', inplace=True)
    
    # Step - c : modifier la fidelité 
    # Corpus = Corpus[Corpus['existence.confidence'].between(0.8,0.99)]
    Corpus = Corpus[Corpus['existence.confidence']>=0.7]
    
    # Step - d : tirer le nombre des tweets au hasard
    Corpus = random_tirer(Corpus,456)
    # print(Corpus)

    ############# Si vous voulez avoir un train_file/test_file avec les nettoiment 
    
    # #prétaitment
    Corpus['tweet'] = [entry.lower() for entry in Corpus['tweet']]

    # #clean les punctuations
    Corpus['tweet'] = [clean(elem) for elem in Corpus['tweet']]
    
    # #tokenization
    # Corpus['tweet'] = [tokens(elem) for elem in Corpus['tweet']]
    
    # #tokenization et supprimer les stopswords
    # Corpus['tweet'] = [token_swds(elem) for elem in Corpus['tweet']]
    
    
    # #token, supprimer les stopswords, lemmetization
    # Corpus['tweet'] = [stopslemme(elem) for elem in Corpus['tweet']]
    
    #modifier Corpus['existence']
    Corpus['existence'] = [dict_y[elem] for elem in Corpus['existence']]
    

    ###########
    
    #split en train / test
    Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['tweet'],Corpus['existence'],test_size=0.2, random_state = 0)
    # print(Train_X.values[0])
    # Train_X, Test_X, Train_Y, Test_Y = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)
    
    #train to file csv
    train={'existence':Train_Y.values,
        'tweet':Train_X.values
        }
    data = DataFrame(train)
    data.to_csv(train_file,index=None,sep=",")
    
    #test to file csv
    test={'existence':Test_Y.values,
        'tweet':Test_X.values
        }
    data_test = DataFrame(test)
    data_test.to_csv(test_file,index=None,sep=",")
    
    


file_in = "tweet_global_warming_v1.csv"
train_file = "train_file.csv"
test_file = "test_file.csv"

pretraitement(file_in,train_file,test_file)

    