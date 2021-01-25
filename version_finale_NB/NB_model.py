#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:22:30 2020

@author: mei
"""
'''
The script is to train a clissififier on global warming corpus. 
The train and test data need to be split before with the script 'pretraitment_nettoiment.py'

'''
import re
import pandas as pd
from nltk.stem import WordNetLemmatizer  
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer() 
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import naive_bayes
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
# from sklearn.externals import joblib
import joblib


class predict_pretaitement(object):
        
    def clean(self,elem) : 
    
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
    
    def tokens(self,elem) :
        tokens = word_tokenize(elem)
        return ' '.join(tokens)
    
    
    def token_swds(self,elem) :
        tokens = word_tokenize(elem)
        return ' '.join(token for token in tokens if token not in stop_words and token != '' )
    
    def tokenlemme(self,elem):
        tokens = word_tokenize(elem)
        return ' '.join(lemmatizer.lemmatize(token)for token in tokens if token != '' )
    
    def stopslemme(self,elem):
        tokens = word_tokenize(elem)
        return ' '.join(lemmatizer.lemmatize(token)for token in tokens if token not in stop_words and token != '' )
        
    
    def pretraitement(self,sentences): 
        self.sentences = sentences
        if isinstance(self.sentences,str):
            self.sentences = [sentences]
        ############# Si vous voulez avoir un train_file/test_file avec les nettoiment 
        
        # #prétaitment
        self.sentences = [entry.lower() for entry in self.sentences]
        
        # #clean les punctuations
        self.sentences = [self.clean(elem) for elem in self.sentences]
        
        # #tokenization
        # # Corpus['tweet'] = [tokens(elem) for elem in Corpus['tweet']]
        
        # #tokenization et supprimer les stopswords
        # # Corpus['tweet'] = [token_swds(elem) for elem in Corpus['tweet']]
        
        # #token, supprimer les stopswords, lemmetization
        self.sentences = [self.stopslemme(elem) for elem in self.sentences]
        
        #modifier Corpus['existence']
        self.sentences = [self.dict_y[elem] for elem in self.sentences]
        
    
    
class classifier_pipeline(object):
    def __init__(self):
        self.train_file = "train_file.csv"
        self.test_file = "test_file.csv"
    
    def load_model(self,model_file,vector_file):
        '''
    
        Parameters
        ----------
        model_file : 
            the path and the file name of the model file
        vector_file : TYPE
            the path and the file name of the tfidfvector model file

        Returns
        -------
        None.

        '''
        print(f"Loading model {model_file}...")
        self.vectorizer = joblib.load(vector_file)
        print(f"Loading model {vector_file}...")
        self.Naive = joblib.load(model_file) 
        # self.model = joblib.load(load_model_file) 
        

    def BowSimple(self):
        '''
        create bag of words 

        Returns
        -------
        BTrain_X : 
            training data of Bow
        BTest_X : 
            test data of Bow
        BTrain_Y : 
            training label of Bow
        BTest_Y : 
            test label of Bow

        '''
        print("Treatment with Bow\n")
        #transfer them according to different choices
        cvectorizer = CountVectorizer()
        BTrain_X = cvectorizer.fit_transform(self.Train_X)
        BTest_X = cvectorizer.transform(self.Test_X)
        
        Encoder = LabelEncoder()
        BTrain_Y = Encoder.fit_transform(self.Train_Y)
        # print(Train_Y)
    
        BTest_Y = Encoder.fit_transform(self.Test_Y)
        
        return BTrain_X,BTest_X, BTrain_Y,BTest_Y
    
    
    def TfidfVector(self, ngram_range = (1, 2)):
        '''
        ngram_range = (1, 1) : unigramme
        Vous pouvez changez en (2,2) qui est bigramme, ou (1,2) qui est unigramme et bigramme, selon vos besoins
        
        create tfidf vector 

        Returns
        -------
        BTrain_X : class 'scipy.sparse.csr.csr_matrix'
            training data of tfidfvector
        BTest_X : class 'scipy.sparse.csr.csr_matrix'
            test data of tfidfvector
        BTrain_Y : class 'scipy.sparse.csr.csr_matrix'
            training label of tfidfvector
        BTest_Y : class 'scipy.sparse.csr.csr_matrix'
            test label of tfidfvector
        '''
        
        print("Treatment with tfidfVector\n")
        #transfer them according to different choices
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, lowercase=False,stop_words=None,norm='l2',max_df = 0.8)
        DTrain_X = self.vectorizer.fit_transform(self.Train_X)
        
        DTest_X = self.vectorizer.transform(self.Test_X)
        
        Encoder = LabelEncoder()
        DTrain_Y = Encoder.fit_transform(self.Train_Y)
        
        DTest_Y = Encoder.fit_transform(self.Test_Y)
        
        return DTrain_X,DTest_X,DTrain_Y,DTest_Y
    
    
    
    def AccuracyScore(self,methodes,predict):
        '''
        print out accuracy score and classcification report (precision, recall and f-score)

        Parameters
        ----------
        methodes : 
            the name of the method.
        predict : 
            the prediction result from the method.

        Returns
        -------
        accuracy : TYPE
            the accuracy of the model

        '''
        accuracy = accuracy_score(predict, self.Test_Y)*100
        print(f"{methodes} Accuracy Score -> ",{accuracy})
        # print(f"{methodes} Accuracy Score -> ",f1_score(predict, Test_Y,average='macro')*100)
            
        print(classification_report(self.Test_Y,predict, target_names=['A','N','Y']))
        return accuracy
    
    def Graphes(self,C):
        '''
        Print out the learning curve of the model
        
        Returns
        -------
        None.

        '''
        print(f"Print out the graph...\n")
        ax1 = plt.subplot()
        ax1.set_xscale('log')
        ax1.plot(C,self.accuracyNB,marker='o',color='g')
        ax1.set_xlabel('alpha')
        ax1.set_ylabel('Accuracy NB')
        ax1.grid()
        plt.show()
    
    '''
    au-dessous les fonctions de méthodes :
    
    '''
    def NaiveBayesNB(self,C,VTrain_X, VTrain_Y,VTest_X,VTest_Y):
        '''
        

        Parameters
        ----------
        C : a list of int or float
            the parameter
        VTrain_X : 
            training data 
        VTrain_Y : 
            training label
        VTest_X : 
            test data
        VTest_Y : 
            test label

        Returns
        -------
        None.

        '''
        self.accuracyNB = []
        for c in C:
            self.Naive = naive_bayes.MultinomialNB(alpha=c)
            self.Naive.fit(VTrain_X,VTrain_Y)
            # cross_val_score(self.Naive, VTrain_X, VTrain_Y, cv=10)
            # print(cross_val_score)
            predictions_NB = self.Naive.predict(VTest_X)
            accuracy = self.AccuracyScore('Naive Bayes',predictions_NB)
            self.accuracyNB.append(accuracy) #pour print les graphes
            # print(f'c={c}')
            print("########################################################\n\n")
        # self.Graphes(C,accuracyNB, 'c', 'Accuracy NB') #pour print les graphes
        
    
    def save(self):
        '''
        save the model

        Returns
        -------
        None.

        '''
        print('Saving model...\n')
        joblib.dump(self.Naive, "NB_model.m")
        print(f'model saved as NB_model.m')
        joblib.dump(self.vectorizer, "tfidfvectorizer_model.m")
        print(f'model saved as tfidfvectorizer_model.m')

    def pipeline(self,save_model=False,plot=False):
        '''
        main function : training the model out of the train file and test file using Multinomial NB 

        Parameters
        ----------
        save_model : optional
             The default is False. 
             If it's Ture, it will save a model
        plot : optional
            The default is False.  
            If it's Ture, it will plot the graph

        Returns
        -------
        None.

        '''
        
        print("Loading training data...\n")
        Train_corpus = pd.read_csv(self.train_file, sep=',',engine='python',error_bad_lines=False)
        Train_corpus.dropna(axis=0, how='any', inplace=True)
        print("Loading test data...")
        Test_corpus = pd.read_csv(self.test_file, sep=',',engine='python',error_bad_lines=False)
        Test_corpus.dropna(axis=0, how='any', inplace=True)
        
        self.Train_X = Train_corpus['tweet']
        self.Test_X = Test_corpus['tweet']
        self.Train_Y = Train_corpus['existence']
        self.Test_Y = Test_corpus['existence']
        
        
        # BTrain_X,BTest_X, BTrain_Y,BTest_Y = self.BowSimple(Train_X,Test_X,Train_Y,Test_Y)
    
        
        VTrain_X,VTest_X, VTrain_Y,VTest_Y = self.TfidfVector(ngram_range = (1, 2))
        
        # print(VTest_X)
        
    
        print("Chargement de l'algorithme Naive bayes multinomial\n")
        A = [0.15] 
        self.NaiveBayesNB(A,VTrain_X, VTrain_Y,VTest_X,VTest_Y)

        # print(self.Naive.score(VTest_X,VTest_Y))
        
        if save_model:
            self.save()
        
        if plot:
            self.Graphes(A)
            
            
    def predict_sentence(self,sentences):
        '''
        predict a label for a list of sentences

        Parameters
        ----------
        sentences : a list of str
            input a list of sentences for predict

        Returns
        -------
        list
            the prediction label by the model

        '''
        predict = predict_pretaitement()
        
        
        label = ['N','Y','A']
        if isinstance(sentences,str):
            sentences = [sentences]
            
        sentences = self.vectorizer.transform([predict.stopslemme(predict.clean(entry.lower())) for entry in sentences])
        # print(type(sentences))
        
        pred_result = self.Naive.predict(sentences)
        # print(pred_result)
        
        # pred_result = pred_result.argmax(axis=1)
        return [label[y] for y in pred_result]
        

        

if __name__ == '__main__':
    
    train_model = classifier_pipeline()
    
    #train a model
    train_model.pipeline(save_model=False)
    
    #load a model
    # train_model.load_model('NB_model.m','tfidfvectorizer_model.m')
    
    # predict
    # test_sentence = ["I am sure that the temperature is increasing, you don't notice that? ",
                        # 'Globle warming is not true']
    # predict_label_list = train_model.predict_sentence(test_sentence)
    # print(f'Prediction result : {predict_label_list}')
    
    
    
    
    
    