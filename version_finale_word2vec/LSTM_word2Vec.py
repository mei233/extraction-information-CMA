#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 21:05:49 2020

@author: mei
"""

import re
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report
from gensim.models import KeyedVectors
from keras.layers.recurrent import LSTM
# from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense,Dropout
# from keras.layers import Dense, Activation,Dropout
from keras.preprocessing import sequence
from keras.optimizers import Adam
import matplotlib.pyplot as plt  
from sklearn.utils import class_weight

class LstmWord2vec(object):
    def __init__(self):
        self.train_file = "train_file_word2vec.csv"
        self.test_file = "test_file_word2vec.csv"
        # model = "GoogleNews-vectors-negative300.bin"
        self.w2v_model_file = "word2vec_twitter_tokens.bin"
        self.w2v_model = ''
    
    def load_model(self,load_model_file):
        print(f"Loading model {load_model_file}...")
        self.lstm_model = load_model(load_model_file)
        if not self.w2v_model:
            print(f"Loading word2vec model...")
            self.w2v_model = KeyedVectors.load_word2vec_format(self.w2v_model_file, 
                                                               binary=True,
                                                               unicode_errors='ignore')

    def clean(self,elem): 
        '''
        Parameters
        ----------
        elem : str
            a sequence needed to delet the punctuation or specific characters 

        Returns
        -------
        str
            the sequence without the punctuation or specific characters.

        '''
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
        elem = re.sub('RT','',elem)
        elem = re.sub('@[^ ]+','',elem)
        elem = re.sub('#[^ ]+','',elem)
        elem = re.sub('http[^ ]+','',elem)
        elem = elem.replace('@','')
        
    
        return ' '.join([word for word in elem.split() if word != ''])
    
    def get_data(self,file):
        '''
        Parameters
        ----------
        file : str
            the filename of a csv file which has 2 columns 'tweet' and 'existence'

        Returns
        -------
        x : array of numpy, of shape (number of sequences, 
                                      number of words per sequences, 
                                      number of features)
            it contains word vectors extracted from the word2vec model, the number 
            of words per sequences is paded to 22.
        y : array of numpy of shape (number of sentences)
            corresponding label of sequence

        '''
        df = pd.read_csv(file, sep=',',engine='python',error_bad_lines=False)
        df.dropna(axis=0, how='any', inplace=True)
        sentences = []
        for elem in df['tweet']:
            sen = word_tokenize(elem)
            sentences.append(sen)
        x = self.text_to_index_array(sentences)
        y = np.array(df['existence'])
        return x,y
    
    def text_to_index_array(self,sentences):
        '''
        Parameters
        ----------
        sentences : list of list of str
            a list sequences. each sub-list is a list of words that need to vectorize with 
            the word2vec model

        Returns
        -------
        array of numpy, of shape (number of sequences, number of words per sequences
                                  , number of features)
            it contains word vectors extracted from the word2vec model, the number 
            of words per sequences is paded to 22.

        '''
        new_sentences = []
        maxlen=22
        nb_features = self.w2v_model.vectors.shape[1]
        for sen in sentences:
            new_sen = []
            for word in sen:
                try:
                    new_sen.append(self.w2v_model[word]) 
                except:
                    new_sen.append(np.zeros(nb_features)) 
            new_sentences.append(new_sen)
        return sequence.pad_sequences(new_sentences,dtype='float64',maxlen=maxlen)
            # avg_sen = np.mean(np.array(new_sen),axis=0)
            # new_sentences.append(np.array(avg_sen))
        # return np.array(new_sentences)
    
    def draw_lc(self):
        '''
        Print out the learning curve of the model LSTM 
        
        Returns
        -------
        None.
            
        '''
        # Plotting Loss Graphs
        plt.figure(figsize=(12, 12))
        plt.plot(self.history_lstm.history['loss'])
        plt.plot(self.history_lstm.history['val_loss'])
        plt.title('Loss')
        plt.legend(['train', 'val'], loc='upper left')
        plt.grid(True)
        plt.show()
        
        # Plotting Accuracy Graphs
        plt.figure(figsize=(12, 12))
        plt.plot(self.history_lstm.history['accuracy'])
        plt.plot(self.history_lstm.history['val_accuracy'])
        plt.title('Accuracy')
        plt.legend(['train', 'val'], loc='upper left')
        plt.grid(True)
        plt.show()
    
    def save(self):
        '''
        save the model

        Returns
        -------
        None.

        '''
        print('Saving model...\n')
        self.lstm_model.save('lstm_w2v_model.h5')
        print(f'model saved as lstm_w2v_model.h5')
    
    def train_lstm(self,save_model=False,plot=False):  
        '''
        training model LSTM with word2vec
        
        Parameters
        ----------
        plot : Boolean, optional
            The default is False. 
            If plot is True, it will print out the learning curve.
            
        Returns
        -------
        None.

        '''
        #load word2vec model
        if not self.w2v_model:
            print(f"Loading word2vec model...")
            self.w2v_model = KeyedVectors.load_word2vec_format(self.w2v_model_file, 
                                                               binary=True,
                                                               unicode_errors='ignore')
        Train_X, Train_Y = self.get_data(self.train_file)
        Test_X, Test_Y = self.get_data(self.test_file)
        
        #create lstm model
        class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(Train_Y),
                                                 Train_Y)
        # print(class_weights)
        self.lstm_model = Sequential()
        self.lstm_model.add(LSTM(units=16))
        self.lstm_model.add(Dropout(0.6))
        self.lstm_model.add(Dense(3, activation='softmax'))
        opt = Adam(lr=0.01,beta_1=0.9,beta_2=0.999,decay=0.01)
        self.lstm_model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', 
                                metrics=['accuracy'])
        batch_size = 512
        # self.history_lstm = self.lstm_model.fit(Train_X,Train_Y,batch_size = batch_size, 
        #                                epochs = 10,verbose=1,validation_data=(Test_X, Test_Y))
        #with validation_split 
        self.history_lstm = self.lstm_model.fit(Train_X,Train_Y,
                                                batch_size = batch_size,
                                                class_weight=class_weights,
                                                epochs = 13,
                                                verbose=0,
                                                validation_split=0.25)
        self.predict_model(self.test_file)
        #draw the graphs of the learning curve of model lstm 
        if plot:
            self.draw_lc()
        if save_model:
            self.save()
    
    def predict_model(self,file):
        '''
        evaluate the model by predicting some sentences in a csv file

        Parameters
        ----------
        ffile : str
            the filename of a csv file which has 2 columns 'tweet' and 'existence'.

        Returns
        -------
        None.

        '''
        Test_X, Test_Y = self.get_data(file)
        #print evaluation result 
        y_pred = self.lstm_model.predict(Test_X)
        # print(y_pred)
        y_pred = y_pred.argmax(axis=1)
        # print(y_pred)
        # target_names = ['class 0', 'class 1', 'class 2']
        print(classification_report(Test_Y, y_pred))
        
        
    def predict_sentence(self,sentences):
        '''
        User input a sentence, then the model will give its prediction :
            'Y' means 'Yes', 'N' means 'No', 'A' means 'we don't know'
            
        Parameters
        ----------
        sentences : str
            a sequence for testing the model/predicting the polarity 

        Returns
        -------
        list of labels

        '''
        label = ['N','Y','A']
        if isinstance(sentences,str):
            sentences = [sentences]
        # sentences = [self.clean(sen) for sen in sentences]
        sentences = [word_tokenize(self.clean(sen)) for sen in sentences]
        x = self.text_to_index_array(sentences)
        y_predict = self.lstm_model.predict(x)
        y_predict = y_predict.argmax(axis=1)
        # print(y_predict)
        
        return [label[y] for y in y_predict]
        

if __name__ == '__main__':
    
    # create an instance of LstmWord2vec 
    lstm_w2v = LstmWord2vec()
    
# =============================================================================
#     train a lstm model
# =============================================================================
    lstm_w2v.train_lstm(save_model=False,plot=True)
    
# =============================================================================
#     load a lstm model if don't want to train a new one
# =============================================================================
    # lstm_w2v.load_model('lstm_w2v_model.h5')
    
# =============================================================================
#     prediction of sentences
# =============================================================================
    # test_sentence = ["I am sure that the temperature is increasing, you don't notice that? ",
    #                  'Globle warming is not true']
    # predict_label_list = lstm_w2v.predict_sentence(test_sentence)
    # print(predict_label_list)
