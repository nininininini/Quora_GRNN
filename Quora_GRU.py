#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 11:45:24 2017

@author: caradumelvin
"""

import sys
import os
import numpy as np
import theano
import theano.tensor as T
import lasagne 
import time
import nltk.data
import pandas as pd
from tqdm import tqdm
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import MiniBatchKMeans
reload(sys)
sys.setdefaultencoding('utf-8')

#Path of the word vectors, here I use the 50k Global Vectors for Word Representation (Pennington et al., Stanford NLP)
#More features could make the embedding more accurate but will need the network to increase it's capacity -> not actually possible due to the memory lack.
# To tackle this issue, further work will convert the preprocessed dataset to h5py chunks and then feed it to the batch iterator. 

Dict_path = '/melvin/Data/Quora/glove.6B.50d.txt'
Data_path = '/melvin/Data/Quora/'
Test_Data_path = '/melvin/Data/Quora/'

Dataset = pd.read_csv(os.path.join(Data_path, "train.csv"))
#Reduce the length of the Dataset to fit in memory:
Dataset = Dataset[0:40000]
Test_set = pd.read_csv(os.path.join(Test_Data_path, "test.csv"))

def loadGloveModel(gloveFile):
    print "Loading Glove Model"
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = [float(val) for val in splitLine[1:]]
        model[word] = embedding
    print "Done.",len(model)," words loaded!"
    return model


#To load tokenizing utilities : nltk.download()
def Split_Tokenize(Data, Training=True):
    X = Data.fillna('empty')
    if Training == True:
        y = X['is_duplicate']
    X1 = []
    X2 = []
    print('Tokenizing data')
    for i in tqdm(range(len(X))):
        X1.append(nltk.wordpunct_tokenize(X.iloc[i, 0]))
        X2.append(nltk.wordpunct_tokenize(X.iloc[i, 1]))
    if Training == True:
        return X1, X2, np.array(y)
    else:
        return X1, X2
    
def Encode_Question(W2V_Dict, Token):
    Q_enc = []
    stop_words = ["a", "able", "about", "across", "after", "all", "almost", "also", "am", "among",
                  "an", "and", "any", "are", "as", "at", "be", "because", "been", "but", "by", "can", 
                  "cannot", "could", "dear", "did", "do", "does", "either", "else", "ever", "every",
                  "for", "from", "get", "got", "had", "has", "have", "he", "her", "hers", "him", "his", 
                  "how", "however", "i", "if", "in", "into", "is", "it", "its", "just", "least", "let", 
                  "like", "likely", "may", "me", "might", "most", "must", "my", "neither", "no", "nor",
                  "not", "of", "off", "often", "on", "only", "or", "other", "our", "own", "rather", "said",
                  "say", "says", "she", "should", "since", "so", "some", "than", "that", "the", "their", 
                  "them", "then", "there", "these", "they", "this", "tis", "to", "too", "twas", "us", 
                  "wants", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
                  "why", "will", "with", "would", "yet", "you", "your", "ain't", "aren't", "can't", "could've", 
                  "couldn't", "didn't", "doesn't", "don't", "hasn't", "he'd", "he'll", "he's", "how'd", 
                  "how'll", "how's", "i'd", "i'll", "i'm", "i've", "isn't", "it's", "might've", "mightn't",
                  "must've", "mustn't", "shan't", "she'd", "she'll", "she's", "should've", "shouldn't", 
                  "that'll", "that's", "there's", "they'd", "they'll", "they're", "they've", "wasn't", 
                  "we'd", "we'll", "we're", "weren't", "what'd", "what's", "when'd", "when'll", "when's",
                  "where'd", "where'll", "where's", "who'd", "who'll", "who's", "why'd", "why'll", "why's", 
                  "won't", "would've", "wouldn't", "you'd", "you'll", "you're", "you've"]
    
    for word in Token:
        if word not in stop_words:
            try:
                Q_enc.append(W2V_Dict[str(word.lower().encode('utf-8'))])
            except (UnicodeDecodeError, Exception):
                pass
    return sum(Q_enc, [])


def List_to_Arr(list_to_convert, width):
    X_arr = np.zeros(shape=(len(list_to_convert), width))
    for q in range(len(X_arr)):
        X_arr[q,0:len(list_to_convert[q])] = list_to_convert[q]
    X_arr = np.reshape(X_arr, (len(X_arr), width, 1))
    return X_arr

def Get_Mask(input_list, max_length):
    mask_i  = np.zeros(shape=(max_length))
    mask = []
    for q_i in range(len(input_list)):
        mask_i[0:len(input_list[q_i])] = [1] * len(input_list[q_i]) 
        mask.append(mask_i.tolist())
    return np.asarray(mask)


def One_Hot_Labels(labels):
    y_enc = np.zeros(shape=(len(labels), 2))
    for l in tqdm(range(len(y_enc))):
        if labels[l] == 1.:
            y_enc[l] = [1., 0.]
        else: 
            y_enc[l] = [0., 1.]
    return y_enc    
    

def Preprocess_Data(Data, Training_set=True):
    if Training_set == True:
        print("Preprocessing Training set")
        X1, X2, y = Split_Tokenize(Data[['question1','question2', 'is_duplicate']], Training=True)
        y = One_Hot_Labels(np.array(y))
    else:
        print("Preprocessing Test set")
        X1, X2= Split_Tokenize(Data[['question1','question2']], Training=False)
    
    Words_Dict = loadGloveModel(Dict_path)
    X = []
    for question in tqdm(range(len(X1))):
        X.append(Encode_Question(Words_Dict, X1[question]) + 
                 Encode_Question(Words_Dict, X2[question]))

    global max_len
    if Training_set == False:
        max_len = max(max_len, max([len(X[i]) for i in range(len(X))]))
    else:
        max_len = max([len(X[i]) for i in range(len(X))])
    
    if Training_set == True:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1260)
        return X_train, X_val, y_train, y_val
    else:
        return X
    
#Not computationally efficient for large Dataset (too large time to load mini-batch) 
def batch_gen(X, y, N, Training=True):
    while True:
        if Training == True:
            idx = np.random.choice(len(y), N)
        else:
            idx = np.array(range(len(y)))
        X_batch = []
        for i in idx:
            X_batch.append(X[i])
        mask_batch = Get_Mask(X_batch, max_len)
        X_batch = List_to_Arr(X_batch, max_len)
        if Training == True:
            yield X_batch.astype('float32'), y[idx].astype('float32'), mask_batch.astype('float32')
        else:
            yield X_batch.astype('float32'), mask_batch.astype('float32')
      
def GRU(MAX_LENGTH, N_HIDDEN1, BATCH):
    print("Building Gated Recurrent Network")
    l_in = lasagne.layers.InputLayer(shape=(BATCH, MAX_LENGTH, 1))
    l_mask = lasagne.layers.InputLayer(shape=(BATCH, MAX_LENGTH))
    
    gate_parameters = lasagne.layers.recurrent.Gate(W_in=lasagne.init.Orthogonal(), 
                                                    W_hid=lasagne.init.Orthogonal(),
                                                    b=lasagne.init.Constant(0.))
    
    l_GRU1 = lasagne.layers.recurrent.GRULayer(l_in, N_HIDDEN1, resetgate=gate_parameters,
                                                updategate=gate_parameters, mask_input=l_mask,
                                                learn_init = True, grad_clipping=100, backwards=True,
                                                hidden_update=gate_parameters)
    
    l_GRU1_back = lasagne.layers.recurrent.GRULayer(l_in, N_HIDDEN1, resetgate=gate_parameters,
                                                updategate=gate_parameters, mask_input=l_mask, 
                                                learn_init=True, grad_clipping=100, backwards=True,
                                                hidden_update=gate_parameters)
    
    l_sum = lasagne.layers.ElemwiseSumLayer([l_GRU1, l_GRU1_back])
    
    
    l_lstm_slice = lasagne.layers.SliceLayer(l_sum, 0, 1)
    
    l_out = lasagne.layers.DenseLayer(l_lstm_slice, num_units=2, 
                                      nonlinearity=lasagne.nonlinearities.sigmoid)
    
    return l_in, l_mask, l_out

def Train_model(BATCH_SIZE, number_of_epochs, lr):
    
    X_train, X_val, y_train, y_val = Preprocess_Data(Dataset, Training_set=True)  
        
    l_in, l_mask, l_out = GRU(MAX_LENGTH=max_len, N_HIDDEN1=20, BATCH_SIZE)
    
    y_sym = T.matrix()

    output = lasagne.layers.get_output(l_out)
    pred = (output > 0.5)

    loss = T.mean(lasagne.objectives.binary_crossentropy(output, y_sym))

    acc = T.mean(T.eq(pred, y_sym))

    params = lasagne.layers.get_all_params(l_out)
    grad = T.grad(loss, params)
    updates = lasagne.updates.adamax(grad, params, learning_rate=lr)

    f_train = theano.function([l_in.input_var, y_sym, l_mask.input_var], [loss, acc], updates=updates)
    f_val = theano.function([l_in.input_var, y_sym, l_mask.input_var], [loss, acc])
    f_predict_probas = theano.function([l_in.input_var, l_mask.input_var], output)
    
    N_BATCHES = len(X_train) // BATCH_SIZE
    N_VAL_BATCHES = len(X_val) // BATCH_SIZE
    train_batches = batch_gen(X_train, y_train, BATCH_SIZE)
    val_batches = batch_gen(X_val, y_val, BATCH_SIZE)

    print("Start training")
    for epoch in range(number_of_epochs):
        train_loss = 0
        train_acc = 0
        start_time = time.time()
        for _ in range(N_BATCHES):
            X, y, mask = next(train_batches)
            loss, acc = f_train(X, y, mask)
            train_loss += loss
            train_acc += acc
        train_loss /= N_BATCHES
        train_acc /= N_BATCHES
            
        val_loss = 0
        val_acc = 0
        for _ in range(N_VAL_BATCHES):
            X, y, mask = next(val_batches)
            loss, acc = f_val(X, y, mask)
            val_loss += loss
            val_acc += acc
        val_loss /= N_VAL_BATCHES
        val_acc /= N_VAL_BATCHES
        print("Epoch {} of {} took {:.3f}s".format(
              epoch + 1, number_of_epochs, time.time() - start_time))
        print('  Train loss: {:.03f} - Validation Loss: {:.03f}'.format(
              train_loss, val_loss))
        print('  Train accuracy: {:.03f}'.format(train_acc))
        print('  Validation accuracy: {:.03f}'.format(val_acc))
    
    print("Testing model: ")
    del X_train, X_val, y_train, y_val
    X_test = Preprocess_Data(Test_set, Training_set=False)
    predictions = np.zeros((len(X_test), 2))
    for i in tqdm(range(len(X_test))):
        X = List_to_Arr(X_test[i:i+1], max_len)
        mask = Get_Mask(X_test[i:i+1], max_len)
        predictions[i] = f_predict_probas(X, mask)
    predictions = predictions[:,0]
    print("Test set predicted successfully, good luck !")
    
    Submission = pd.DataFrame({'id': [i for i in range(1, len(predictions)+1)],
                                      'label': predictions})
    
    Submission.to_csv(os.path.join(Data_path,r'Submission.csv'), header=True, index=False)        
    print("CSV submission file generated in Data_path")

 
    
if __name__ == "__main__":
    #Don't forget to change paths n_epochs before running on AWS !!!!!!
    Train_model(BATCH_SIZE=64, number_of_epochs=30, lr=0.002)
