# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 19:21:25 2020

@author: Ganyu Wang

DeepSpeech model



"""

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Bidirectional, LSTM, Softmax, TimeDistributed, Masking
from keras.utils import to_categorical

import tensorflow.compat.v1 as tf

import numpy as np

from util.Flags import FLAGS



############
optimizer = keras.optimizers.Adam( beta_1=0.9, beta_2=0.999, amsgrad=False)

def ctc_loss(y_true, y_pred):
    # print(y_true)
    # print(y_pred)
    
    y_true = tf.reshape(y_true, (FLAGS.batch_size, FLAGS.time_step_length))
    y_pred = tf.reshape(y_pred, (FLAGS.batch_size, FLAGS.time_step_length, FLAGS.n_character+1))
    
    input_length = np.ones((FLAGS.batch_size, 1))*FLAGS.time_step_length
    label_length = np.ones((FLAGS.batch_size, 1))*FLAGS.time_step_length
    
    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    
    return loss
    

def create_model():
    
    # network parameters
    n_hidden = FLAGS.n_hidden
    rate_dropout = FLAGS.dropout
    time_step_len = FLAGS.time_step_length
    window_len = FLAGS.window_length
    n_mfcc = FLAGS.n_mfcc
    n_class = FLAGS.n_character
    
    # build model 
    model = Sequential()
    
    model.add(Masking(mask_value= 0. , input_shape=(time_step_len, window_len*n_mfcc)))
    model.add(TimeDistributed(Dense(n_hidden, activation='relu', input_dim=(window_len* n_mfcc), )))
    model.add(TimeDistributed(Dropout(rate_dropout)))
    
    model.add(TimeDistributed(Dense(n_hidden, activation='relu', input_dim=(window_len* n_mfcc), )))
    model.add(TimeDistributed(Dropout(rate_dropout)))
    
    model.add(TimeDistributed(Dense(n_hidden, activation='relu', input_dim=(window_len* n_mfcc), )))
    model.add(TimeDistributed(Dropout(rate_dropout)))
    
    model.add(Bidirectional(LSTM(n_hidden, return_sequences=True)))
    model.add(TimeDistributed(Dropout(rate_dropout)))
    
    # predict the null label of ctc loss
    model.add(TimeDistributed(Dense(n_class+1)))
    
    model.add(TimeDistributed(Softmax(axis=-1)))
    
    return model
    
