# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:41:10 2020

@author: Ganyu Wang


Create a data generator. 


"""

import os

import pandas as pd
import numpy as np
import keras


import librosa
import librosa.display
from librosa.feature import mfcc

from util.Flags import FLAGS




class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, train_file_path, alphabet_path, \
                 batch_size=32, time_step=431, window_len=3, n_mfcc=30 , n_classes=28):
        
        'Initialization'
        self.train_file_path = train_file_path
        # self.alphabet_path = alphabet_path
        
        self.batch_size = batch_size
        self.time_step = time_step
        self.n_window = time_step
        self.window_len = window_len
        self.n_mfcc = n_mfcc
        self.n_classes = n_classes
        #
        self.alphabet = {}
        self.create_alphabet_dict(alphabet_path)
        #
        train_df = pd.read_csv(train_file_path)
        self.df_wav_filename = train_df['wav_filename']
        self.df_transcript = train_df['transcript']
        
        
    def create_alphabet_dict(self, alphabet_path):
        # load alphabet to program
        alphabet_dict = {}
        
        with open(alphabet_path, 'r') as alphabet_file:
            lines = alphabet_file.readlines()
            ind = 0
            for line in lines:
                if line[0] == '#':
                    continue
                alphabet_dict[line[0]] = ind
                ind += 1
                
        self.alphabet = alphabet_dict
        
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.df_wav_filename) / self.batch_size))

    def __getitem__(self, idx):
        'Generate one batch of data'
        # Generate indexes of the batch
        
        # Initialization 
        batch_X_before_window = np.zeros((self.batch_size, self.time_step, self.n_mfcc))
        
        # there is a zero masking in the first layer. 
        batch_X = np.zeros((self.batch_size, self.n_window, self.window_len * self.n_mfcc))
        # using the n_class as the null label for CTC LOSS
        batch_y = np.ones((self.batch_size, self.time_step), dtype=int)*self.n_classes
    
        for sample_idx in range(self.batch_size):
             # x
             tmp_x = self.get_mfccs_from_file_name(self.df_wav_filename[idx*self.batch_size+sample_idx])
             tmp_x_shape0 = tmp_x.shape[0] # here is the actual time step length
             batch_X_before_window[sample_idx, :tmp_x_shape0,] = tmp_x
             
             window_num = self.n_window - self.window_len + 1
             for window_idx in range(window_num):
                 batch_X[sample_idx, window_idx, :] =\
                     batch_X_before_window[sample_idx, window_idx: window_idx+self.window_len, :]\
                         .reshape(self.window_len * self.n_mfcc) 
             
             # y
             tmp_y = np.array(self.character2idx(self.df_transcript[idx*self.batch_size+sample_idx]))
             tmp_y_shape0 = tmp_y.shape[0]
             
             batch_y[sample_idx, :tmp_y_shape0] = tmp_y
             
             
        return batch_X, batch_y
    
    def get_mfccs_from_file_name(self, file_name):
        
        # looks like the name is changed to MP3. 
        pre, ext = os.path.splitext(file_name)
        y, sr = librosa.load(FLAGS.audio_archive + '/' + pre + '.mp3')      
        
        mfccs = mfcc(y=y, sr=sr, n_mfcc= self.n_mfcc).T
        mfccs = np.array(mfccs)
        return mfccs
    
    def character2idx(self, sentence):
        idx_vect = []
        for character in sentence:
            idx_vect.append(self.alphabet[character])
        
        return idx_vect


#%% Test
import numpy as np    
    
batch_X_before_window = np.zeros((32, 431, 30)) 
  
batch_X = np.zeros((32, 431, 3 * 30))

for sample_idx in range(32): 
    
    tmp_x = np.random.rand(200, 30)
    tmp_x_shape0 = tmp_x.shape[0]     
        
    batch_X_before_window[sample_idx, :tmp_x_shape0,] = tmp_x
    
    window_num = 431 - 3 + 1
    
    for window_idx in range(window_num):
        batch_X[sample_idx, window_idx, :] =\
            batch_X_before_window[sample_idx, window_idx: window_idx+3, :].reshape(3 * 30) 



