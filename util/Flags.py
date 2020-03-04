# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:02:08 2020

@author: Ganyu Wang


All the main parameter for every part. 


"""


from absl import flags


FLAGS = flags.FLAGS


def create_flags():
    
    f = flags
    
    
    # data generator
    f.DEFINE_string('audio_archive', "data/CV_EN/clips/", 'the folder path that save all of the audio files.')
    
    f.DEFINE_string('train_file_path', "data/CV_EN/train.csv", "the file path for train.csv")
    f.DEFINE_string('test_file_path', "data/CV_EN/test.csv", "the file path for test.csv") 
    f.DEFINE_string('dev_file_path', "data/CV_EN/dev.csv", "the file path for validation.csv")

    f.DEFINE_string('alphabet_path', "data/CV_EN/alphabet.txt", 'the file path of Alphabet.')
    
    
    f.DEFINE_integer('time_step_length', 432, "the max time step length for one sample. default 432")
    f.DEFINE_integer('window_length', 3, "the length of each time window.")
    f.DEFINE_integer('n_mfcc', 20, "the number of the MFCCs")
    f.DEFINE_integer('n_character', 28, "The number of chracters in alphabet")
    

    # Net work parameter： use small number for test.
    f.DEFINE_integer('epochs', 3, "The number of epochs")
    f.DEFINE_integer('batch_size', 1, "Batch size for training")
    f.DEFINE_integer('n_hidden', 32, "the number of hidden unit in Dense layer.")
    f.DEFINE_float('dropout', 0.05, "the number of dropout rate. in each layer.")


        
    
    
    
