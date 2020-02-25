# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:02:20 2020

@author: Ganyu Wang

The main code for training DeepSpeech with keras. 



"""


import keras
from util.Feeding import DataGenerator
from util.Flags import create_flags, FLAGS
import absl.app

from Model1 import create_model, ctc_loss, optimizer


def main(_):
    
    data_preprocess_params = {'batch_size': FLAGS.batch_size+5,
                              'time_step': FLAGS.time_step_length+5, 
                              'window_len': FLAGS.window_length+5,
                              'n_mfcc' : FLAGS.n_mfcc, 
                              'n_classes': FLAGS.n_character
                              }
    train_generator = DataGenerator(FLAGS.train_file_path, FLAGS.alphabet_path, **data_preprocess_params)
    dev_generator = DataGenerator(FLAGS.dev_file_path, FLAGS.alphabet_path, **data_preprocess_params)

    
    model = create_model()
    
    model.compile(optimizer=optimizer, loss=ctc_loss)
    model.fit_generator(generator=train_generator,
                        validation_data=dev_generator,
                        use_multiprocessing=False)      # True 
    
    
if __name__ == "__main__":
    create_flags()
    absl.app.run(main)

