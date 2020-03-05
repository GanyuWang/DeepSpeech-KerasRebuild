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

import os

from Model1 import create_model, ctc_loss, optimizer


def main(_):
    
    # generator 
    data_preprocess_params = {'batch_size': FLAGS.batch_size,
                              'time_step': FLAGS.time_step_length, 
                              'window_len': FLAGS.window_length,
                              'n_mfcc' : FLAGS.n_mfcc, 
                              'n_classes': FLAGS.n_character
                              }
    train_generator = DataGenerator(FLAGS.train_file_path, FLAGS.alphabet_path, **data_preprocess_params)
    dev_generator = DataGenerator(FLAGS.dev_file_path, FLAGS.alphabet_path, **data_preprocess_params)

    # check point
    # Include the epoch in the file name (uses `str.format`)
    checkpoint_path = "checkpoint1/model1-{epoch:04d}.ckpt"
    #checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights every 5 epochs
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
                                                filepath=checkpoint_path, 
                                                verbose=1, 
                                                save_weights_only=True,
                                                period=1)
    
    early_stoping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=0)
    
    
    # model
    model = create_model()
    
    model.save_weights("checkpoint/cp0") #initial checkpoint
    
    model.compile(optimizer=optimizer, loss=ctc_loss)
    
    model.fit_generator(generator=train_generator,
                        validation_data=dev_generator,
                        use_multiprocessing=False,
                        epochs=FLAGS.epochs,
                        callbacks=[checkpoint_callback, early_stoping_callback]
                        )
    
    
if __name__ == "__main__":
    create_flags()
    absl.app.run(main)


