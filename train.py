# -*- coding: utf-8 -*-
"""

+----------------+
| Model Training |
+----------------+

Model training has been done 
with custom training loop.
Do visit tensorflow documentation
on custom training loop.
Link : https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch

Copyright © Sagnik Roy  July, 2021.
All rights reserved.

"""


# Libraries
from convAE import CAE
from data import train_loader , validation_loader
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Input , Conv2D , Conv2DTranspose , MaxPooling2D , UpSampling2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
import time
from utils import (
    load_data_from_source, parse_datagen_hparams, parse_training_hparams
                   )




# Data Directory Formation
print('Initializing Environment Setup.......')
load_data_from_source()


# Data Loding
print('Data Generation initializing.........')
train_data = train_loader()
val_data = validation_loader()


# Model Loading
print('Loading Model.........................')
IMG_SHAPE,_ ,_ ,_ = parse_datagen_hparams()
cae = CAE(IMG_SHAPE + (3,))
model = cae.network()
print('Model has been loaded.................')


# Training Parameters
loss_fn = MeanSquaredError()
optimizer = Adam(1e-4)
EPOCHS, steps_per_epoch, loss_tracker_index = parse_training_hparams()


# for storing the loss data
train_loss_counter = []
val_loss_counter = []


print('\n\n-----------------------------------------------------------\n')


print('Model Training has been started..........')
for epoch in range(EPOCHS) :
    
    # Epoch's initial time
    start_time = time.time()
    print(f'Epoch : {epoch + 1}')
    print('Train Loss : [',end = " ")
    
    for index, patch in enumerate(train_data):
        
        with tf.GradientTape() as tape:
            # finding loss
            outputs = model(patch , training = True)
            patch_loss = loss_fn(patch , outputs)
            
        # applying gradients on model weights
        grads = tape.gradient(patch_loss , model.trainable_weights)
        optimizer.apply_gradients(zip(grads , model.trainable_weights))
        # printing loss
        if index % loss_tracker_index == 0:
          print('%.5f'%(float(patch_loss)) , end = " , " if index != steps_per_epoch else " ]\n")
        if index == steps_per_epoch:
          train_loss_counter.append(patch_loss)
          break
      
        
    # validation loss
    tot_val_loss = 0
    for index , val_patch in enumerate(val_data):
        # updating validation loss
        val_output = model(val_patch , training = False)
        tot_val_loss += loss_fn(val_patch , val_output)
        
        if index == steps_per_epoch // 10:
            break
    print('Validation Loss : %.4f'%(tot_val_loss / index))
    
    val_loss_counter.append(tot_val_loss / index)
    
    print('Time taken for epoch %d : %.2f seconds.'%(epoch + 1 , time.time() - start_time ))
    print('...................................................\n\n')
    
print('Training Complete !')