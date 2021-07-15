# -*- coding: utf-8 -*-
"""

+----------------+
| Data Generator |
+----------------+

These files holds the augmentations
and the data loader function

Copyright © Sagnik Roy  July, 2021.
All rights reserved.

"""


# Libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator as GEN
from utils import parse_datagen_hparams



# datagenartor hyperparameters
IMG_SHAPE, BATCH_SIZE, SEED , image_folder = parse_datagen_hparams()


# Only augmenting the train datas as more data need 
# to find the best fitted model and reduce bias.


# train data generator
train_augment = GEN(
    rescale = 1. / 255 ,
    horizontal_flip = True , 
    zoom_range = 0.1 ,
    shear_range = 0.1 ,
    validation_split = 0.2 , 
)


# validation data generator
val_augment = GEN(
    rescale = 1. / 255 ,
    validation_split = 0.2 , 
)


def train_loader():
    train_data = train_augment.flow_from_directory(
    image_folder , 
    target_size = IMG_SHAPE ,
    batch_size = BATCH_SIZE , 
    subset = 'training' , 
    class_mode = None , 
    seed = SEED , 
    )
    
    print('Train data has been prepared..........')
    
    return train_data


def validation_loader():
    val_data = val_augment.flow_from_directory(
    image_folder , 
    target_size = IMG_SHAPE , 
    batch_size = BATCH_SIZE , 
    subset = 'validation' , 
    class_mode = None , 
    seed = SEED , 
    )
    
    print('Validation data has been prepared.......')
    
    return val_data
