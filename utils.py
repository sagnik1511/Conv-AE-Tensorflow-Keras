# -*- coding: utf-8 -*-
"""

+ --------- +
| Utilities |
+ --------- +

Necessary functions are
stored in this file.

Copyright © Sagnik Roy  July, 2021.
All rights reserved.

"""

# Libraries
import matplotlib.pyplot as plt
import os
import shutil
import wget
from zipfile import ZipFile



# utils
url = 'https://github.com/sagnik1511/Conv-AE-Tensorflow-Keras/releases/download/1.0/pokemon_jpg.zip'
zip_path = 'pokemon_jpg.zip'
image_folder = 'data/'
primary_img_dir = 'pokemon_jpg/'
destination_img_dir = 'data/images/'


# Hyperparameters
IMG_SHAPE = (128 , 128)
BATCH_SIZE = 10
SEED = 42
EPOCHS = 10
steps_per_epoch = 1500
loss_tracker_index = steps_per_epoch // 10


# downloading and storing data
def load_data_from_source():
  if os.path.isdir( destination_img_dir ) == False:
    # root directory
    root = os.getcwd()
    # folder tree to store data
    os.mkdir( 'data/' )
    os.mkdir( destination_img_dir )
    # download zip file
    wget.download( url )
    # extract data from zip file
    zf = ZipFile(zip_path , 'r')
    zf.extractall(root)
    zf.close()

    # move the files into the folder  tree
    for filepath in os.listdir(primary_img_dir):
      path = os.path.join(primary_img_dir , filepath)
      shutil.move(path , destination_img_dir)
    
    # removing unnecessary data and folder
    os.rmdir(primary_img_dir)
    os.remove(zip_path)

    print('Data Directory Created.............')

  else:
    print('Directory already created.\nNot initiating download again...........')
    
    
# returning data generator hyperparametrs
def parse_datagen_hparams():
    return IMG_SHAPE, BATCH_SIZE, SEED, image_folder


# returning training hyperparametrs
def parse_training_hparams():
    return EPOCHS, steps_per_epoch, loss_tracker_index


# function to plot loss curves
def show_loss_curves(train_loss, validation_loss):
    plt.figure( figsize = (10 , 6))
    plt.title('Model Loss')
    plt.plot(train_loss , label = 'Train')
    plt.plot(validation_loss , label = 'Validation')
    plt.legend()
    plt.xlabel('EPOCH')
    plt.ylabel('Loss')
    plt.show()
    
    
# function to visualize model performance
def visualize(data_patch , model , train = True):
    
    #predict over data
    pred_patch = model.predict( data_patch )
    fig , ax = plt.subplots( 4 , 4 , figsize = (20 , 8))
    if train:
        plt.suptitle('Model Evaluation on Train Data' , size = 18)
    else:
        plt.suptitle('Model Evaluation on Validation' , size = 18)
    for i in range(8):
        
        plt.subplot(4 , 4 , i*2 + 1 )
        plt.imshow(data_patch[i])
        plt.title('Image')
        plt.axis('off')
        plt.subplot(4 , 4 , i*2 + 2 )
        plt.imshow(pred_patch[i])
        plt.title('Predicted')
        plt.axis('off')
        
    plt.show()
