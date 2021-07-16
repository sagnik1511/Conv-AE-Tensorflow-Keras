# -*- coding: utf-8 -*-
"""

+--------------------------------+
| Convolutional Auto-Encoder     |
+--------------------------------+

The main model is stored here.

Copyright © Sagnik Roy  July, 2021.
All rights reserved.

"""

# Libraries
import tensorflow as tf
from tensorflow.keras.layers import (
    Input , Conv2D , Conv2DTranspose , MaxPooling2D , UpSampling2D
)
from tensorflow.keras.models import Model




# convAE model
class CAE():
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.c1 = Conv2D(16, (3, 3), activation='relu', padding='same')
        self.c2 = Conv2D(8, (3, 3), activation='relu', padding='same')
        self.c3 = Conv2D(8, (3, 3), activation='relu', padding='same')
        self.pool = MaxPooling2D((2, 2))
        self.ct1 = Conv2DTranspose(16, (3, 3), activation='relu', padding='same')
        self.ct2 = Conv2DTranspose(8, (3, 3), activation='relu', padding='same')
        self.ct3 = Conv2DTranspose(8, (3, 3), activation='relu', padding='same')
        self.ups = UpSampling2D((2, 2))
    
    def network(self):
        # Reseting backend
        tf.keras.backend.clear_session()
        
        # Model Architecture
        input_layer = Input(shape=self.input_shape)
        x = self.c1(input_layer)
        x = self.pool(x)
        x = self.c2(x)
        x = self.pool(x)
        x = self.c3(x)

        code_layer = self.pool(x)

        x = self.ct3(code_layer)
        x = self.ups(x)
        x = self.ct2(x)
        x = self.ups(x)
        x = self.ct1(x)
        x = self.ups(x)
        output_layer = Conv2D(3, (3, 3), padding='same', name="Output_layer")(x)
        
        # Functional Model
        model = Model(inputs = input_layer, outputs = output_layer, name = 'Poke_CAE')
        
        return model
