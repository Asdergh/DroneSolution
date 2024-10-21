import numpy as np
import matplotlib.pyplot as plt
import random as rd

from tensorflow.keras.layers import Layer, Input, Conv2D, BatchNormalization, Activation, Concatenate, MaxPool2D, UpSampling2D, GlobalAveragePooling2D, Multiply, Dense
from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError
from tensorflow.keras.metrics import Mean
from tensorflow.math import reduce_mean
from tensorflow.keras import Model, Sequential
from tensorflow import  Module

from layers import *


class SeBlock(Module):

    def __init__(self, ch, ratio):

        gap_layer = GlobalAveragePooling2D()
        fe_layer = Dense(units=ch // ratio, activation="linear")
        out_layer = Dense(units=ch, activation="sigmoid")
        mul_layer = Multiply()
        
        self.layers = [gap_layer, fe_layer, out_layer, mul_layer]
    
    def __call__(self, inputs):

        x = inputs
        for layer in self.layers[:-1]:
            x = layer(x)
        
        x = self.layers[-1]([inputs, x])
        return x
    

input_sh = (128, 128, 3)
input = Input(shape=input_sh)
conv_layer = Conv2D(filters=32, kernel_size=3, strides=2, padding="same", activation="tanh")(input)
se_layer = SeBlock(ch=32, ratio=2)(conv_layer)
se_layer = SeBlock(ch=32, ratio=2)(se_layer)
se_layer = SeBlock(ch=32, ratio=2)(se_layer)

model = Model(inputs=input, outputs=se_layer)
print(model.summary())



        
        

        

        