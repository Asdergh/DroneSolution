import numpy as np
import matplotlib.pyplot as plt
import random as rd
import tensorflow as tf

from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input, Conv2D, LayerNormalization, Dropout, Activation, Add, Concatenate, Multiply, Flatten, Dense, AvgPool2D
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.metrics import Mean, MeanIoU
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import Model, Sequential

from tensorflow.math import reduce_mean, reduce_max
from tensorflow.math import exp, sin, cos, minimum, maximum
from tensorflow import GradientTape, py_function, argmax

from generators import BDIGenerator






class ResConv2D(Layer):

    def __init__(self, filters, kernel_size, padding, activation_fn, strides, dropout_rate=0.56):

        
        super(ResConv2D, self).__init__()
        conv = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides)
        dropout = Dropout(rate=dropout_rate)
        norm = LayerNormalization()
        activation = Activation(activation_fn)
        add = Add()

        self.layers = [conv, dropout, norm, activation, add]
    
    def __call__(self, input):

        x = input
        for layer in self.layers[:-1]:
            x = layer(x)
        
        x = self.layers[-1]([x, input])
        return x

class Conv2DBlock(Layer):

    def __init__(self, filters, kernel_size, padding, activation_fn, strides, layers_n=3, dropout_rate=0.45):

        super(Conv2DBlock, self).__init__()
        self.layers = []
        for _ in range(layers_n):

            conv = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides)
            dropout = Dropout(rate=dropout_rate)
            normalization = LayerNormalization()
            activation = Activation(activation_fn)
            
            self.layers += [conv, dropout, normalization, activation]
    
    def __call__(self, input):

        x = input
        for layer in self.layers:
            x = layer(x)
        
        return x



class ConvDetector(Model):

    def __init__(self, input_sh, dp_rate, num_cl=None, **kwargs):

        super().__init__(**kwargs)
        self.num_cl = num_cl
        if self.num_cl is None:
            self.num_cl = 1

        self.input_sh = input_sh
        self.dp_rate = dp_rate
        self.model = self.__model__()
        

        grad_input = Input(shape=input_sh)
        self.grad_model = Model(inputs=grad_input, 
                           outputs=[self.model.get_layer("last_conv_layer").output, self.model.output[1]])
        
    
    def compile(self, optimizer, loss_fn, metrics=None):

        super().compile()
        self.optimizer = optimizer
        self.bb_loss_fn = loss_fn[0]
        self.bba_loss_fn = loss_fn[1]
        self.cll_loss_fn = loss_fn[2]

        if metrics is not None:
            self.bb_loss_tracker, self.cll_loss_tracker = metrics
        
        else:

            self.bb_loss_tracker = Mean(name="bounding_box_loss")
            self.cll_loss_tracker = Mean(name="classification_loss")

    
    
    
    @property
    def metrics(self):

        return [
            self.bb_loss_tracker,
            self.cll_loss_tracker
        ]
    
    
    def __model__(self):

        input_layer = Input(shape=self.input_sh)
        conv_block_conf = {"kernel_size": 3, "strides": 1, "padding": "same", "layers_n": 2, "activation_fn": "tanh"}
        res_block_conf = {"kernel_size": 3, "strides": 1, "padding": "same", "activation_fn": "tanh"}
        
        conv_model = Sequential(layers=[
            Conv2DBlock(filters=64, **conv_block_conf),
            ResConv2D(filters=64, **res_block_conf),
            AvgPool2D(pool_size=2),
            Conv2DBlock(filters=32, **conv_block_conf),
            ResConv2D(filters=32, **res_block_conf),
            AvgPool2D(pool_size=2),
           
        ])(input_layer)
        
        last_conv_layer = Conv2D(filters=32, kernel_size=3, strides=1, padding="same", name="last_conv_layer")(conv_model)
        linear_model = Sequential(layers=[
                Flatten(),
                Dense(units=64, activation="linear"),
                Dropout(rate=self.dp_rate),
                Dense(units=32, activation="linear"),
                Dropout(rate=self.dp_rate),
                Dense(units=12, activation="linear"),
                Dropout(rate=self.dp_rate)
        ])(last_conv_layer)

        out1_layer = Dense(units=4, activation="relu", name="outpur1")(linear_model)
        out2_layer = Dense(units=self.num_cl, activation="sigmoid", name="output2")(linear_model)

        model = Model(inputs=input_layer, outputs=[out1_layer, out2_layer])
        return model

        
    def train_step(self, inputs):

        images, features = inputs

        bb = features[0]
        cll_labels = features[1]

        with GradientTape() as gr_tape:

            bb_pred, cll_pred = self.model(images)
        
            bb_loss = self.bb_loss_fn(bb_pred, bb)
            cll_loss = self.cll_loss_fn(cll_pred, cll_labels)

            model_loss = bb_loss + cll_loss        

        t_variables = self.model.trainable_variables
        grads = gr_tape.gradient(model_loss, t_variables)
        self.optimizer.apply_gradients(zip(grads, t_variables))
        
        self.bb_loss_tracker.update_state(bb_loss)
        self.cll_loss_tracker.update_state(cll_loss)

        return {
            
            self.bb_loss_tracker.name: self.bb_loss_tracker.result(),
            self.cll_loss_tracker.name: self.cll_loss_tracker.result()
        }

    @tf.function
    def call(self, input):
        return self.model(input)
    

    
    
            
            


            

    

    


    
     

            





    


        
        
        
    
    