import numpy as np
import matplotlib.pyplot as plt
import json as js
import os

from tensorflow.keras.layers import Layer, Concatenate, Add, Multiply, Activation, BatchNormalization, Input, Dense, Conv2D, UpSampling2D, GlobalAveragePooling2D, AvgPool2D
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model, Sequential
from tensorflow import Module, GradientTape


class SeBlock(Module):

    def __init__(self, ch, ratio):

        gap_layer = GlobalAveragePooling2D()
        fe_layer = Dense(units=(ch // ratio), activation="relu")
        out_layer = Dense(units=ch, activation="sigmoid")

        self.layers = [gap_layer, fe_layer, out_layer]
    
    def __call__(self, inputs):

        x = inputs
        for layer in self.layers:
            x = layer(x)
        
        x = Multiply()([x, inputs])
        return x


class EncodingBlock(Module):

    def __init__(self, filters):

        norm_layer = BatchNormalization()
        act_layer = Activation("relu")
        down_layer = AvgPool2D(pool_size=2)

        norm1_layer = BatchNormalization()
        act1_layer = Activation("relu")
        conv_layer = Conv2D(filters=filters, kernel_size=3, padding="same", strides=1)
        se_layer = SeBlock(ch=filters, ratio=2)

        self.layers = [norm_layer, act_layer, down_layer, norm1_layer, act1_layer, conv_layer, se_layer]
    
    def __call__(self, inputs):

        x = inputs
        for layer in self.layers:
            x = layer(x)
        
        return x

class DecodingLayer(Module):

    def __init__(self, filters):

        norm_layer = BatchNormalization()
        act_layer = Activation("relu")
        up_layer = UpSampling2D(size=2)

        norm1_layer = BatchNormalization()
        act1_layer = Activation("relu")
        conv_layer = Conv2D(filters=filters, kernel_size=3, padding="same", strides=1)
        
        se_layer = SeBlock(ch=filters, ratio=2)
        self.layers = [norm_layer, act_layer, up_layer, norm1_layer, act1_layer, conv_layer, se_layer]
    
    def __call__(self, inputs):

        x = inputs
        for layer in self.layers:
            x = layer(x)
        
        return x

class CodeBlock(Module):

    def __init__(self, filters):

        norm_layer = BatchNormalization()
        relu_layer = Activation("relu")
        conv_layer = Conv2D(filters=filters, kernel_size=3, strides=1, padding="same")
        se_layer = SeBlock(ch=filters, ratio=2)
    
    def __call__(self, inputs):

        x = inputs
        for layer in self.layers:
            x = layer(x)
        
        return x


class IsFusionBlock(Module):

    def __init__(self, filters):

        conct_layer = Concatenate()

        norm_layer = BatchNormalization()
        act_layer = Activation("relu")
        conv_layer = Conv2D(filters=filters, kernel_size=1, strides=1, padding="same")
        act1_layer = Activation("sigmoid")

        add_layer = Add()
        self.layers = [conct_layer, norm_layer, act_layer, conv_layer, act1_layer, add_layer]
    
    def __call__(self, inputs):

        input1, input2 = inputs
        x = self.layers[0]([input1, input2])
        for layer in self.layers[1:-1]:
            x = layer(x)
        
        x = self.layers[-1]([x, input1])
        return x


class BSS(Model):

    def __init__(self, input_sh=(128, 128, 3), **kwargs):

        super().__init__(**kwargs)
        self.input_sh = input_sh
        self.model = self.__build__()

    def __build__(self):

        input_layer = Input(shape=self.input_sh)
        conv_layer = Conv2D(filters=128, kernel_size=3, padding="same", strides=1)(input_layer)
        
        encoding1_layer = EncodingBlock(filters=128)(conv_layer)
        encoding2_layer = EncodingBlock(filters=64)(encoding1_layer)
        encoding3_layer = EncodingBlock(filters=32)(encoding2_layer)

        decoding1_layer = DecodingLayer(filters=32)(encoding3_layer)
        decoding2_layer = DecodingLayer(filters=64)(decoding1_layer)
        decoding3_layer = DecodingLayer(filters=128)(decoding2_layer)

        out_layer = Conv2D(filters=self.input_sh[-1], kernel_size=1, padding="same", strides=1)(decoding3_layer)
        out_layer = Activation("sigmoid")(out_layer)

        model = Model(inputs=input_layer, outputs=out_layer)
        return model
    
    def compile(self, optimizer, loss):

        super().compile()
        self.optimizer = optimizer
        self.loss = loss
        self.loss_tracker = Mean(name="segmentation loss")
    
    @property
    def metrics(self):
        return [
            self.loss_tracker
        ]

    def train_step(self, inputs):

        images_t, images_l = inputs
        with GradientTape() as gr_tape:

            segmented_images = self.model(images_t)
            loss = self.loss(segmented_images, images_l)
        
        train_vars = self.model.trainable_variables
        grads = gr_tape.gradient(loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))

        self.loss_tracker.update_state(loss)
        return {
            "segmentation_loss": self.loss_tracker.result()
        }
    
    def call(self, inputs):

        out = self.model(inputs)
        return out



if __name__ == "__main__":

    random_data = np.random.normal(0, 0.12, (100, 128, 128, 3))
    seg_model = BSS(input_sh=(128, 128, 3))
    seg_model.compile(optimizer=Adam(), loss=MeanSquaredError())
    seg_model.fit(random_data, random_data, epochs=1, batch_size=32)
    
    plt.style.use("dark_background")
    fig, axis = plt.subplots(ncols=2)
    
    random_image = random_data[0]
    random_image = np.expand_dims(random_image, axis=0)
    seg_sample = seg_model.predict(random_image)

    seg_sample = np.squeeze(seg_sample)
    random_image = np.squeeze(random_image)

    axis[0].imshow(random_image)
    axis[1].imshow(seg_sample)

    plt.show()

    
    