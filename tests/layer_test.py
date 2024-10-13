import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.layers import Input, Layer, Dense, Conv2D, Conv2DTranspose, Lambda, Concatenate
from tensorflow.keras import Model, Sequential



class Split(Layer):

    def __init__(self, splits_s_or_n=2, axis=0):

        super(Split, self).__init__()
        self.splits_s_or_n = splits_s_or_n
        self.axis = axis
    
    def call(self, input):

        split = tf.split(input, self.splits_s_or_n, axis=self.axis)
        output = tf.reduce_mean(split, axis=0)
        return output

class MeanE(Layer):

    def __init__(self, axis):

        super(MeanE, self).__init__()
        self.axis = axis
    
    def call(self, input):

        mean = tf.reduce_mean(input, axis=self.axis)
        expand = tf.expand_dims(mean, axis=self.axis)
        return expand


input_sh = (128, 128, 3)
input = Input(shape=input_sh)

conv_layer = Conv2D(filters=32, kernel_size=3, strides=1, activation="relu", padding="same")(input)
fnn_layer = Dense(units=12, activation="sigmoid")(conv_layer)
print(fnn_layer.shape)
split_layer = Split(axis=-1)(fnn_layer)
test_model = Model(inputs=input, outputs=split_layer)


random_noise = np.random.normal(0.12, 12.4, (100, 128, 128, 3))
model_out = test_model.predict(random_noise)

plt.style.use("dark_background")
fig, axis = plt.subplots()
axis.imshow(model_out[0][:, :, :3])
plt.show()