import numpy as np
import matplotlib.pyplot as plt
import json as js
import cv2
import os

from tensorflow.keras.callbacks import Callback
from tensorflow.keras import Model

plt.style.use("dark_background")

class SegmentationModelCallback(Callback):

    def __init__(self, model, run_folder, train_data, samples_n=25):

        super().__init__()
        self.model = model
        self.conv_layers = [layer for layer in self.model.layers if "conv" in layer.name]
        self.out_model = Model(inputs=self.model.inputs, 
                               outputs=[layer.output for layer in self.conv_layers])
        
        self.train_data = train_data
        self.samples_n = samples_n
        self.run_folder = run_folder
        self.__check_path__(self.run_folder)
        
        self.activations_folder = os.path.join(self.run_folder, "activations")
        self.weights_folder = os.path.join(self.run_folder, "weights")
        self.epoch_folder = os.path.join(self.run_folder, "epoch")

        meta = [self.activations_folder, self.weights_folder, self.epoch_folder]
        for folder in meta:
            self.__check_path__(folder)

        self.epoch_n = 0
    
    def __check_path__(self, path):
        
        if not os.path.exists(path):
            os.mkdir(path)
        
    def on_train_begin(self, logs=None):

        random_idx = np.random.randint(0, self.train_data.shape[0], self.samples_n)
        samples = self.train_data[random_idx]
        activations = self.out_model.predict(samples)

        for layer_n, (weights, activation) in enumerate(zip([layer.get_weights[0] for layer in self.conv_layers], activations)):
            
            weights *= 255
            weights = weights.astype("int")
            activation *= 255
            activation = activation.astype("int")
            
            activation_path = os.path.join(self.activations_folder, f"layer{layer_n}.png")
            weights_path = os.path.join(self.weights_folder, f"layer{layer_n}.png")

            cv2.imwrite(activation_path, activation)
            cv2.imwrite(weights_path, weights)
    
    def on_epoch_end(self, logs=None):

        random_idx = np.random.randint(0, self.train_data, self.samples_n)
        samples_r = int(np.sqrt(self.samples_n))
        samples = self.train_data[random_idx]
        
        sample_n = 0
        epoch_path = os.path.join(self.epoch_folder, f"epoch{self.epoch_n}")
        fig, axis = plt.subplots(nrows=samples_r, ncols=samples_r)
        for i in range(axis.shape[0]):
            for j in range(axis.shape[1]):

                axis[i, j].imshow(samples[sample_n])
                sample_n += 1
    
        fig.savefig(fname=epoch_path)
        self.epoch_n += 1
    
    def on_train_end(self, logs=None):

        model_weights_f = os.path.join(self.run_folder, "weights.weights.h5")
        self.model.save_weights(filepath=model_weights_f)

        
        

        
        
