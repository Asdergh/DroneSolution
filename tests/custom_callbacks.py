import numpy as np
import matplotlib.pyplot as plt
import json as js
import cv2
import os

from tensorflow.keras.callbacks import Callback
from tensorflow.keras import Model
from tensorflow import reduce_mean

plt.style.use("dark_background")

class SegmentationModelCallback(Callback):

    def __init__(self, model, run_folder, train_data, samples_n=25):

        super().__init__()
        self.seg_model = model
        self.conv_layers = [layer for layer in self.seg_model.layers if "conv" in layer.name]
        self.out_model = Model(inputs=self.seg_model.inputs, 
                               outputs=[layer.output for layer in self.conv_layers])
        
        self.train_data = train_data
        self.samples_n = samples_n
        self.run_folder = run_folder
        self.__check_path__(self.run_folder)
        
        self.activations_folder = os.path.join(self.run_folder, "activations")
        self.epoch_folder = os.path.join(self.run_folder, "epoch")

        meta = [self.activations_folder, self.epoch_folder]
        for folder in meta:
            self.__check_path__(folder)

    
    def __check_path__(self, path):
        
        if not os.path.exists(path):
            os.mkdir(path)
        
    def on_train_begin(self, logs=None):

        random_idx = np.random.randint(0, self.train_data.shape[0], self.samples_n)
        samples = self.train_data[random_idx]
        activations = self.out_model.predict(samples)

        for layer_n, activation in enumerate(activations):
            
            
            activation = reduce_mean(activation, axis=0)
            activation = reduce_mean(activation, axis=-1)

            fig0, axis0 = plt.subplots()
            axis0.imshow(activation, cmap="jet")
            activation_path = os.path.join(self.activations_folder, f"layer{layer_n}.png")

            fig0.savefig(activation_path)
    
    def on_epoch_end(self, epoch, logs=None):

        random_idx = np.random.randint(0, self.train_data.shape[0], self.samples_n)

        samples_r = int(np.sqrt(self.samples_n))
        samples = self.train_data[random_idx]
        preds = self.seg_model.predict(samples)
        
        sample_n = 0
        epoch_path = os.path.join(self.epoch_folder, f"epoch{epoch}")
        fig, axis = plt.subplots(nrows=samples_r, ncols=samples_r)
        for i in range(axis.shape[0]):
            for j in range(axis.shape[1]):

                axis[i, j].imshow(preds[sample_n], cmap="jet")
                sample_n += 1
    
        fig.savefig(fname=epoch_path)
    
    def on_train_end(self, logs=None):

        model_weights_f = os.path.join(self.run_folder, "weights.weights.h5")
        entire_model_f = os.path.join(self.run_folder, "segmentation_model.keras")
        
        self.seg_model.save_weights(filepath=model_weights_f)
        self.seg_model.save(entire_model_f)
        

        
        

        
        
