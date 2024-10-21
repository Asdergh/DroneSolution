import numpy as np
import matplotlib.pyplot as plt
import random as rd
import tensorflow as tf
import cv2
import os

from conv_detector import ConvDetector
from tensorflow.keras import Model, Sequential
from tensorflow import squeeze, reduce_mean, argmax, GradientTape



def video_capturing(detector, frames_n=200, images_sz=(128, 128), bb_color=(255, 0, 0)):

    cam = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter('C:\\Users\\1\\Desktop\\drone_solution\\tests\\detection_video.mp4', fourcc, 20.0, (128, 128))
    images_n = 0
    while True:
        
        if images_n == frames_n:
            break

        _, frame = cam.read()
        frame = cv2.resize(frame, images_sz)
        bb_pred, _ = detector.predict(np.expand_dims(frame, axis=0))
        bb_pred = bb_pred.astype("int")
        bb_pred = bb_pred[0]
        bb_pred[0] = bb_pred[0] - int(bb_pred[2] / 2)
        bb_pred[0] = bb_pred[1] - int(bb_pred[3] / 2)

        frame = cv2.rectangle(frame, bb_pred[:2], bb_pred[2:], bb_color)
        video_out.write(frame)
        images_n += 1
    
    
def grads_inference(inputs, model):

    
    with GradientTape() as gr_tape:
        conv_preds, logits_preds = model(inputs)
    
    grads = gr_tape.gradients(logits_preds, conv_preds)
    pool_grads = reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_preds @ logits_preds[:, tf.newaxis]
    
    return heatmap

    

    
    
if __name__ == "__main__":

    detector = ConvDetector(input_sh=(128, 128, 3), dp_rate=0.45)
    detector.load_weights(filepath="c:\\Users\\1\\Desktop\\drone_solution_meta\\models_weights\\yolo_weights.weights.h5")
    video_capturing(detector=detector)
    
        
    
    
    
    

        
    