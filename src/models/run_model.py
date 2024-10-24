import numpy as np
import cv2
import matplotlib.pyplot as plt


from segmentation_model import BSS
from custom_callbacks import SegmentationModelCallback



def __run__():
    
    cap = cv2.VideoCapture(0)
    model = BSS()
    model.model.load_weights("model_weights//weights.weights.h5")

    while True:
        
        _, frame = cap.read()
        
        frame = cv2.resize(frame, (128, 128))
        frame = frame / 255.0
        frame = np.expand_dims(frame, axis=0)

        segmented_image = model.predict(frame)
        segmented_image = np.squeeze(segmented_image)
        
        
        frame = cv2.resize(np.squeeze(frame), (640, 640))
        segmented_image = cv2.resize(segmented_image, (640, 640))
        cv2.imshow("frame", frame)
        cv2.imshow("segmentation", segmented_image)

        if cv2.waitKey(1) == ord("q"):
            break


if __name__ == "__main__":

    __run__()