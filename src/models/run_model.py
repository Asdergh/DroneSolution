import numpy as np
import cv2
import matplotlib.pyplot as plt


from segmentation_model import BSS
from custom_callbacks import SegmentationModelCallback



def __run__():
    
    cap = cv2.VideoCapture(0)
    model = BSS()
    model.model.load_weights("C:\\Users\\1\\Desktop\\drone_solution\\src\\models\\model_weights\\weights.weights.h5")

    while True:
        
        _, frame = cap.read()
        
        frame = cv2.resize(frame, (128, 128))
        frame = frame / 255.0
        frame = np.expand_dims(frame, axis=0)

        segmented_image = model.predict(frame)
        segmented_image = np.squeeze(segmented_image)
        
        plt.imshow(segmented_image, cmap="jet")
        plt.show()
        

        cv2.waitKey(1)


if __name__ == "__main__":

    __run__()