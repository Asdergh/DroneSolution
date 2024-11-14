import numpy as np
import cv2
import matplotlib.pyplot as plt


from segmentation_model import BSS
from custom_callbacks import SegmentationModelCallback



def make_vidbuffer(videopath=None):
    
    img_buffer = []
    cap = cv2.VideoCapture(videopath)
    if videopath is None:
        cap = cv2.VideoCapture()
    
    model = BSS()
    model.model.load_weights("C:\\Users\\1\\Desktop\\drone_solution\\src\\models\\model_weights\\weights.weights.h5")

    while True:
        
        acc, frame = cap.read()
        if acc:

            frame = cv2.resize(frame, (128, 128))
            frame = frame / 255.0
            
            segmented_image = model.predict(np.expand_dims(frame, axis=0))
            segmented_image = np.squeeze(segmented_image)
            segmented_image = cv2.resize(segmented_image, (620, 620))
            
            img_buffer.append(segmented_image)
        
        else:
            break
    
    return img_buffer

def write_video(vid_buffer, videpath=None):
    
    out = cv2.VideoWriter(videpath, cv2.VideoWriter_fourcc(*"DIVX"), 15, (620, 620))
    for frame in vid_buffer:
        
        frame = (frame * 256).astype(np.uint8)
        frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
        out.write(frame)
    


if __name__ == "__main__":

    img_buffer = make_vidbuffer(videopath="C:\\Users\\1\\Desktop\\drone_solution\\meta_data\\214912_small.mp4")
    write_video(vid_buffer=img_buffer, videpath="C:\\Users\\1\\Desktop\\drone_solution\\meta_data\\segmentation_video.mp4")
    
    
    