import numpy as np
import matplotlib.pyplot as plt
import json as js
import random as rd
import cv2
import os
import time as t




class BDIGenerator:

    def __init__(self, input_sh=(640, 640, 3), data_folder=None, usg="train"):

        self.input_sh = input_sh
        self.data_folder = data_folder
        if self.data_folder is None:
            self.data_folder = "c:\\Users\\1\\Desktop\\drone_solution_meta\\data"
        
        self.data_folder = os.path.join(self.data_folder, usg)
        json_file = os.path.join(self.data_folder, "_annotations.coco.json")
        with open(json_file, "r") as json:
            self.json_specs = js.load(json) 

        self.images_json = self.json_specs["images"]
        self.annotations_json = self.json_specs["annotations"]
    
    def __collect_data__(self, image_path):

        remote_path = os.path.join(self.data_folder, image_path)

        image = cv2.imread(remote_path)
        if self.input_sh is not None:
            image = cv2.resize(image, self.input_sh[:-1])
        
        image_id = self.images_json[image_path]["id"]
        image_annot = self.annotations_json[str(image_id)]

        image_bb = image_annot["bbox"]
        image_bba = image_annot["area"] 
        cll_label = image_annot["category_id"] 
        
        return (image, image_bb, image_bba, cll_label)

    def __iter__(self):

        while True:
            
            
            rd_path = rd.choice(os.listdir(self.data_folder))
            if rd_path == "_annotations.coco.json":
                rd_path = rd.choice(os.listdir(self.data_folder))

            data = self.__collect_data__(image_path=rd_path)

            yield data
    
    def __next__(self):

        random_img = rd.choice(os.listdir(self.data_folder))
        path = os.path.join(self.data_folder, random_img)
        data = self.__collect_data__(image_path=path)

        return data


class IsGenerator:

    def __init__(self, input_sz, images_path, segmentation_path):

        self.input_sz = input_sz
        self.images_path = images_path
        self.segmentation_path = segmentation_path
    
    def __collect_data__(self):

        
        random_idx = np.random.randint(0, len(os.listdir(self.images_path)))
        random_image = cv2.imread(os.path.join(self.images_path, os.listdir(self.images_path)[random_idx]))
        random_segim = cv2.imread(os.path.join(self.segmentation_path, os.listdir(self.segmentation_path)[random_idx]))
        
        random_image = np.asarray(random_image)
        random_segim = np.asarray(random_segim)

        random_image = cv2.resize(random_image, self.input_sz)
        random_segim = cv2.resize(random_segim, self.input_sz)

        

        return random_image, random_segim

    def __iter__(self):

        while True:
            
            image, segim = self.__collect_data__()
            yield image, segim
    
    def __next__(self):

        image, segim = self.__collect_data__()
        return image, segim 

        
        
if __name__ == "__main__":


    s_time = t.time()
    generator = BDIGenerator()
    images = []
    bb = []
    bba = []
    cll_labels = []
    for (sample_n, sample) in enumerate(iter(generator)):

        if sample_n == 1000:
            break

        images.append(sample[0])
        bb.append(sample[1])
        bba.append(sample[2])
        cll_labels.append(sample[3])
    
    images = np.asarray(images)
    bb = np.asarray(bb)
    bba = np.asarray(bba)
    cll_labels = np.asarray(cll_labels)
    t_end = t.time()

    print(images.shape, bb.shape, bba.shape, cll_labels.shape, t_end - s_time)

        


    
    
    
