import numpy as np
import json as js
import json as js
import cv2
import os
import matplotlib.pyplot as plt


def mk_path(path):
        
    if not os.path.exists(path=path):
        os.mkdir(path=path)
    

def make_segmentation(datafolder_path, seg_datafolder_path):

    mk_path(seg_datafolder_path)
    color_bounds = [(22, 105, 155), (127, 211, 222)]

    
    for batch_folder in os.listdir(datafolder_path):

        if "txt" not in batch_folder:

            batch_path = os.path.join(datafolder_path, batch_folder)
            seg_batch_path = os.path.join(seg_datafolder_path, batch_folder)
            mk_path(seg_batch_path)

            annots_path = os.path.join(batch_path, "_annotations.coco.json")
            with open(annots_path, "r") as js_file:
                annots_buffer = js.load(js_file)
            
            for image_f in os.listdir(batch_path):
                
                if image_f != "_annotations.coco.json":
                    
                    try:
                        
                        image_path = os.path.join(batch_path, image_f)
                        image_path_seg = os.path.join(seg_batch_path, image_f)
                        image = cv2.imread(image_path)
                        
                        image_cll = annots_buffer["categories"][
                            annots_buffer["annotations"][
                                str(annots_buffer["images"][image_f]["id"])
                                ]["category_id"]
                            ]["name"]
                        
            
                            
                        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                        image_mask = cv2.inRange(image_hsv, color_bounds[0], color_bounds[1])
                        result = cv2.bitwise_and(image, image, mask=image_mask)
                        cv2.imwrite(image_path_seg, result)

                    except BaseException:
                        pass


if __name__ == "__main__":

    datafolder_path = "c:\\Users\\1\\Desktop\\drone_solution_meta\\data"
    seg_datafolder_path = "c:\\Users\\1\\Desktop\\drone_solution_meta\\segmentation_data"
    make_segmentation(datafolder_path=datafolder_path, seg_datafolder_path=seg_datafolder_path)




                