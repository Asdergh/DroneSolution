import numpy as np
import json as js
import os 


def formate_json(filepath):
    

    with open(filepath, "r") as json_file:

        js_buffer = js.load(json_file)
        images_buffer = js_buffer["images"]
        annots_buffer = js_buffer["annotations"]

        new_images_buffer = {}
        new_annots_buffer = {}
        for (image_conf, annot_conf) in zip(images_buffer, annots_buffer):

            new_images_buffer[image_conf["file_name"]] = image_conf
            new_annots_buffer[annot_conf["id"]] = annot_conf
        
        js_buffer["images"] = new_images_buffer
        js_buffer["annotations"] = new_annots_buffer

    with open(filepath, "w") as json_file:
        js.dump(js_buffer, json_file)

if __name__ == "__main__":

    all_tr_status = ["train", "test", "valid"]
    for tr_status in all_tr_status:

        filepath = f"C:\\Users\\1\\Desktop\\drone_solution\\data\\{tr_status}\\_annotations.coco.json"
        formate_json(filepath=filepath)
    
    
        
        

                  
        
    
    
