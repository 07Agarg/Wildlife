import json
import os
import yaml
import requests
from PIL import Image
from tqdm import tqdm


# PATH to the train and validation labels in JSON format
#train_json_path = "../../dataset/labels/train.json"
#valid_json_path = "../../dataset/labels/validation.json"
train_json_path = "/home/ashimag/wii_data_species_2022/labels/wii_aite_2022_0.8.json"

# class xcentre ycentre width height  <-- normalised values
def bbox_to_yolo(bbox, width, heigth):
    if len(bbox) == 0:
        return "0.5 0.5 0.9999 0.9999"
    x_centre = str((bbox[0] + bbox[2]) / (2 * width))
    y_centre = str((bbox[1] + bbox[3]) / (2 * heigth))
    wi = str(bbox[2] / width)
    hi = str(bbox[3] / heigth)
    return x_centre + " " + y_centre + " " + wi + " " + hi


if __name__ == "__main__":
    confirmation = "x"
    confidence=0.5
    
    # path = "../../dataset/labels/train/"  # update the path where the training labels txt files will be saved
    path= "/home/ashimag/wii_data_species_2022/labels/wii_all_labels/"
    f = open(train_json_path)
       

    if os.path.isdir(path) and confirmation != "y":
        confirmation = input("Path '%s" %path + "' already exists. Overwrite? (y/n)")
        
    elif not os.path.isdir(path):
        os.mkdir(path)

    print("Transforming labels!")


    check_set = set()
    training_data = json.load(f)
    count=0
    for i, ann in enumerate(training_data["images"]):
        
        if "detections" not in ann:
            print('No detection in json')
            continue
        list_bbox= ann["detections"]
        if len(list_bbox)==0:
            print('Empty Bbox')
            
            continue
        else:
    
            im_path = ann["file"]
            category_id = str(im_path.split('/')[-2])
            im = Image.open(requests.get(im_path, stream=True).raw if im_path.startswith('http') else im_path)  # open
            width, height = im.size  # image size
            dir= os.path.join(path, category_id)
            if not dir:
                os.makedirs(dir)
            name = str(im_path.split('/')[-1].split('.jpg')[0])
            file_name = dir + name + ".txt"
            for item in list_bbox:
                if item['conf'] >0.4 and item['category']==1: 
                    print('Saving for animal category')
                    bbox = item['bbox']  # "bbox": [x,y,width,height]
                    # imag = training_data['images'][i]
                    content = category_id + " " + bbox_to_yolo(bbox, width, height)
                    if name in check_set:
                        file = open(file_name, "a")
                        file.write("\n")
                        file.write(content)
                        file.close()
                    else:
                        check_set.add(name)
                        file = open(file_name, "w")
                        file.write(content)
                        file.close()
                elif item['conf'] > 0.8 and (item['category'] == 2 or item['category'] == 3): 
                    print('Saving for person/vehicle category')
                    bbox = item['bbox']  # "bbox": [x,y,width,height]
                    # imag = training_data['images'][i]
                    content = category_id + " " + bbox_to_yolo(bbox, width, height)
                    if name in check_set:
                        file = open(file_name, "a")
                        file.write("\n")
                        file.write(content)
                        file.close()
                    else:
                        check_set.add(name)
                        file = open(file_name, "w")
                        file.write(content)
                        file.close()

    print(f'count is {count}')
