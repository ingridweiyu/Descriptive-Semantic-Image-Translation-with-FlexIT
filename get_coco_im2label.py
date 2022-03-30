import cv2
import json
import numpy as np
import os


# If loading pre-computed im2label JSON, set this to False.
# To recompute, set to True.
recompute_im2label = False



thing_val2017_json = open('data/cocostuff/annotations/instances_val2017.json')
thing_val2017 = json.load(thing_val2017_json)

stuff_val2017_json = open('data/cocostuff/annotations/stuff_val2017.json')
stuff_val2017 = json.load(stuff_val2017_json)

thing_annotations = thing_val2017['annotations']
stuff_annotations = stuff_val2017['annotations']



# Creates a dictionary mapping category ID to category name and supercategory
# Examples of a key-value pair in dictionary:
# {2: ['bicycle', 'vehicle]}
# The key is 2, referring to the category ID.
# The value is a list, where the first element is the category name, and
# the second value is the supercategory.

categories_raw = thing_val2017['categories']
categories_raw.extend(stuff_val2017['categories'])
categories = {}
for category in categories_raw:
    supercategory = category['supercategory']
    category_id = category['id']
    name = category['name']
    
    categories[category_id] = [name, supercategory]





# Creates a dictionary mapping image_id to labels
# Example of key-value pair in dictionary:
# {12345: [
#           [[121,200,345,109], 1],
#           [[491,103,491,21], 20]
#         ]
# }
# The key is 12345, referring to the image 000000012346.jpg.
# The value is a list of sub-lists.
# The first element in the sub-list is the bounding box [xmin, ymin, width, height].
# The second element in the sub-list is the class ID.

if recompute_im2label:
    
    im2label = {}
    i = 1
    total = len(thing_annotations)
    for ann in thing_annotations:
        print("Thing annotation", i, "/", total)
        image_id = str(ann['image_id'])
        bbox = ann['bbox']
        category_id = ann['category_id']
        
        try:
            label_list = im2label[image_id]
            label_list.append([bbox, category_id])
        except KeyError:
            im2label[image_id] = [[bbox, category_id]]
        
        i += 1
            
    i = 1
    total = len(stuff_annotations)
    for ann in stuff_annotations:
        print("Stuff annotation", i, "/", total)
        image_id = str(ann['image_id'])
        bbox = ann['bbox']
        category_id = ann['category_id']
        
        try:
            label_list = im2label[image_id]
            label_list.append([bbox, category_id])
        except KeyError:
            im2label[image_id] = [[bbox, category_id]]
        
        i += 1
    
    with open('data/cocostuff/val2017_im2label.json', 'w') as f:
        json.dump(im2label, f)

else:
    im2label = json.load(open('data/cocostuff/val2017_im2label.json'))




def visualize(image_id, image_dir, im2label_dict):
    '''
    Visualizes the image with bounding boxes of objects

    Parameters
    ----------
    image_id : str
        Image ID number
    image_dir : str
        Path to directory containing image. Current directory is represented
        by the empty string.
    im2label_dict : dict
        Dictionary mapping image_id to labels, where each label is a list of
        tuples. The first element of the tuple is a list of 4 coordinates
        representing the bounding box. The second element of the tuple is the
        class ID of the object.

    Returns
    -------
    None.
    '''
    filename = ''
    for i in range(12 - len(str(image_id))):
        filename += '0'
    filename += str(image_id) + '.jpg'
    
    image_path = os.path.join(image_dir, filename)
    image = cv2.imread(image_path)
    
    for label in im2label_dict[image_id]:
        bb = label[0]
        xmin = bb[0]
        ymin = bb[1]
        width = bb[2]
        height = bb[3]
        xmax = xmin + width
        ymax = ymin + height
        
        start_point = int(xmin), int(ymin)
        end_point = int(xmax), int(ymax)
        
        BGR = np.random.randint(0, 256, 3)
        color = (int(BGR[0]), int(BGR[1]), int(BGR[2])) # BGR color
        thickness = 2
        cv2.rectangle(image, start_point, end_point, color, thickness)
    
    cv2.imshow(filename, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 


