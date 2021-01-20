import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools

"""

This code is to convert image/semantic/instance(isi) annotation type (used in SceneGAN) to coco type.

isi means for each image file you have two additional annotation files, 
one is semantic segmentation(1 channel image format), 
the other is instance segmentation(1 channel image format).
For example: ( img1.jpg, sem1.png, ins1.png ) ( img2.jpg, sem2.png, ins2.png ) .... 
and have to be spatially aligned. 

For sem.png each number represent a class(not aware of instance)
For ins.png each number represent an instance(not aware of class) 

According to your dataset, you may want to rewrite get_mapping() and get_files()

NOTE: during process we assume 0 as unlabled class, and will not process it 

I ran into some issues when setup pycococreatortools, so I directly download the folder pycococreatortools
from https://github.com/waspinator/pycococreator and put it here

"""




def get_mapping():     
    """

    THIS FUNCTION SHOULD BE OVERWRITEN ACCORDING TO YOUR OWN DATA.
    
    The goal of this function is to return a mapping dict where key is 
    semantic class and value is name. {1:people, 2:car, .....}    
    """
    
    import pandas as pd
    data = pd.read_csv('/home/yuheng/Downloads/ADE20K_2016_07_26/objectInfo150.txt',sep='\t',lineterminator='\n') 
    mapping = {}
    for i in range(150):
        line = data.loc[i]
        mapping[ int(line['Idx']) ] = line['Name']
    
    return mapping



def get_files():
    """
    
    THIS FUNCTION SHOULD BE OVERWRITEN ACCORDING TO YOUR OWN DATA.
    
    The goal of this function is to return three lists, each contains
    all files(path) in given dir. Their length has to be same and they
    HAVE TO follow the same order.        
    """

    img_dir = '../ADE20K_2016_07_26/full_data/images/validation/'
    sem_dir = '../ADE20K_2016_07_26/full_data/annotations/validation/'
    ins_dir = '../ADE20K_2016_07_26/full_data/annotations_instance/validation/'

    img_files = os.listdir(img_dir)
    sem_files = os.listdir(sem_dir)
    ins_files = os.listdir(ins_dir)
    
    img_files = [  os.path.join(img_dir,item)  for item in img_files  ]
    sem_files = [  os.path.join(sem_dir,item)  for item in sem_files  ]
    ins_files = [  os.path.join(ins_dir,item)  for item in ins_files  ]
    
    img_files.sort()
    sem_files.sort()
    ins_files.sort()
    
    return img_files, sem_files, ins_files 



def get_categories(mapping):
    """
    The input mapping should be output from get_mapping function.
    and it will return a categories list, which is part of coco file.
    """
    categories = []
    
    for idx, name in mapping.items():        
        temp = {'id':idx, 'name':name, 'supercategory':'NA'}
        categories.append(temp)
        
    return categories


INFO = {
    "description": "NA",
    "url": "NA",
    "version": "0.1.0",
    "year": 2020,
    "contributor": "Yuheng",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "NA",
        "url": "NA"
    }
]


#########################################################################################################





def fire(mapping, img_files, sem_files, ins_files, output_file):

    CATEGORIES = get_categories(mapping)

    coco_output = { "info": INFO,
                    "licenses": LICENSES,
                    "categories": CATEGORIES,
                    "images": [],
                    "annotations": [] }


    ###### the rest of code is to fill in "images" and "annotations" #####

    image_id = 1
    segmentation_id = 1

    for img_file, sem_file, ins_file in zip(img_files, sem_files, ins_files):
        print(image_id)
        
        img = Image.open(img_file)
        image_info = pycococreatortools.create_image_info(image_id, os.path.basename(img_file), img.size)
        coco_output["images"].append(image_info)
        
        # then for this image, process each instance segmentation 
        sem = np.array( Image.open(sem_file) )
        ins = np.array( Image.open(ins_file) )
        
        for j in range(ins.max()+1):
            this_ins = (ins==j)*1
            this_ins_sem = this_ins*sem  # In most cases this should have 2 values: 0 and sem class. (could be 0 then just 1 value)
            unique_value = np.unique(this_ins_sem)
            
            if (len(unique_value)>2):
                print('%sth instance of image %s has more than one semantic label, and it was passed' % (j,image_id))
            else:
                # get class lable of this instance  
                class_label = int(unique_value[-1])   
                
                if class_label !=0: 
                    # we start to process this instance 
                    category_info = {'id': class_label, 'is_crowd': False}
                    binary_mask = this_ins.astype('uint8')
                    annotation_info = pycococreatortools.create_annotation_info( segmentation_id, image_id, category_info, binary_mask, img.size, tolerance=2)
        
                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)
                    segmentation_id = segmentation_id + 1
                    
        image_id = image_id + 1

    with open(output_file, 'w') as output_json_file:
        json.dump(coco_output, output_json_file)











if __name__ == '__main__':


    mapping = get_mapping()

    output_file = 'output.json'
    assert not os.path.exists(output_file), 'Output file exists'

    img_files, sem_files, ins_files = get_files()
    assert len(img_files) == len(sem_files) == len(ins_files)

    fire(mapping, img_files, sem_files, ins_files, output_file)