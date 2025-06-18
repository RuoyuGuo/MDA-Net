import os
import glob
from multiprocessing.pool import Pool
import json

import numpy as np
from tqdm import tqdm
from PIL import Image

from degrad_de_pce import *
from utils_de import imread, imwrite
import argparse

random.seed(2023)
np.random.seed(2023)
sizeX = 512
sizeY = 512

type_map = ['001', '010', '011', '100', '101', '110', '111'] 
num_type = 16
# '111' means: DE_BLUR, DE_SPOT, DE_ILLUMINATION


def generate_type_list(num_type):
    type_list = []
    if num_type >= len(type_map):
        for i in range(num_type):
            t = random.randint(0, len(type_map) - 1)
            type_list.append(type_map[t])
    else:
        for i in range(num_type):
            t = random.randint(0, len(type_map) - 1)
            type_list.append(type_map[t])
    return type_list


def process(image_name_list, resized_img_path, degraded_img_path):  

    for image_name in tqdm(image_name_list): 
        image_name =  image_name.split('/')[-1]
        image_path = os.path.join(resized_img_path, 'image', image_name)
        mask_path = os.path.join(resized_img_path, 'mask', image_name)
        name = image_name.split('.')[0]
        
        img = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize((sizeX, sizeY), Image.Resampling.BICUBIC)
        mask = np.expand_dims(mask, axis=2)
        mask = np.array(mask, np.float32).transpose(2, 0, 1)/255.0
        
        type_list = generate_type_list(num_type)
        for i, t in enumerate(type_list):
            r_img, r_params = DE_process(img, mask, sizeX, sizeY, t)
            dst_img = os.path.join(degraded_img_path, 'de_image', f'{name}_{i}.png')
            imwrite(dst_img, r_img)
            param_dict = {
                'name': image_name,
                'type': t,
                'params': r_params
            }
            with open(os.path.join(degraded_img_path, 'de_js_file', f'{name}_{i}.json'), 'w') as json_file:
                json.dump(param_dict, json_file)
            
        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--dsize', type=int, required=True)
    args = parser.parse_args()
    
    RESIZED_IMG_PATH  = args.input_path
    DEGRADED_IMG_PATH = args.out_path
    sizeX = args.dsize
    sizeY = args.dsize
    
    #create folder if not exist
    if os.path.exists(DEGRADED_IMG_PATH) is False:
        os.mkdir(DEGRADED_IMG_PATH)
    if os.path.exists(os.path.join(DEGRADED_IMG_PATH, 'de_image')) is False:
        os.mkdir(os.path.join(DEGRADED_IMG_PATH, 'de_image'))
    if os.path.exists(os.path.join(DEGRADED_IMG_PATH, 'de_js_file')) is False:
        os.mkdir(os.path.join(DEGRADED_IMG_PATH, 'de_js_file')) 

    #degrad image in each folder
    #for i_rimg, e_rimg in enumerate(RESIZED_IMG_PATH):
    print(f'processing {RESIZED_IMG_PATH} set...')
    image_name_list = os.path.join(RESIZED_IMG_PATH, 'image')
    image_name_list = glob.glob(os.path.join(image_name_list, '*.png'))
    print(len(image_name_list))
    
    process(image_name_list, RESIZED_IMG_PATH, DEGRADED_IMG_PATH)

    print()
