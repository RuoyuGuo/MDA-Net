from utils_de import *
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing.pool import Pool
import cv2 as cv

import csv
import argparse
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

dsize = (512,512)


def process(image_list, resized_path):  
    #resized_path = RESIZED_IMG_PATH[i_path]
    
    for image_path in image_list: 
        name = image_path.split('/')[-1].split('.')[0] + '.png'
        dst_image_path = os.path.join(resized_path, 'image', name)
        dst_mask_path = os.path.join(resized_path, 'mask', name)

        #print(dst_image_path, dst_mask_path)

        try:
            #print('1111')
            img = imread(image_path)
            img, mask = preprocess(img)
            
#             #if not 512 size image , comment this part
#             h, w,_ = img.shape
#             centre_h = h//2 - 256
#             centre_w = w//2 - 256
            
#             #if not 512 size image , comment this part
#             crop_img = img[int(y):int(y+h), int(x):int(x+w)]
            
#             #if not 512 size image , comment this part
#             img = img[centre_h:centre_h+512,centre_w:centre_w+512]
#             mask = mask[centre_h:centre_h+512,centre_w:centre_w+512]
            
            
            img = cv.resize(img, dsize)
            mask = cv.resize(mask, dsize)
            #print(img.shape, mask.shape)
            imwrite(dst_image_path, img)
            imwrite(dst_mask_path, mask)
        except:
            print(image_path)
            continue

        # #print(image_path)
        # #break
        # img = imread(image_path)
        # #print(img.shape)
        # img, mask = preprocess(img)

        # break
        # # img = cv.resize(img, dsize)
        # # mask = cv.resize(mask, dsize)
        # # #print(img.shape, mask.shape)
        # # imwrite(dst_image_path, img)
        # # imwrite(dst_mask_path, mask)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--dsize', type=int, required=True)
    
    args = parser.parse_args()


    ORIGINAL_IMG_PATH = args.input_path
    RESIZED_IMG_PATH = args.out_path
    dsize = (args.dsize, args.dsize)
    
    #for e in RESIZED_IMG_PATH:
    if os.path.exists(RESIZED_IMG_PATH) is False:
        os.mkdir(RESIZED_IMG_PATH)
    if os.path.exists(os.path.join(RESIZED_IMG_PATH, 'image')) is False:
        os.mkdir(os.path.join(RESIZED_IMG_PATH, 'image'))
    if os.path.exists(os.path.join(RESIZED_IMG_PATH, 'mask')) is False:
        os.mkdir(os.path.join(RESIZED_IMG_PATH, 'mask'))
        

    #for i_oimg, e_oimg in enumerate(ORIGINAL_IMG_PATH):
    print(f'processing {ORIGINAL_IMG_PATH} set...')
    image_list = glob.glob(os.path.join(ORIGINAL_IMG_PATH, '*.jpeg')) + glob.glob(os.path.join(ORIGINAL_IMG_PATH, '*.jpg')) + glob.glob(os.path.join(ORIGINAL_IMG_PATH, '*.png'))
    #print(image_list)

    #print(image_list)
    
    patches = 16
    patch_len = int(len(image_list)/patches)
    filesPatchList = []
    for i in range(patches-1):
        fileList = image_list[i*patch_len:(i+1)*patch_len]
        filesPatchList.append(fileList)
    filesPatchList.append(image_list[(patches-1)*patch_len:])

    # mutiple process
    temp_param = list(zip(filesPatchList, \
                            [RESIZED_IMG_PATH] * len(filesPatchList)))
    pool = Pool(patches)
    pool.starmap(process, temp_param)
    pool.close()
    
    print()

