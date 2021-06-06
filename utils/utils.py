import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import os
import cv2

def filter_tbud_count(path_bud_info, img_file, thold=3):
    '''
    1296-10-81-b2-budInfo.txt
    orj-1317-37-45.jpg
    mask-1317-37-45.jpg
    '''
    buds = 0
    # Parse image ID: orj-1317-37-45.jpg -> 1317-37-45
    img_id = '-'.join((img_file.split('.')[0]).split('-')[1:])

    files_filtered = [(f.split('/')[-1], (f.split('/')[-1]).split('-')[-2]) for f in glob(path_bud_info+'*.txt') if '-'.join((f.split('/')[-1]).split('-')[:-2]) == img_id]
    
    if files_filtered:
        buds = int(files_filtered[0][1][1:])
    # return: [('1296-10-81-b2-budInfo.txt', 'b2')]
    return buds >= thold

def read_image(file, img_size, mask=False):

    # Parameters
    IMG_HEIGHT = img_size
    IMG_WIDTH = img_size

    if not mask:
        img = cv2.imread(file, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
        return img
    else:
        img_mask = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        img_mask = cv2.resize(img_mask, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
        img_mask = np.expand_dims(img_mask, axis=-1)
        return img_mask

def rename_orj(dir_img, id):
    '''
    Rename Orj Img files for easy convertion
    '''
    for f in os.listdir(dir_img):
        lst_f = [s.strip("0") for s in f.split('-')]
        name_new = 'orj-' + '-'.join([str(id)]+lst_f)
        os.rename(os.path.join(dir_img, f), os.path.join(dir_img, name_new))

def make_clean(img_mask, thold_area = 1000):
    if not img_mask is None:
        thold_area = thold_area
        img_cleaned = np.zeros_like(img_mask)
        
        # print('[make_clean] Num of contours: {}'.format(len(conts_mask)))

        kernel = np.ones((3, 3), np.uint8)
        
        img_mask = cv2.dilate(img_mask, kernel, iterations=1)
        img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        conts_mask, hierachy = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #conts_mask, hierachy = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print('[make_clean] Num of contours after MORPH_OPEN: {}'.format(len(conts_mask)))

        
        #conts_mask, hierachy = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #print('[make_clean] Num of contours after dilate: {}'.format(len(conts_mask)))

        for c in conts_mask:
            if cv2.contourArea(c) > thold_area:
                cv2.drawContours(img_cleaned, [c], -1, 255, -1)

        conts_mask, hierachy = cv2.findContours(img_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #print('[make_clean] Num of contours after AREA THOLD: {}'.format(len(conts_mask)))
        return img_cleaned
    else:
        print('[WARNING] check img_mask!')      
        
def mapper_image(img_ann, img_pred, fname, output_dir='.', clean=False, thold_area=100):
    added_image = img_ann.copy()#cv2.addWeighted(img_ann, 0.7, img_pred, 0.3, 0)
    
    # makepred clean
    # get connetcted comps
    if not img_pred is None:
        
        if clean:
            img_pred = make_clean(img_pred, thold_area)
        conts_mask, hierachy = cv2.findContours(img_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in conts_mask:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(added_image, (x-5,y-5), (x+w+5, y+h+5), (255, 0, 0), 3)
    # get rectangle of it
    # draw on ann
    dir_write = os.path.join(output_dir,'Prediction_mapped')
    if not os.path.exists(dir_write):
        os.makedirs(dir_write)
    #print(dir_write + fname)
    cv2.imwrite(os.path.join(dir_write, fname), added_image)
    
    return added_image