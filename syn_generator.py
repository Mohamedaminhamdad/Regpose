import random
import cv2 
import numpy as np
import os
from glob import glob
import argparse
"""
This Code is used to render background of synthetic Images    
"""
def load_background_image(im_real, bg_list):
    """laod background image and resize it to fit to synthetic image
    """
    h, w, c = im_real.shape
    bg_num = len(bg_list)
    idx = random.randint(0, bg_num - 1)
    bg_path = bg_list[idx]
    bg_im = cv2.imread(bg_path, cv2.IMREAD_COLOR)
    bg_h, bg_w, bg_c = bg_im.shape
    real_hw_ratio = float(h) / float(w)
    bg_hw_ratio = float(bg_h) / float(bg_w)
    if real_hw_ratio <= bg_hw_ratio:
        crop_w = bg_w
        crop_h = int(bg_w * real_hw_ratio)
    else:
        crop_h = bg_h 
        crop_w = int(bg_h / bg_hw_ratio)
    bg_im = bg_im[:crop_h, :crop_w, :]
    bg_im = cv2.resize(bg_im, (w, h), interpolation=cv2.INTER_LINEAR)
    return bg_im
        
def background(rgb, msk,bg_list):
    """
    change  synthetic image background image's background
    """
    bg_im = load_background_image(rgb, bg_list)
    msk = np.dstack([msk, msk, msk]).astype(np.bool)
    bg_im[msk] = rgb[msk]
    return bg_im
def add_background(args,images,mask):
    bg_list=glob(os.path.join(args.background,'*.jpg')) # list all background of VoC dataset 
    for image, mask in zip(sorted(images),sorted(mask)): 
        file_name=image[-25:]
        rgb=cv2.imread(image)
        mask=cv2.imread(mask,cv2.COLOR_BGR2GRAY)
        rgb_new=background(rgb,mask,bg_list) # New Synthetic image with background
        cv2.imwrite(file_name,rgb_new) # write  synthetic image with new background  
    
def create_synt(args):
    # Read the names of the images to annotations
    image_names_file=glob(os.path.join(args.train+'/images/data_syn/','*color.png'))
    images=image_names_file # List of synthetic images
    mask=glob(os.path.join(args.train+'/images/data_syn/','*label.png')) # Find  mask of objects containend in an Image
    add_background(args,images,mask) # Function to add background 
    return

if __name__ == "__main__":
    path = os.path.dirname(os.path.abspath(__file__))
    parser=argparse.ArgumentParser()
    parser.add_argument("--ouput_path",type=str,help="Output Path where to save the data", default=os.path.join(path,'train.json'))
    parser.add_argument("--real",type=str,help="Path to images and Bbox",default=os.path.join(path,'images'))
    parser.add_argument("--train",type=str,help="Path to training data",default=path)
    parser.add_argument("--model_path",type=str,help="Path to training data",default=os.path.join(path,'models'))
    parser.add_argument("--background",type=str,help="Path to training data",default=os.path.join(path,'JPEGImages'))
    parser.add_argument('--save_path',type=str,default=os.path.join(path,'data_syn'))

    args = parser.parse_args()
    create_synt(args)