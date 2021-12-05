import os
import argparse
import random
import scipy.io
from transforms3d.quaternions import mat2quat,qmult, qinverse
from transforms3d.euler import euler2quat
import numpy as np
import cv2
import json
from PIL import Image
from glob import glob
classes_all = ('background', '002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', \
                         '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana', '019_pitcher_base', \
                         '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', '040_large_marker', \
                         '051_large_clamp', '052_extra_large_clamp', '061_foam_brick')

def egocentric2allocentric(qt, T):
    """
    egocentric2allocentric from PoseCNN https://github.com/NVlabs/PoseCNN-PyTorch/tree/f7d28f2abd38fcfc297d8d421bb6b69248626eb5
    Transform egocentric rotation to allocentric rotation
    """
    dx = np.arctan2(T[0], -T[2])
    dy = np.arctan2(T[1], -T[2])
    quat = euler2quat(-dy, -dx, 0, axes='sxyz')
    quat = qmult(qinverse(quat), qt)
    return quat
def load_object_points(classes,args):
    """
    Load all 3D models of the YCB dataset

    Args:
        classes (tuple): tuple containing all classes of the YCB-V dataset
        args (argpase): arguments containing the location of the 3D models

    Returns:
     points_all (numpy.ndarray): ndaray containing all 3D models of the YCB-V dataset (x,y,z coordinates), and a dummy 3D model of zeros for the background Size: 22x2620x3 
    """
    points = [[] for _ in range(len(classes))]
    num = np.inf
    num_classes = len(classes)
    for i in range(1, num_classes):
        point_file = os.path.join(args.model_path, classes[i], 'points.xyz')
        print(classes[i])
        print(point_file)
        assert os.path.exists(point_file), 'Path does not exist: {}'.format(point_file)
        points[i] = np.loadtxt(point_file)
        if points[i].shape[0] < num:
            num = points[i].shape[0]

    points_all = np.zeros((num_classes, num, 3), dtype=np.float32)
    for i in range(1, num_classes):
        points_all[i, :, :] = points[i][:num, :]

    return points_all
def pixel_filter(pix_u,pix_v):
    """
    This function ensures that all pixels lying outside the image are removed eg. theire value set to 0 if the pixel values <threshold 
    The value is set to threshold if pixel value >threshold

    Args:
        pix_u (numy.ndarray): pixel of projected 3D model in image plane in x-direction (width=640)
        pix_v (numy.ndarray): pixel of projected 3D model in image plane in y-direction (width=480)

    Returns:
        pix_u (numy.ndarray): filtered pixels pixel of projected 3D model in image plane in x-direction 
        pix_v (numy.ndarray): filtered pixel of projected 3D model in image plane in y-direction (width=480)
    """
    for i,u in enumerate(pix_u):
        if u<0:
            pix_u[i]=0
        elif u>= 640:
            pix_u[i]=640
    for i,v in enumerate(pix_v):
        if v<0:
            pix_v[i]=0
        elif v>=480:
            pix_v[i]=480
    return pix_u,pix_v
def calc_area(x_coord, y_coord):
    """Calculated area occupied by object in pixel

    Args:
        x_coord (numy.ndarray): proejcted 3D model coordinates in x-direction 
        y_coord (numy.ndarray): proejcted 3D model coordinates in y-direction 
    Returns:
        [np.float64]: area of object in image plane
    """
    return (np.max(x_coord) - np.min(x_coord)) * (np.max(y_coord) - np.min(y_coord))
def save_json(args,images,categories):
    """
    generates a annotation File in CoCo Format. 

    Args:
        images (list): List containing all image names to be annotated
        categories (list): List containing annotation of categories as its the case for CoCo Format
    Returns: 
        Save json file containing the annotation in Coco Format.
    """
    num_of_images = len(images) # number of images 
    random.seed(0) 
    image_id_index = random.sample([i for i in range(0, num_of_images)], num_of_images) # sample a unique image_id index for every image in the list.
    images_annot=[]
    annotation=[]
    points=load_object_points(classes_all,args) # Load all 3D Models of objects
    id=0
    for image_counting,image in enumerate(images): # iterates over all images in the list. 
        if image[-10:] in  '-color.png': #This is for synthetic images as the image name has already -color.png in it
            file_name=image[-25:]
            mat = scipy.io.loadmat(os.path.join(args.real,image[:-10])+ "-meta.mat") # Load mat-file containing Rotation, Translation of object relative to camera, intrinsics 
        else: # For real data
            file_name = image + '-color.png' # For each real image in the list add -color.png to the name
            mat = scipy.io.loadmat(os.path.join(args.real,image)+ "-meta.mat") # load mat-file for each image 
        image_id = image_id_index[image_counting] # give unique image_ id to image. 
        image_annotation = {'file_name':file_name, 'id':image_id,"width":640,"height":480,"intrinsic":mat["intrinsic_matrix"].flatten().tolist()} # Write image annotation as dic
        images_annot.append(image_annotation) # append list containing image annotations with dic 
        if image[-10:] in  '-color.png': # For synthetic images
            number_of_objects=np.expand_dims(mat["cls_indexes"], axis=2).astype(np.int32)  # Read all objects contained in an image from mat file
            number_of_objects=number_of_objects[0]
        else: # For real images
            number_of_objects=mat["cls_indexes"] # Read all objects contained in an image from mat file
        #print(poses.shape)
        bbox=np.zeros((1,4))
        for k,obj in enumerate(number_of_objects): # iterate over all objects in an image [object id starts with 1 - 22]
            R= mat['poses'][:, :3, k] # Get theire Rotation  mat['poses']=[R|T] is the extrinsics
            T= mat['poses'][:, 3, k] # Get theire Translation 
            qt = mat2quat(R) # transform rotation into quaternions 
            P3D = np.ones((4, points.shape[1]), dtype=np.float32) # Create dumb np.ndarray to store the 3D model of object obj 
            P3D[0, :] = points[obj,:,0] 
            P3D[1, :] = points[obj,:,1]
            P3D[2, :] = points[obj,:,2]
            x2d = np.matmul(mat['intrinsic_matrix'], np.matmul(mat['poses'][:, :, k], P3D)) # Project 3D model into image plane using intrinsics and extrinsics
            x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
            x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
            area_orig=calc_area(x2d[0, :],x2d[1, :]) # calculate area occupied by object with all 
            u,v=pixel_filter(x2d[0, :],x2d[1, :]) # filter filter lying outside the image plane
            area_red=calc_area(u,v) # calculate real area 
            if area_red/area_orig<0.1: # if the area occupied by the object is less then 10% of the original area then skip the object (Fully occluded)
                continue
            #Create bbox using the filtered pixel coordinates 
            bbox[0,0]= np.min(u) 
            bbox[0,1] = np.min(v)
            bbox[0,2]= np.max(u)- np.min(u)
            bbox[0,3] = np.max(v)- np.min(v)
            bbox_annot=bbox.astype(np.int).flatten()

            
            position={"quaternions":qt.flatten().tolist(),"position":T.flatten().tolist()}
            annotations={"id":id,"image_id":image_id,"relative_pose":position,"bbox":bbox_annot.tolist(),"area":area_red,'iscrowd':0,'category_id':obj.tolist()[0]} 
            id+=1
            annotation.append(annotations)
    data={'images':images_annot, "categories": categories,"annotations":annotation} # Coco Annotation Format 
    with open(args.output_name, 'w') as outfile:
        json.dump(data, outfile) # Save .json FILE
    return



def annotate(args):
    class_file = open(args.cls)
    classes = class_file.read().splitlines() # Read .txt File containing classes
    category_id = 0
    categories = []
    for classe in classes:
        category_id += 1
        category = {'supercategory':classe, 'id':category_id, 'name':classe} # Annotated category as it is done in CoCo Format
        categories.append(category)
        classes = class_file.read().splitlines()
    class_file.close()
    # Read the names of the images to generator annotations
    image_names_file = open(args.train_real) # Read all real images from .txt File
    images = image_names_file.read().splitlines() # append list with real images name
    image_names_file.close()
    if args.Flag:    # If annotated with synthetic Data set True, For Testing, Flag must be set to False
        image_names_file=glob(args.train_syn) # find all images in data_syn directory with *color.png ending
        images=images+image_names_file # append list containing real images with synthetic images
    save_json(args,images,categories)
    return

if __name__ == "__main__":
    path = os.path.dirname(os.path.abspath(__file__))
    parser=argparse.ArgumentParser()
    parser.add_argument("--output_name",type=str,help="Output Name of .json File", default='Test_run.json')
    parser.add_argument("--cls",type=str,help="Path to .txt File containing the classes", default=os.path.join(path,'image_sets/classes.txt'))
    parser.add_argument("--real",type=str,help="Path to images and Bbox",default=os.path.join(path,'images'))
    parser.add_argument("--train_real",type=str,help="Path to txt File containing training data",default=os.path.join(path,'image_sets/keyframe.txt'))
    parser.add_argument("--Flag",type=str,help="Set Flag to true if annotation is with or without synthetic data",default=False)
    parser.add_argument("--train_syn",type=str,help="Path to Folder containing synthetic images",default=os.path.join(path+'images/data_syn/','*color.png'))
    parser.add_argument("--model_path",type=str,help="Path to training data",default=os.path.join(path,'models'))
    args = parser.parse_args()
    annotate(args)
