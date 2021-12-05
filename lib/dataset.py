import numpy as np
import cv2
import os
import utils.logger as logger
from glob import glob  
import json
import torch
from transforms3d.euler import euler2quat,euler2mat
from transforms3d.quaternions import mat2quat, qmult, quat2mat,qinverse
import director

class Dataset():
    def __init__(self, arg,split='train'):
        logger.info('==> initializing {} {} data.'.format(arg.dataset.name, split))
        self.rot_rep=arg.network.rot_representation
        self.split=split
        if split == 'train':
            self.dir=director.dataset_dir
            with open(os.path.join(director.dataset_dir,'keyframe.json'),'r') as f: 
                self.data=json.load(f)
            self.real_num = len(self.data['images'])
        else: 
            self.dir=director.dataset_dir
            with open(os.path.join(director.dataset_dir,'keyframe.json'),'r') as f: 
                self.data=json.load(f)
            self.real_num = len(self.data['images'])
    def allocentric2egocentric(self,qt, T):
        dx = np.arctan2(T[0], -T[2])
        dy = np.arctan2(T[1], -T[2])
        quat = euler2quat(-dy, -dx, 0, axes='sxyz')
        quat = qmult(qinverse(quat), qt)
        return quat
    def __getitem__(self, idx):
        item_={}
        annot=self.data['images'][idx]
        image_name=annot['file_name']
        item_['rgb_pth'] =os.path.join(director.dataset_dir,'images',image_name)
        image_id=annot['id']
        item_['width']=annot['width']
        item_['height']=annot['height']
        item_['intrinsic']=annot['intrinsic']
        value=filter(lambda item1: item1['image_id']==image_id,self.data['annotations'])
        empty_array_id = np.array([])
        empty_array_bbox = np.array([])
        empty_array_pos=np.array([])
        empty_array_quat=np.array([])
        for item2 in value: 
            empty_array_id=np.append(empty_array_id,np.asarray(item2['category_id']))
            bbox=[item2['bbox'][0],item2['bbox'][1],item2['bbox'][0]+item2['bbox'][2],item2['bbox'][1]+item2['bbox'][3]]
            empty_array_bbox=np.append(empty_array_bbox,np.asarray(bbox))
            pos_pixel=item2['relative_pose']["position"]
            empty_array_pos=np.append(empty_array_pos,np.asarray(pos_pixel))
            relative_quat=np.array(item2['relative_pose']["quaternions"]).reshape(-1,4).flatten()
            relative_quat=self.allocentric2egocentric(relative_quat, np.asarray(pos_pixel))
            if self.rot_rep=='rot':
                relative_quat=quat2mat(relative_quat)
            empty_array_quat=np.append(empty_array_quat,np.asarray(relative_quat))
        empty_array_id=empty_array_id.reshape(-1,1)
        empty_array_pos=empty_array_pos.reshape(-1,3)
        if self.rot_rep=='rot':
            empty_array_quat=empty_array_quat.reshape(-1,3,3)
        else: 
            empty_array_quat=empty_array_quat.reshape(-1,4)
        empty_array_bbox=empty_array_bbox.reshape(-1,4)
        obj=np.array([empty_array_id])-1
        if image_name[:5]=='/home':
            image_name=image_name[-25:]
        rgb_path=os.path.join(director.dataset_dir,'images',image_name)
        rgb= cv2.imread(rgb_path)
        relative_pose=np.array([empty_array_pos]).astype(np.float32)
        relative_quat=np.array([empty_array_quat]).astype(np.float32)
        bbox=np.array([empty_array_bbox]).astype(np.float32)
        rgb = rgb.transpose(2, 0, 1).astype(np.float32) / 255.
        obj, rgb, relative_pose, relative_quat,bbox=torch.from_numpy(obj), torch.from_numpy(rgb), torch.from_numpy(relative_pose), torch.from_numpy(relative_quat), torch.from_numpy(bbox)
        annot=torch.cat((bbox,obj),dim=2)
        return  obj, rgb, relative_pose, relative_quat,bbox,annot,np.asarray(item_['intrinsic']).reshape(-1,3)
        
    def __len__(self):
        return self.real_num
    @staticmethod
    def collate_fn(batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        rgb = list()
        obj = list()
        relative_pose = list()
        relative_quat = list()
        bbox=list()
        annot=list()

        for b in batch:
            rgb.append(b[1])
            obj.append(b[0])
            relative_pose.append(b[2])
            relative_quat.append(b[3])
            bbox.append(b[4])
            annot.append(b[5])


        rgb = torch.stack(rgb, dim=0)
        return obj, rgb, relative_pose, relative_quat,bbox,annot
