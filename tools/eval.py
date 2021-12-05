import torch
import os
import numpy as np
import random
import torch.utils.data
import matplotlib
matplotlib.use('Agg')
import cv2
from torchinfo import summary
cv2.ocl.setUseOpenCL(False)
import _init_paths
from config import config
from tqdm import tqdm 
import utils.logger as logger 
import director
from model_test import build_model
from dataset import Dataset
from test import test
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 

classes_YCB = ('002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', \
                         '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana', '019_pitcher_base', \
                         '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', '040_large_marker', \
                         '051_large_clamp', '052_extra_large_clamp', '061_foam_brick') # Class Names for the YCB-V Dataset
classes={'YCB':classes_YCB}
def model_info(points):
    """
    Args:
        points : given the 3D POint CLoud of an Object. 
    Returns:
        infos: returns diameter of object, its min and max coordinates
    """
    infos = {}
    extent= 2 * np.max(np.absolute(points), axis=0)
    infos['diameter'] = np.sqrt(np.sum(extent * extent))
    infos['min_x'],infos['min_y'],infos['min_z']=np.min(points,axis=0)
    infos['max_x'],infos['max_y'],infos['max_z']=np.min(points,axis=0)
    return infos
def main():
    arg = config().parse() # Load Reguired Configurations -Set-up
    network= build_model(arg) # build model using model_test.py
    if arg.pytorch.gpu > -1:
        logger.info('GPU{} is used'.format(arg.pytorch.gpu))
        network = network.cuda(arg.pytorch.gpu) # Use Network within a GPU
    else: 
        logger.info('GPU Is not in Use{}'.format(arg.pytorch.gpu))
        network = network # Use Network with CPU    
    def _worker_init_fn():
        """
        Function to define workers for DATALOADER   
        """
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2**32 - 1
        random.seed(torch_seed)
        np.random.seed(np_seed)
    # ---- Initialize Test-Dataset----------
    test_loader = torch.utils.data.DataLoader(
        Dataset(arg, 'test'),
        batch_size= 1,
        shuffle=True,
        num_workers=int(arg.pytorch.threads_num),
        worker_init_fn=_worker_init_fn()
    )

    obj_vtx = {}
    model_info_={}
    logger.info('Load 3D Models')
    for obj in tqdm(classes[arg.dataset.name]):
        """
        Load every 3D model and save it into obj_vtx['cls'] and each model info containing diameter min and max coordinates of 3D Models
        """
        obj_vtx[obj] = np.loadtxt(os.path.join(director.model_dir,obj,'points.xyz'))
        model_info_[obj] = model_info(obj_vtx[obj])
    test(arg, test_loader, network, classes[arg.dataset.name],obj_vtx,model_info_)
    
    
if __name__ == '__main__':
    main()

