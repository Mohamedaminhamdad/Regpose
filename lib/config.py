import os
import yaml
import argparse
import copy
import numpy as np
from easydict import EasyDict as edict
from tensorboardX import SummaryWriter
import sys
import director
from datetime import datetime
import utils.logger as logger
from tensorboardX import SummaryWriter
from pprint import pprint

def config_pytorch():
    config = edict()
    config.exp_id = 'Respose'          # Experiment ID
    config.task = 'class'             # 'class-bbox | Postion |'
    config.gpu = 1                    # Use GPU set to >-1 else Run model on CPU
    config.threads_num = 12         # 'nThreads' For Dataloader
    config.save_mode = 'all'        # 'all' | 'best', save all models or only save the best model
    config.load_model = ''          # path to a previously trained model
    config.test = False             # run in test mode or not
    config.gt=True                 # Test model with gt bounding boxes |True| predicted |False|
    return config



def train_config():
    config = edict()
    config.begin_epoch = 1  # Default value begin epoch
    config.end_epoch = 20  # default value end epoch
    config.test_interval = 1  # default value test_interval
    config.train_batch_size = 8 # default test/train batch-size
    config.lr = 1e-4 # default lr rate for adamw
    config.lr_epoch_step = [10, 20, 30] #default epochs for lr-reduction lr*lr_factor
    config.lr_factor = 0.1 # lr-reduction factor
    config.optimizer_name = 'adamw'  # optimizer
    config.momentum = 0.0
    config.weightDecay = 0.0
    config.alpha = 0.99
    config.epsilon = 1e-8
    config.Beta=0
    return config

def loss_config():
    config = edict()
    config.class_loss_type='FocalLoss' # default loss for CLassification Head
    config.class_loss_weight=1 # default weight for classifaction error  
    config.reg_loss_weight=1 # default weight for bbounding box error
    config.rot_loss_type = 'quatloss' # default loss for Rotation
    config.rot_loss_weight = 1 # default weight for rotation
    config.trans_loss_type = 'L1' #default loss for translation
    config.trans_loss_weight = 1 # default loss weight for translation
    return config

def network_config():
    config = edict()
    # ------ backbone -------- #
    config.arch = 'resnet'   # Backbone resnet 
    config.back_freeze=False # default conf. for backbone freeze
    config.back_input_channel = 3 # Input channles backbone 
    config.back_layers_num=34 # number os layers for backbone
    # -------regression-------#
    config.class_head_freeze=False # default freeze class-head
    # ------ rotation head -------- #
    config.rot_head_freeze = False # default freeze rotation head
    config.rot_representation='quat'  # default  rotation representation quat-head: |quat| 6D representation: |rot|
    # ------ translation head -------- #
    config.trans_head_freeze = False   # default freeze translation head
    return config
def get_default_dataset_config():
    config = edict()
    config.name = 'YCB' # Default Dataset name
    return config
def get_base_config():
    """
    Here all default configurations are loaded
    """
    base_config = edict()
    base_config.dataset=get_default_dataset_config()
    base_config.pytorch = config_pytorch()
    base_config.train = train_config()
    base_config.network = network_config()
    base_config.loss = loss_config()
    return base_config

def update_config_from_file(_config, config_file, check_necessity=True):
    """
        Update default configuration using config-file

    Args:
        _config : config dictionary containing default config params
        config_file: yaml-File containing configurations 
        check_necessity (bool, optional): [add config params who are in default configurations]. Defaults to True.
        


    Returns:
        config : dict containing updataed configuaration 
    """
    config = copy.deepcopy(_config)
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    for vk, vv in v.items():
                        if vk in config[k]:
                            if isinstance(vv, list) and not isinstance(vv[0], str):
                                config[k][vk] = np.asarray(vv)
                            else:
                                config[k][vk] = vv
                        else:
                            if check_necessity:
                                raise ValueError("{}.{} not exist in config".format(k, vk))
                else:
                    raise ValueError("{} is not dict type".format(v))
            else:
                if check_necessity:
                    raise ValueError("{} not exist in config".format(k))
    return config

class config():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='pose experiment')
        self.parser.add_argument('--cfg', type=str,default='../Config-Files/Quat-Head/Config-Test.yaml', help='path/to/configure_file') ## Put here config-file 
        self.parser.add_argument('--test', action='store_true', help='')

    def parse(self):
        config = get_base_config()                  # get default arguments
        args, rest = self.parser.parse_known_args() # get arguments from command line
        for k, v in vars(args).items():
            config.pytorch[k] = v 
        config_file = config.pytorch.cfg
        config = update_config_from_file(config, config_file, check_necessity=False) # update arguments from config file
        # complement config regarding dataset
        # automatically correct config
        if config.network.back_freeze == True:
            config.loss.backbone_loss_weight = 0
        if config.network.rot_head_freeze == True:
            config.loss.rot_loss_weight = 0
        if config.network.trans_head_freeze == True:
            config.loss.trans_loss_weight = 0

        if config.pytorch.test:
            config.pytorch.exp_id = config.pytorch.exp_id + 'TEST'

        # complement config regarding paths
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        # save path
        config.pytorch['save_path'] = os.path.join(director.exp_dir, config.pytorch.exp_id, now)
        if not os.path.exists(config.pytorch.save_path):
            os.makedirs(config.pytorch.save_path, exist_ok=True)
        # logger path
        logger.set_logger_dir(config.pytorch.save_path, action='k')

        pprint(config)
        # copy and save current config file
        os.system('cp {} {}'.format(config_file, os.path.join(config.pytorch.save_path, 'config_copy.yaml')))
        # save all config infos
        args = dict((name, getattr(config, name)) for name in dir(config) if not name.startswith('_'))
        refs = dict((name, getattr(director, name)) for name in dir(director) if not name.startswith('_'))
        file_name = os.path.join(config.pytorch.save_path, 'config.txt')
        with open(file_name, 'wt') as cfg_file:
            cfg_file.write('==> Cmd:\n')
            cfg_file.write(str(sys.argv))
            cfg_file.write('\n==> Opt:\n')
            for k, v in sorted(args.items()):
                cfg_file.write('  %s: %s\n' % (str(k), str(v)))
            cfg_file.write('==> Ref:\n')
            for k, v in sorted(refs.items()):
                cfg_file.write('  %s: %s\n' % (str(k), str(v)))

        return config
