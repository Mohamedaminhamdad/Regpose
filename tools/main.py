import torch
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 
import numpy as np
import random
import torch.utils.data
import matplotlib
matplotlib.use('Agg')
from torchinfo import summary
import _init_paths
from config import config
import utils.logger as logger 
from train import train
from utils.loss import FocalLoss, QuatLoss, GEodistance     
from model import build_model, save_model
from dataset import Dataset
from utils.utils import CustomDataParallel
def main():
    
    cfg = config().parse()
    network, optimizer = build_model(cfg)
    criterions = {'L1': torch.nn.SmoothL1Loss(beta=1/9),
                  'quatloss': QuatLoss(),
                  'acos': GEodistance(),
                  'FocalLoss':FocalLoss()}

    if cfg.pytorch.gpu > -1:
        logger.info('Using GPU{}'.format(cfg.pytorch.gpu))
        network = network.cuda()
        #network= CustomDataParallel(network,2) # This is to parallelize Datas across multiple GPUs
        for k in criterions.keys():
            criterions[k] = criterions[k].cuda(cfg.pytorch.gpu)
    def _worker_init_fn():
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2**32 - 1
        random.seed(torch_seed)
        np.random.seed(np_seed)
    train_loader = torch.utils.data.DataLoader(
        Dataset(cfg, 'train'),
        batch_size=cfg.train.train_batch_size,
        shuffle=True,
        num_workers=int(cfg.pytorch.threads_num),
        worker_init_fn=_worker_init_fn(),
        collate_fn=Dataset.collate_fn)

    for epoch in range(cfg.train.begin_epoch, cfg.train.end_epoch + 1):
        mark = epoch if (cfg.pytorch.save_mode == 'all') else 'last'
        log_dict_train, _ = train(epoch, cfg, train_loader, network, criterions, optimizer)
        save_model(network,os.path.join(cfg.pytorch.save_path, 'model_{}.checkpoint'.format(mark)))  # optimizer
        if epoch in cfg.train.lr_epoch_step:
            if optimizer is not None:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= cfg.train.lr_factor
                    logger.info("drop lr to {}".format(param_group['lr']))

    torch.save(network.cpu(), os.path.join(cfg.pytorch.save_path, 'model_cpu.pth'))

if __name__ == '__main__':
    main()
