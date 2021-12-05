import torch
import numpy as np
from utils.eval import  Evaluation
from progress.bar import Bar
import time
from utils.utils import AverageMeter
from transforms3d.quaternions import mat2quat, qmult, quat2mat
from transforms3d.euler import euler2quat,euler2mat
import utils.logger as logger

def train( epoch,arg, data_loader, model, criterions, optimizer=None):
    """

    Args:
        arg ([dic]): Config dic containing all config params
        data_loader : Test Dataloader
        model : Osition estimation network
        classes (class tuple): tuple containing class names
        prj_vtx (class dict): dict containing 3D models prj_vrtx['class']:[x,y,z]
        model_info (class dict): dict containing model 3D model summary: min and max coordinates of 3D model,  diameter and class name
    """
    time_monitor = False
    model.train()
    Loss = AverageMeter()
    Loss_rot = AverageMeter()
    Loss_trans = AverageMeter()
    Loss_classification=AverageMeter()
    Loss_reg=AverageMeter()
    num_iters = len(data_loader)
    bar = Bar('{}'.format(arg.pytorch.exp_id[-60:]), max=num_iters)
    i=0
    for  obj, rgb, relative_pose, relative_quat,bbox, annotations in data_loader:
        bs=len(rgb)
        cur_iter = i + (epoch - 1) * num_iters
        # Check if GPU is used or CPU 
        if arg.pytorch.gpu > -1:
            inp_var = rgb.cuda(arg.pytorch.gpu, non_blocking=True).float() # Pass Image to cuda 
            for k in range(bs):
                #As the number of objects changes upon every image gt-annotations are load in a list. 
                
            	obj[k] = obj[k].cuda(arg.pytorch.gpu, non_blocking=True).float() # pass object name to cuda
            	relative_quat[k] = relative_quat[k].cuda(arg.pytorch.gpu, non_blocking=True).float() # pass gt-quaternion to cuda
            	relative_pose[k] = relative_pose[k].cuda(arg.pytorch.gpu, non_blocking=True).float() # pas gt-transltation to cuda
            	bbox[k]=bbox[k].cuda(arg.pytorch.gpu, non_blocking=True).float() # pass bbox to cuda
            	annotations[k]=annotations[k].cuda(arg.pytorch.gpu,non_blocking=True).float() # pass [[bbox,obj],...] to cuda 
            L = torch.empty(1,0, 5).cuda(arg.pytorch.gpu,non_blocking=True).float() # create empty tensor to store ground truth [bbox,obj] as tensor
            if arg.network.rot_representation=='rot':
                quat_label=torch.empty(1,0,3,3).cuda(arg.pytorch.gpu,non_blocking=True).float() # create empty tensor to store gt Rotation matrix as tensor
            else: 
                quat_label=torch.empty(1,0,4).cuda(arg.pytorch.gpu,non_blocking=True).float()  #create empty tensor to store gt quaternion as tensor
            trans_label=torch.empty(1,0,3).cuda(arg.pytorch.gpu,non_blocking=True).float() # create empty tensor to store gt translation as tensor 
            conc=zip(bbox,relative_pose, relative_quat) 
            for u, (bbox_batch,relative_pose_batch,relative_quat_batch) in enumerate(conc):
                """
                Iterate over the number of batches and store the values in tensors. 
                """
                _,k,_=bbox_batch.shape
                bbox_roi=torch.cat(((u)*torch.ones(1,k,1).cuda(arg.pytorch.gpu,non_blocking=True),bbox_batch.cuda()),2)
                L=torch.cat((L,bbox_roi),dim=1)
                quat_label=torch.cat((quat_label,relative_quat_batch.cuda()),dim=1)
                trans_label=torch.cat((trans_label,relative_pose_batch.cuda()),dim=1)
        else:
            inp_var = rgb.float()
            L = torch.empty(1,0, 5) # create empty tensor to store ground truth [bbox,obj] as tensor
            if arg.network.rot_representation=='rot':
                quat_label=torch.empty(1,0,3,3)  # create empty tensor to store gt Rotation matrix as tensor
            else:
                quat_label=torch.empty(1,0,4) #create empty tensor to store gt quaternion as tensor
            trans_label=torch.empty(1,0,3)
            conc=zip(bbox,relative_pose, relative_quat) # create empty tensor to store gt translation as tensor 
            for u, (bbox_batch,relative_pose_batch,relative_quat_batch) in enumerate(conc):
                """
                Iterate over the number of batches and store the values in tensors. 
                """
                _,k,_=bbox_batch.shape
                bbox_roi=torch.cat(((u)*torch.ones(1,k,1),bbox_batch),2)
                L=torch.cat((L,bbox_roi),dim=1)
                quat_label=torch.cat((quat_label,relative_quat_batch),dim=1)
                trans_label=torch.cat((trans_label,relative_pose_batch),dim=1)
        # Squeeze first dimension  of tensors L quat, trans and intrinsic
        L=torch.squeeze(L,0)
        quat_label=torch.squeeze(quat_label,0)
        trans_label=torch.squeeze(trans_label,0)
        #intrinsic=np.asarray(intrinsic.squeeze(0))
        
        # forward propagation
        T_begin = time.time()
        anchors,regressor, classes,q,t= model(inp_var,L)
        T_end = time.time()
        
        if time_monitor:
            logger.info("time for a batch forward of resnet model is {}".format(T_end))
        # loss

        if 'class' in arg.pytorch.task.lower() and not arg.network.class_head_freeze:
            cls_loss, reg_loss  = criterions[arg.loss.class_loss_type](classes, regressor, anchors, annotations)
            cls_loss=cls_loss.mean()
            reg_loss=reg_loss.mean()
            loss_rot = criterions[arg.loss.rot_loss_type](q, quat_label, bs)
            loss_trans = criterions[arg.loss.trans_loss_type](t, trans_label)*trans_label.shape[0]/bs
        else: 
            cls_loss, reg_loss=0,0
        if 'position' in arg.pytorch.task.lower():
            loss_rot = criterions[arg.loss.rot_loss_type](q, quat_label, bs)
            loss_trans = criterions[arg.loss.trans_loss_type](t, trans_label)*trans_label.shape[0]/bs
        #else:
            #loss_rot=0
            #loss_trans=0
        loss = arg.loss.rot_loss_weight * loss_rot + arg.loss.trans_loss_weight * loss_trans+arg.loss.class_loss_weight*cls_loss+arg.loss.reg_loss_weight*reg_loss     
        Loss.update(loss.item() if loss != 0 else 0, bs)
        Loss_rot.update(loss_rot.item() if loss_rot != 0 else 0, bs)
        Loss_trans.update(loss_trans.item() if loss_trans != 0 else 0, bs)
        Loss_classification.update(cls_loss.item() if cls_loss != 0 else 0, bs)
        Loss_reg.update(reg_loss.item() if reg_loss != 0 else 0, bs)

        optimizer.zero_grad()
        model.zero_grad()
        T_begin = time.time()
        loss.backward()
        optimizer.step()
        T_end = time.time() - T_begin
        if time_monitor:
            logger.info("time for backward of model: {}".format(T_end))
       
        Bar.suffix = 'train Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:} | Loss {loss.avg:.4f} | Loss_rot {loss_rot.avg:.4f} | Loss_trans {loss_trans.avg:.4f} | Loss_class{loss_class.avg:.4f}| Loss_reg{loss_reg.avg:.4f}'.format(
            epoch, i, num_iters, total=bar.elapsed_td, eta=bar.eta_td, loss=Loss, loss_rot=Loss_rot, loss_trans=Loss_trans,loss_class=Loss_classification,loss_reg=Loss_reg)
        i=i+1
        bar.next()
    bar.finish()
    
    return {'Loss': Loss.avg, 'Loss_rot': Loss_rot.avg, 'Loss_trans': Loss_trans.avg}, 1