import torch
import numpy as np
from utils.eval import  Evaluation
from progress.bar import Bar
import time
from transforms3d.quaternions import mat2quat, qmult, quat2mat
from transforms3d.euler import euler2quat,euler2mat
from bbox_matching import match_bboxes
def allocentric2egocentricR(R_pr, T):
    """
    Transform aLlocentric quaternion into egocentric Rotation Matrix

    Args:
        R_pr [np.float]: Predicted egocentric Rotation Matrix shape: 3x3
        T [np.float]: Predicted Translation  shape: 1x3

    Returns:
        R[np.float]: Egocentric Rotation shape: 3x3
    """
    dx = np.arctan2(T[0], -T[2])
    dy = np.arctan2(T[1], -T[2])
    Rc = euler2mat(-dy, -dx, 0, axes='sxyz')
    R = np.matmul(Rc, R_pr)
    return R

classes_YCB = ('002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', \
                         '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana', '019_pitcher_base', \
                         '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', '040_large_marker', \
                         '051_large_clamp', '052_extra_large_clamp', '061_foam_brick')
def allocentric2egocentric(q_pr, T):
    """
    Transform allocentric quaternion into egocentric quaternion

    Args:
        q_pr [np.float]: Predicted egocentric quaternion shape: 1x4
        T [np.float]: Predicted Translation  shape: 1x3

    Returns:
    quatR[np.float]: Egocentric quaternion shape: 1x4
    """
    dx = np.arctan2(T[0], -T[2])
    dy = np.arctan2(T[1], -T[2])
    quat = euler2quat(-dy, -dx, 0, axes='sxyz')
    quat = qmult(quat, q_pr)
    return quat
def test( arg, data_loader, model, classes_model,prj_vtx,model_info):
    """

    Args:
        arg ([dic]): Config dic containing all config params
        data_loader : Test Dataloader
        model : Osition estimation network
        classes (class tuple): tuple containing class names
        prj_vtx (class dict): dict containing 3D models prj_vrtx['class']:[x,y,z]
        model_info (class dict): dict containing model 3D model summary: min and max coordinates of 3D model,  diameter and class name
    """
    epoch=1
    i=0
    model.eval()
    num_iters = len(data_loader)
    Eval = Evaluation(prj_vtx,classes_model,model_info)
    bar = Bar('{}'.format(arg.pytorch.exp_id[-60:]), max=num_iters)
    for  obj, rgb, relative_pose, relative_quat,bbox, annotations, intrinsic in data_loader:
        i+=1
        bs=len(rgb)
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
        intrinsic=np.asarray(intrinsic.squeeze(0))
        
        # forward propagation
        T_begin = time.time()
        if arg.pytorch.gt and arg.pytorch.test:
            q,t = model(inp_var,L)
        elif not arg.pytorch.gt  and arg.pytorch.test:
            classes,pred_bbox,q,t = model(inp_var,L)
        else: 
            raise ValueError("You are Testing so please set arg.pytorch.test to True in your config File, If you want to train got to main.py")
        T_end = time.time()
        
        T=t.cpu().detach().numpy() # Pass predicted Translation to cpu and transform it into a numpy array
        q=q.cpu().detach().numpy() # Pass predicted quaternions/Rotation-Matrix to cpu and transform it into a numpy array
        
        obj=obj.detach().numpy().squeeze(0).squeeze(0).squeeze(1) # pass class to cpu and transfrom it to numpy 
        quat_gt=quat_label.cpu().detach().numpy()  # Pass gt Translation to cpu and transform it into a numpy array
        trans_gt=trans_label.cpu().detach().numpy() # Pass gt quaternions/Rotation-Matrix to cpu and transform it into a numpy array
        if arg.pytorch.gt:
            # Test with gt bounding boxes 
            if arg.network.rot_representation=='rot':
                # Evaluation if Rotation matrix is predicted 
                for idx in range(trans_gt.shape[0]):
                    T_begin = time.time()
                    q1=q[idx,:]
                    q1=allocentric2egocentricR(q1,T[idx,:].reshape(3, 1)) # Transform allocentric rotation into egocentric rotation
                    pose_est = np.concatenate((q1, T[idx,:].reshape(3, 1)), axis=1) # [R,T] predicted
                    pose_gt = np.concatenate((quat_gt[idx,:], trans_gt[idx,:].reshape(3, 1)), axis=1) #[R,T] gt 
                    obj_=classes_model[int(obj[idx])]
                    intrinsics_=intrinsic
                    Eval.quaternion_est_all[obj_].append(mat2quat(q1))  
                    Eval.quaternion_gt_all[obj_].append(mat2quat(quat_gt[idx,:]))
                    Eval.pose_est_all[obj_].append(pose_est)
                    Eval.pose_gt_all[obj_].append(pose_gt)
                    Eval.translation_gt[obj_].append(np.asarray((trans_gt[idx,:]).reshape(3, 1)))
                    Eval.translation_est[obj_].append(np.asarray((T[idx,:].reshape(3, 1))))
                    Eval.num[obj_] += 1
                    Eval.camera_k[obj_].append(intrinsics_)
                    Eval.numAll += 1
            else: 
                # Evalutation if quaternions are predicted
                for idx in range(trans_gt.shape[0]):
                    q1=q[idx,:]
                    q1=allocentric2egocentric(q1,T[idx,:].reshape(3, 1)) # Transform allocentric quaternion into egocentric quaternion
                    pose_est = np.concatenate((quat2mat(q1), T[idx,:].reshape(3, 1)), axis=1) #[R,T] predicted
                    pose_gt = np.concatenate((quat2mat(quat_gt[idx,:]), trans_gt[idx,:].reshape(3, 1)), axis=1) #[R,T] gt
                    obj_=classes_model[int(obj[idx])]
                    intrinsics_=intrinsic
                    Eval.quaternion_est_all[obj_].append(q1)
                    Eval.quaternion_gt_all[obj_].append(quat_gt[idx,:])
                    Eval.pose_est_all[obj_].append(pose_est)
                    Eval.pose_gt_all[obj_].append(pose_gt)
                    Eval.translation_gt[obj_].append(np.asarray((trans_gt[idx,:]).reshape(3, 1)))
                    Eval.translation_est[obj_].append(np.asarray((T[idx,:].reshape(3, 1))))
                    Eval.num[obj_] += 1
                    Eval.camera_k[obj_].append(intrinsics_)
                    Eval.numAll += 1
        else:
            # Evaluation for predicted bounding boxes from Regpose
            bbox=bbox.squeeze(0).squeeze(0).numpy()
            pred_bbox=pred_bbox.squeeze(0)
            idx_gt_actual, idx_pred_actual, _,_ =match_bboxes(bbox,pred_bbox) # match predicted bounding boxes and gt-bounding boxes
            j=0
            if arg.network.rot_representation=='rot':
                # Evaluation if Rotation matrix is predicted 
                for id_pred,id_gt in zip(list(idx_pred_actual),list(idx_gt_actual)):
                    q1=q[id_pred,:]
                    id=idx_gt_actual[j]
                    q1=allocentric2egocentricR(q1,T[id_pred,:].reshape(3, 1))  #Transform allocentric Rotation matrix into egocentric rotation matrix 
                    pose_est = np.concatenate((q1, T[id_pred,:].reshape(3, 1)), axis=1) #[R,T] predicted  
                    pose_gt = np.concatenate((quat_gt[id_gt,:], trans_gt[id_gt,:].reshape(3, 1)), axis=1) # [R,T] gt
                    obj_=classes_model[int(obj[id_gt])]
                    intrinsics_=intrinsic
                    Eval.quaternion_est_all[obj_].append(mat2quat(q1))
                    Eval.quaternion_gt_all[obj_].append(mat2quat(quat_gt[id_gt,:]))
                    Eval.pose_est_all[obj_].append(pose_est)
                    Eval.pose_gt_all[obj_].append(pose_gt)
                    Eval.translation_gt[obj_].append(np.asarray((trans_gt[id_gt,:]).reshape(3, 1)))
                    Eval.translation_est[obj_].append(np.asarray((T[id_pred,:].reshape(3, 1))))
                    Eval.num[obj_] += 1
                    Eval.camera_k[obj_].append(intrinsics_)
                    Eval.numAll += 1
            else:
                # Evaluation if quaternion is predicted 
                for id_pred,id_gt in zip(list(idx_pred_actual),list(idx_gt_actual)):
                    q1=q[id_pred,:]
                    id=idx_gt_actual[j]
                    q1=allocentric2egocentric(q1,T[id_pred,:].reshape(3, 1)) # Transform allocentric quaternion into egocentric quaternion
                    pose_est = np.concatenate((quat2mat(q1), T[id_pred,:].reshape(3, 1)), axis=1) # [R,T] predicted
                    pose_gt = np.concatenate((quat2mat(quat_gt[id_gt,:]), trans_gt[id_gt,:].reshape(3, 1)), axis=1) # [R,T] gt
                    obj_=classes_model[int(obj[id_gt])]
                    intrinsics_=intrinsic
                    Eval.quaternion_est_all[obj_].append(q1)
                    Eval.quaternion_gt_all[obj_].append(quat_gt[id_gt,:])
                    Eval.pose_est_all[obj_].append(pose_est)
                    Eval.pose_gt_all[obj_].append(pose_gt)
                    Eval.translation_gt[obj_].append(np.asarray((trans_gt[id_gt,:]).reshape(3, 1)))
                    Eval.translation_est[obj_].append(np.asarray((T[id_pred,:].reshape(3, 1))))
                    Eval.num[obj_] += 1
                    Eval.camera_k[obj_].append(intrinsics_)
                    Eval.numAll += 1
        Bar.suffix = 'Test: [{0}][{1}/{2}]| Elapsed-Time: {elaptime:} | Remaining-Time: {rmtime:}'.format(
        epoch, i, num_iters, elaptime=bar.elapsed_td, rmtime=bar.eta_td)
        bar.next()
    #print(wrong_detected)

    bar.finish()
    Eval.te3d()
    Eval.calculate_class_avg_rotation_error('mohamed/')
    Eval.evaluate_pose()
    Eval.evaluate_pose_add('ADD')
    Eval.evaluate_pose_add('ADD-S','symmetric')
    Eval.evaluate_trans()
    return regressor, classes