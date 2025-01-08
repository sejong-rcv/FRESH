import torch
from mmdet.models.losses.utils import weighted_loss
from ..builder import LOSSES
from torch import nn as nn
import torch.nn.functional as F




def euler_to_quaternion(roll, pitch, yaw):
    qx = torch.sin(roll / 2) * torch.cos(pitch / 2) * torch.cos(yaw / 2) - torch.cos(roll / 2) * torch.sin(pitch / 2) * torch.sin(yaw / 2)
    qy = torch.cos(roll / 2) * torch.sin(pitch / 2) * torch.cos(yaw / 2) + torch.sin(roll / 2) * torch.cos(pitch / 2) * torch.sin(yaw / 2)
    qz = torch.cos(roll / 2) * torch.cos(pitch / 2) * torch.sin(yaw / 2) - torch.sin(roll / 2) * torch.sin(pitch / 2) * torch.cos(yaw / 2)
    qw = torch.cos(roll / 2) * torch.cos(pitch / 2) * torch.cos(yaw / 2) + torch.sin(roll / 2) * torch.sin(pitch / 2) * torch.sin(yaw / 2)
    return torch.stack([qw, qx, qy, qz], dim=-1)

def quaternion(pred_rpy, target_rpy):
    """
    pred_rpy: (N,3),
    target_rpy: (N,3)
    """
    pred_quat = euler_to_quaternion(pred_rpy[:, 0], pred_rpy[:, 1], pred_rpy[:, 2]) # (N,4)
    target_quat = euler_to_quaternion(target_rpy[:, 0], target_rpy[:, 1], target_rpy[:, 2]) # (N,4)
    
    return pred_quat, target_quat

def rotation_matrix_from_rpy(rpy_batch):
    roll = rpy_batch[:, 0]
    pitch = rpy_batch[:, 1]
    yaw = rpy_batch[:, 2]

    # Batch rotation matrix for roll (x-axis rotation)
    Rx = torch.stack([torch.ones_like(roll), torch.zeros_like(roll), torch.zeros_like(roll),
                      torch.zeros_like(roll), torch.cos(roll), -torch.sin(roll),
                      torch.zeros_like(roll), torch.sin(roll), torch.cos(roll)], dim=-1).view(-1, 3, 3)
    
    # Batch rotation matrix for pitch (y-axis rotation)
    Ry = torch.stack([torch.cos(pitch), torch.zeros_like(pitch), torch.sin(pitch),
                      torch.zeros_like(pitch), torch.ones_like(pitch), torch.zeros_like(pitch),
                      -torch.sin(pitch), torch.zeros_like(pitch), torch.cos(pitch)], dim=-1).view(-1, 3, 3)
    
    # Batch rotation matrix for yaw (z-axis rotation)
    Rz = torch.stack([torch.cos(yaw), -torch.sin(yaw), torch.zeros_like(yaw),
                      torch.sin(yaw), torch.cos(yaw), torch.zeros_like(yaw),
                      torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)], dim=-1).view(-1, 3, 3)
    
    # Combined rotation matrices: R = Rz * Ry * Rx (in batch)
    R = Rz @ Ry @ Rx
    return R

def stem_axis_vector_dot_product(pred_rpy, target_rpy):
    """
    pred_rpy: (N,3),
    target_rpy: (N,3)
    """
    # Ensure both inputs are of shape (N, 3)
    assert pred_rpy.shape == target_rpy.shape, "Input batches must have the same shape (N, 3)"

    # import pdb;pdb.set_trace()
    pred_quat = rotation_matrix_from_rpy(pred_rpy) ## (N,3,3)
    target_quat = rotation_matrix_from_rpy(target_rpy) ## (N,3,3)
    
    # Define x-axis unit vector (1, 0, 0) and broadcast for N frames
    x_axis_vector = torch.tensor([1.0, 0.0, 0.0]).unsqueeze(0).repeat(pred_rpy.shape[0], 1).cuda() # (N,3)

    # Transform the x-axis vector for each frame using the corresponding rotation matrices
    transformed_pred = torch.bmm(pred_quat, x_axis_vector.unsqueeze(-1)).squeeze(-1) ## (N,3)
    transformed_gt = torch.bmm(target_quat, x_axis_vector.unsqueeze(-1)).squeeze(-1) ## (N,3)
    
    
    ################################# added for transform_frame
    trans=torch.tensor([[0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0]])
    trans_mat=trans.unsqueeze(0).repeat(pred_rpy.shape[0],1,1).cuda()
    
    transformed_pred = torch.bmm(trans_mat, transformed_pred.unsqueeze(-1)).squeeze(-1) ## (N,3)
    transformed_gt = torch.bmm(trans_mat, transformed_gt.unsqueeze(-1)).squeeze(-1) ## (N,3)
    ################################################################## 
     
    # Compute the dot product for each pair of transformed x-axis vectors
    dot_product = torch.sum(transformed_pred * transformed_gt, dim=-1) # (N)

    return 1 - dot_product # (N)


@weighted_loss
def quaternion_loss(pred, target):
    
    pred = pred.cuda()
    target = target.cuda()

    pred_q, target_q = quaternion(pred, target)
    angle_loss = F.mse_loss(pred_q, target_q).mean(dim=-1)
    
    

    return angle_loss

@weighted_loss
def mse_loss(pred, target):
    pred = pred.cuda()
    target = target.cuda()
    
    angle_loss = 0.1 * (F.mse_loss(pred, target).mean(dim=-1))
    
    return angle_loss


@weighted_loss
def stem_vector_loss(pred, target):
    pred = pred.cuda()
    target = target.cuda()
    
    angle_loss = stem_axis_vector_dot_product(pred, target)


    return 0.2*angle_loss

@LOSSES.register_module()
class Angle3DLoss(nn.Module):
    """Calculate the angle loss of rotated bounding boxes.

    Args:
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.
    """

    def __init__(self, mode='mse', loss_weight=1.0):
        super().__init__()
        
        if mode == 'quat':
            self.loss = quaternion_loss
        elif mode == 'stem_vector':
            self.loss = stem_vector_loss
        else:
            self.loss = mse_loss
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                **kwargs):
        """Forward function of loss calculation.

        Args:
            pred (torch.Tensor): Bbox predictions with shape [..., 9]
                (x, y, z, w, l, h, alpha).
            target (torch.Tensor): Bbox targets (gt) with shape [..., 9]
                (x, y, z, w, l, h, alpha).
            weight (torch.Tensor | float, optional): Weight of loss.
                Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): Method to reduce losses.
                The valid reduction method are 'none', 'sum' or 'mean'.
                Defaults to None.

        Returns:
            torch.Tensor: IoU loss between predictions and targets.
        """
        if weight is not None and not torch.any(weight > 0):
            return pred.sum() * weight.sum()  # 0
        if weight is not None and weight.dim() > 1:
            weight = weight.mean(-1)
        loss = self.loss_weight * self.loss(
            pred,
            target,
            weight,
            **kwargs)


        return loss