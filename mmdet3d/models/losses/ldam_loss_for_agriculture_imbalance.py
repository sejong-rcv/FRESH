# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
# from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss
import numpy as np
from ..builder import LOSSES


@LOSSES.register_module()
class LDAMLossAgriculture(nn.Module):

    def __init__(self,
                  max_m=0.5, 
                  weight=None, 
                  s=30,
                  cls_num_=4):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            activated (bool, optional): Whether the input is activated.
                If True, it means the input has been activated and can be
                treated as probabilities. Else, it should be treated as logits.
                Defaults to False.
        """
        super(LDAMLossAgriculture, self).__init__()
        if cls_num_!=4:
            print("change cls_num_list")
            import pdb;pdb.set_trace()
        self.cls_num_list=[403,568,1622,293]
        m_list = 1.0 / np.sqrt(np.sqrt(self.cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list.tolist())
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self,
                pred,
                target,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
                The target shape support (N,C) or (N,), (N,C) means
                one-hot form.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        import pdb;pdb.set_trace()
        index = torch.zeros_like(pred, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        pred_m = pred - batch_m
    
        output = torch.where(index, pred_m, pred)
        
        return F.cross_entropy(self.s*output, target, weight=self.weight)
        


# @MODELS.register_module()
# class FocalAgriculture(nn.Module):

#     def __init__(self,
#                  use_sigmoid=True,
#                  num_classes=-1,
#                  gamma=2.0,
#                  alpha=0.25,
#                  reduction='mean',
#                  loss_weight=1.0,
#                  activated=False):
#         """`Focal Loss for V3Det <https://arxiv.org/abs/1708.02002>`_

#         Args:
#             use_sigmoid (bool, optional): Whether to the prediction is
#                 used for sigmoid or softmax. Defaults to True.
#             num_classes (int): Number of classes to classify.
#             gamma (float, optional): The gamma for calculating the modulating
#                 factor. Defaults to 2.0.
#             alpha (float, optional): A balanced form for Focal Loss.
#                 Defaults to 0.25.
#             reduction (str, optional): The method used to reduce the loss into
#                 a scalar. Defaults to 'mean'. Options are "none", "mean" and
#                 "sum".
#             loss_weight (float, optional): Weight of loss. Defaults to 1.0.
#             activated (bool, optional): Whether the input is activated.
#                 If True, it means the input has been activated and can be
#                 treated as probabilities. Else, it should be treated as logits.
#                 Defaults to False.
#         """
#         super(FocalAgriculture, self).__init__()
#         assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
#         self.use_sigmoid = use_sigmoid
#         self.num_classes = num_classes
#         self.gamma = gamma
#         self.alpha = alpha
#         self.reduction = reduction
#         self.loss_weight = loss_weight
#         self.activated = activated

#         assert self.num_classes != -1

#         # custom output channels of the classifier
#         self.custom_cls_channels = True
#         # custom activation of cls_score
#         self.custom_activation = True
#         # custom accuracy of the classsifier
#         self.custom_accuracy = True

#     def get_cls_channels(self, num_classes):
#         assert num_classes == self.num_classes
#         return num_classes

#     def get_activation(self, cls_score):

#         fine_cls_score = cls_score[:, :self.num_classes]

#         score_classes = fine_cls_score.sigmoid()

#         return score_classes

#     def get_accuracy(self, cls_score, labels):

#         fine_cls_score = cls_score[:, :self.num_classes]

#         pos_inds = labels < self.num_classes
#         acc_classes = accuracy(fine_cls_score[pos_inds], labels[pos_inds])
#         acc = dict()
#         acc['acc_classes'] = acc_classes
#         return acc

#     def forward(self,
#                 pred,
#                 target,
#                 weight=None,
#                 avg_factor=None,
#                 reduction_override=None):
#         """Forward function.

#         Args:
#             pred (torch.Tensor): The prediction.
#             target (torch.Tensor): The learning label of the prediction.
#             weight (torch.Tensor, optional): The weight of loss for each
#                 prediction. Defaults to None.
#             avg_factor (int, optional): Average factor that is used to average
#                 the loss. Defaults to None.
#             reduction_override (str, optional): The reduction method used to
#                 override the original reduction method of the loss.
#                 Options are "none", "mean" and "sum".

#         Returns:
#             torch.Tensor: The calculated loss
#         """
#         assert reduction_override in (None, 'none', 'mean', 'sum')
#         reduction = (
#             reduction_override if reduction_override else self.reduction)
#         if self.use_sigmoid:

#             num_classes = pred.size(1)
#             target = F.one_hot(target, num_classes=num_classes + 1)
#             target = target[:, :num_classes]
#             calculate_loss_func = py_sigmoid_focal_loss

#             loss_cls = self.loss_weight * calculate_loss_func(
#                 pred,
#                 target,
#                 weight,
#                 gamma=self.gamma,
#                 alpha=self.alpha,
#                 reduction=reduction,
#                 avg_factor=avg_factor)

#         else:
#             raise NotImplementedError
#         return loss_cls