try:
    import MinkowskiEngine as ME
except ImportError:
    import warnings
    warnings.warn(
        'Please follow `getting_started.md` to install MinkowskiEngine.`')

import torch
from mmcv.cnn import bias_init_with_prob
from mmcv.ops import nms3d, nms3d_normal
from mmcv.runner import BaseModule
from torch import nn
from torch import Tensor
from mmcv.utils import ext_loader
from mmcv.ops.diff_iou_rotated import oriented_box_intersection_2d
from mmdet3d.models.losses import chamfer_distance

ext_module = ext_loader.load_ext('_ext', [
    'iou3d_boxes_overlap_bev_forward', 'iou3d_nms3d_forward',
    'iou3d_nms3d_normal_forward'
])

from mmdet3d.models.builder import HEADS, build_loss
from mmdet.core.bbox.builder import BBOX_ASSIGNERS, build_assigner

from scipy.spatial import ConvexHull
import numpy as np

class plane():
    def __init__(self,p0,n):
        self.p0 = p0
        self.n = n/np.linalg.norm(n)
        
    def __init__(self,p0,p1,p2):
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.v = np.stack([p1-p0,p2-p0]).T
        n = np.cross(self.v[:,0],self.v[:,1])
        self.n = n/np.linalg.norm(n)
    
    def intersect_lines(self, lines, eps=1e-8):
        """Finds intersections with lines given by tuples"""
        m = lines[:,1]-lines[:,0] # gradient
        dot = self.n @ m.T # angles between plane and lines

        valid = abs(dot) > eps # only non parallel lines have valid solutions

        t = (self.n @ (self.p0 - lines[valid,0]).T) / dot[valid] # line parameter
        intersections = lines[valid,0] + (m[valid]*t[..., np.newaxis])
        
        return intersections, t
    
    def project_points(self, points, check=True, eps = 1e-8):
        v = (points-self.p0).T
        dist = self.n @ v
        prj_points = points - self.n * dist[..., np.newaxis]
        
        if check:
            t = np.linalg.inv(self.v.T @ self.v) @ self.v.T @ v
            valid = (0-eps<=t[0]) & (t[0]<=1+eps) & (0-eps<=t[1]) & (t[1]<=1+eps)
            points = points[valid]
            prj_points = prj_points[valid]
            dist = dist[valid]
        return list(zip(abs(dist),points,prj_points))

class OBB():
    def __init__(self, T, dimensions):
        self.corners = np.array([[-0.5, -0.5, -0.5, 1],
                                  [0.5, -0.5, -0.5, 1],
                                  [-0.5, 0.5, -0.5, 1],
                                  [-0.5, -0.5, 0.5, 1],
                                  [0.5, 0.5, 0.5, 1],
                                  [-0.5, 0.5, 0.5, 1],
                                  [0.5, -0.5, 0.5, 1],
                                  [0.5, 0.5, -0.5, 1]]).T
        self.edges = [[0, 1], [1, 7], [7, 2], [2, 0], [3, 6], [6, 4],
                      [4, 5], [5, 3], [0, 3], [1, 6], [7, 4], [2, 5]]
        self.faces = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [4, 5, 6], [4, 5, 7], [4, 6, 7]]
        self.T = T
        self.dimensions = dimensions

    def get_box_points(self):
        return (self.T @ self.corners)[:3].T
    
    def get_box_faces(self):
        cor = self.get_box_points()
        face_array = []
        for p0, p1, p2 in self.faces:
            f = plane(cor[p0], cor[p1], cor[p2])
            face_array.append(f)
        return face_array
    
    def get_box_edges(self):
        cor = self.get_box_points()
        edges = []
        for [e0, e1] in self.edges:
            edges.append([cor[e0], cor[e1]])
        return np.array(edges)
    
    def get_point_indices_within_bounding_box(self, points, eps=1e-10):
        temp = np.linalg.inv(self.T) @ np.vstack((points.T, np.ones(len(points))))
        return np.all(temp[:3] <= 0.5 + eps, 0) & np.all(temp[:3] >= -0.5 - eps, 0)

    def intersect_lines(self, lines, check=True, eps=1e-10):
        poi = np.empty([0, 3])
        for face in self.get_box_faces():
            inters, t = face.intersect_lines(lines, eps)
            poi = np.concatenate([poi, inters])

        if check:
            valid = self.get_point_indices_within_bounding_box(poi)
            poi = poi[valid]
        return poi

    def IoU_v(self, box2, eps=1e-10):
        from scipy.spatial import ConvexHull
        
        poi = self.get_box_points()
        poi = np.vstack((poi, box2.get_box_points()))
        
        edges = box2.get_box_edges()
        poi = np.vstack((poi, self.intersect_lines(edges, False)))
        
        edges = self.get_box_edges()
        poi = np.vstack((poi, box2.intersect_lines(edges, False)))

        valid = (self.get_point_indices_within_bounding_box(poi, eps=eps) &
                 box2.get_point_indices_within_bounding_box(poi, eps=eps))

        try:
            h = ConvexHull(poi[valid])
        except:
            return 0

        intersection = h.volume
        union = self.volume() + box2.volume() - intersection
        IoU = intersection / union
        return IoU

    def volume(self):
        p, r, d = self.get_prd()
        return np.prod(d)

    def get_prd(self):
        p = self.T[:3, 3]
        r = self.T[:3, :3]
        d = np.linalg.norm(r, axis=0)
        r = r / d
        return p, r, d
    
    def pd(self, box2):
        """ returns the distance the centers of the boxes """
        p, r, d = self.get_prd()
        p2, r2, d2 = box2.get_prd()
        return positionDifference(p, p2)


def rotate_3d_bbox(pred):
    """
    Compute rotation for bbox

    Args:
        pred: A tensor of shape (N, 9) containing predicted boxes.
               The last dimension represents (x, y, z, w, h, l, roll, pitch, yaw).
    Returns:
        pred_boxes: rotated bbox (N,8)
    """
    
    N = pred.shape[0]  # Number of predicted boxes

    # Create OBB objects for predicted and target boxes
    pred_boxes = [OBB(create_transformation_matrix(pred[i].detach().cpu().numpy()), 
                    np.array([pred[i][3].detach().cpu().numpy(), 
                                pred[i][4].detach().cpu().numpy(), 
                                pred[i][5].detach().cpu().numpy()])) 
                for i in range(N)]
    return pred_boxes
    
def diff_diou_all_rotated_3d(target, pred,device_type):
    """
    Compute the DIoU loss between predicted and target bounding boxes.

    Args:
        pred: A tensor of shape (9) containing predicted boxes.
               The last dimension represents (x, y, z, w, h, l, roll, pitch, yaw).
        target: A tensor of shape (M, 9) containing target boxes.

    Returns:
        losses: A tensor of DIoU loss values for the pairs of boxes.
    """
    
    
    N = len(pred)  # Number of predicted boxes

    iou_volume = torch.zeros(N, dtype=torch.float32, device=device_type)
    for i in range(N):
        iou_volume[i] = target.IoU_v(pred[i])
    
    
    return iou_volume


def create_transformation_matrix(box_params):
    """
    Create a transformation matrix from the bounding box parameters.

    Args:
        box_params: A list or array containing the bounding box parameters.
                     Expected format: [x, y, z, w, h, l, roll, pitch, yaw].

    Returns:
        A 4x4 transformation matrix.
    """
    x, y, z, w, h, l, roll, pitch, yaw = box_params

    # Create a rotation matrix from roll, pitch, yaw
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])

    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])

    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

    # Combined rotation matrix
    R = R_z @ R_y @ R_x

    # Create the transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array([x, y, z])
    
    return T

def nms3d_angle(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    """Normal 3D NMS function GPU implementation. The overlap of two boxes for
    IoU calculation is defined as the exact overlapping area of the two boxes
    WITH their yaw angle set to 0.

    Args:
        boxes (torch.Tensor): Input boxes with shape (N, 7).
            ([x, y, z, dx, dy, dz, heading]).
        scores (torch.Tensor): Scores of predicted boxes with shape (N).
        iou_threshold (float): Overlap threshold of NMS.

    Returns:
        torch.Tensor: Remaining indices with scores in descending order.
    """
    assert boxes.shape[1] == 9, 'Input boxes shape should be (N, 7)'
    boxes_no_rotate = boxes[:,:7]
    
    return nms3d_normal(boxes_no_rotate, scores, iou_threshold)

@HEADS.register_module()
class TR3DHead(BaseModule):
    def __init__(self,
                 n_classes,
                 in_channels,
                 n_reg_outs,
                 voxel_size,
                 assigner,
                 bbox_loss=dict(type='AxisAlignedIoULoss', reduction='none'),
                 cls_loss=dict(type='FocalLoss', reduction='none'),
                 train_cfg=None,
                 test_cfg=None):
        super(TR3DHead, self).__init__()
        self.voxel_size = voxel_size
        self.assigner = build_assigner(assigner)
        self.bbox_loss = build_loss(bbox_loss)
        self.cls_loss = build_loss(cls_loss)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers(n_classes, in_channels, n_reg_outs)

    def _init_layers(self, n_classes, in_channels, n_reg_outs):
        self.bbox_conv = ME.MinkowskiConvolution(
            in_channels, n_reg_outs, kernel_size=1, bias=True, dimension=3)
        self.cls_conv = ME.MinkowskiConvolution(
            in_channels, n_classes, kernel_size=1, bias=True, dimension=3)

    def init_weights(self):
        nn.init.normal_(self.bbox_conv.kernel, std=.01)
        nn.init.normal_(self.cls_conv.kernel, std=.01)
        nn.init.constant_(self.cls_conv.bias, bias_init_with_prob(.01))

    # per level
    def _forward_single(self, x):
        reg_final = self.bbox_conv(x).features
        reg_distance = torch.exp(reg_final[:, 3:6])
        reg_angle = reg_final[:, 6:]
        bbox_pred = torch.cat((reg_final[:, :3], reg_distance, reg_angle), dim=1)
        cls_pred = self.cls_conv(x).features

        bbox_preds, cls_preds, points = [], [], []
        for permutation in x.decomposition_permutations:
            bbox_preds.append(bbox_pred[permutation])
            cls_preds.append(cls_pred[permutation])
            points.append(x.coordinates[permutation][:, 1:] * self.voxel_size)

        return bbox_preds, cls_preds, points

    def forward(self, x):
        bbox_preds, cls_preds, points = [], [], []
        for i in range(len(x)):
            bbox_pred, cls_pred, point = self._forward_single(x[i])
            bbox_preds.append(bbox_pred)
            cls_preds.append(cls_pred)
            points.append(point)
        return bbox_preds, cls_preds, points

    @staticmethod
    def _bbox_to_loss(bbox):
        """Transform box to the axis-aligned or rotated iou loss format.
        Args:
            bbox (Tensor): 3D box of shape (N, 6) or (N, 7).
        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        # rotated iou loss accepts (x, y, z, w, h, l, heading)
        if bbox.shape[-1] != 6:
            return bbox

        # axis-aligned case: x, y, z, w, h, l -> x1, y1, z1, x2, y2, z2
        return torch.stack(
            (bbox[..., 0] - bbox[..., 3] / 2, bbox[..., 1] - bbox[..., 4] / 2,
             bbox[..., 2] - bbox[..., 5] / 2, bbox[..., 0] + bbox[..., 3] / 2,
             bbox[..., 1] + bbox[..., 4] / 2, bbox[..., 2] + bbox[..., 5] / 2),
            dim=-1)

    @staticmethod
    def _bbox_pred_to_bbox(points, bbox_pred):
        """Transform predicted bbox parameters to bbox.
        Args:
            points (Tensor): Final locations of shape (N, 3)
            bbox_pred (Tensor): Predicted bbox parameters of shape (N, 6)
                or (N, 8).
        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        if bbox_pred.shape[0] == 0:
            return bbox_pred

        x_center = points[:, 0] + bbox_pred[:, 0]
        y_center = points[:, 1] + bbox_pred[:, 1]
        z_center = points[:, 2] + bbox_pred[:, 2]
        base_bbox = torch.stack([
            x_center,
            y_center,
            z_center,
            bbox_pred[:, 3],
            bbox_pred[:, 4],
            bbox_pred[:, 5]], -1)

        # axis-aligned case
        if bbox_pred.shape[1] == 6:
            return base_bbox

        # rotated case: ..., sin(2a)ln(q), cos(2a)ln(q)
        scale = bbox_pred[:, 3] + bbox_pred[:, 4]
        q = torch.exp(
            torch.sqrt(
                torch.pow(bbox_pred[:, 6], 2) + torch.pow(bbox_pred[:, 7], 2)))
        alpha = 0.5 * torch.atan2(bbox_pred[:, 6], bbox_pred[:, 7])
        return torch.stack(
            (x_center, y_center, z_center, scale / (1 + q), scale /
             (1 + q) * q, bbox_pred[:, 5] + bbox_pred[:, 4], alpha),
            dim=-1)

    # per scene
    def _loss_single(self,
                     bbox_preds,
                     cls_preds,
                     points,
                     gt_bboxes,
                     gt_labels,
                     img_meta):
        assigned_ids = self.assigner.assign(points, gt_bboxes, gt_labels, img_meta)
        bbox_preds = torch.cat(bbox_preds)
        cls_preds = torch.cat(cls_preds)
        points = torch.cat(points)

        # cls loss
        n_classes = cls_preds.shape[1]
        pos_mask = assigned_ids >= 0

        if len(gt_labels) > 0:
            cls_targets = torch.where(pos_mask, gt_labels[assigned_ids], n_classes)
        else:
            cls_targets = gt_labels.new_full((len(pos_mask),), n_classes)
        cls_loss = self.cls_loss(cls_preds, cls_targets)

        # bbox loss
        pos_bbox_preds = bbox_preds[pos_mask]
        if pos_mask.sum() > 0:
            pos_points = points[pos_mask]
            pos_bbox_preds = bbox_preds[pos_mask]
            bbox_targets = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)
            pos_bbox_targets = bbox_targets.to(points.device)[assigned_ids][pos_mask]
            if pos_bbox_preds.shape[1] == 6:
                pos_bbox_targets = pos_bbox_targets[:, :6]
            bbox_loss = self.bbox_loss(
                self._bbox_to_loss(
                    self._bbox_pred_to_bbox(pos_points, pos_bbox_preds)),
                self._bbox_to_loss(pos_bbox_targets))            
        else:
            bbox_loss = None
        return bbox_loss, cls_loss, pos_mask

    def _loss(self, bbox_preds, cls_preds, points,
              gt_bboxes, gt_labels, img_metas):
        bbox_losses, cls_losses, pos_masks = [], [], []
        for i in range(len(img_metas)):
            bbox_loss, cls_loss, pos_mask = self._loss_single(
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                points=[x[i] for x in points],
                img_meta=img_metas[i],
                gt_bboxes=gt_bboxes[i],
                gt_labels=gt_labels[i])
            if bbox_loss is not None:
                bbox_losses.append(bbox_loss)
            cls_losses.append(cls_loss)
            pos_masks.append(pos_mask)
        return dict(
            bbox_loss=torch.mean(torch.cat(bbox_losses)),
            cls_loss=torch.sum(torch.cat(cls_losses)) / torch.sum(torch.cat(pos_masks)))

    def forward_train(self, x, gt_bboxes, gt_labels, img_metas):
        bbox_preds, cls_preds, points = self(x)
        return self._loss(bbox_preds, cls_preds, points,
                          gt_bboxes, gt_labels, img_metas)

    def _nms(self, bboxes, scores, img_meta):
        """Multi-class nms for a single scene.
        Args:
            bboxes (Tensor): Predicted boxes of shape (N_boxes, 6) or
                (N_boxes, 7).
            scores (Tensor): Predicted scores of shape (N_boxes, N_classes).
            img_meta (dict): Scene meta data.
        Returns:
            Tensor: Predicted bboxes.
            Tensor: Predicted scores.
            Tensor: Predicted labels.
        """
        n_classes = scores.shape[1]
        yaw_flag = bboxes.shape[1] == 7
        nms_bboxes, nms_scores, nms_labels = [], [], []
        for i in range(n_classes):
            ids = scores[:, i] > self.test_cfg.score_thr
            if not ids.any():
                continue

            class_scores = scores[ids, i]
            class_bboxes = bboxes[ids]
            if yaw_flag:
                nms_function = nms3d
            else:
                class_bboxes = torch.cat(
                    (class_bboxes, torch.zeros_like(class_bboxes[:, :1])),
                    dim=1)
                nms_function = nms3d_normal

            nms_ids = nms_function(class_bboxes, class_scores,
                                   self.test_cfg.iou_thr)
            nms_bboxes.append(class_bboxes[nms_ids])
            nms_scores.append(class_scores[nms_ids])
            nms_labels.append(
                bboxes.new_full(
                    class_scores[nms_ids].shape, i, dtype=torch.long))

        if len(nms_bboxes):
            nms_bboxes = torch.cat(nms_bboxes, dim=0)
            nms_scores = torch.cat(nms_scores, dim=0)
            nms_labels = torch.cat(nms_labels, dim=0)
        else:
            nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores = bboxes.new_zeros((0, ))
            nms_labels = bboxes.new_zeros((0, ))

        if yaw_flag:
            box_dim = 7
            with_yaw = True
        else:
            box_dim = 6
            with_yaw = False
            nms_bboxes = nms_bboxes[:, :6]
        nms_bboxes = img_meta['box_type_3d'](
            nms_bboxes,
            box_dim=box_dim,
            with_yaw=with_yaw,
            origin=(.5, .5, .5))

        return nms_bboxes, nms_scores, nms_labels

    def _get_bboxes_single(self, bbox_preds, cls_preds, points, img_meta):
        scores = torch.cat(cls_preds).sigmoid()
        bbox_preds = torch.cat(bbox_preds)
        points = torch.cat(points)
        max_scores, _ = scores.max(dim=1)

        if len(scores) > self.test_cfg.nms_pre > 0:
            _, ids = max_scores.topk(self.test_cfg.nms_pre)
            bbox_preds = bbox_preds[ids]
            scores = scores[ids]
            points = points[ids]

        boxes = self._bbox_pred_to_bbox(points, bbox_preds)
        boxes, scores, labels = self._nms(boxes, scores, img_meta)
        return boxes, scores, labels

    def _get_bboxes(self, bbox_preds, cls_preds, points, img_metas):
        results = []
        for i in range(len(img_metas)):
            result = self._get_bboxes_single(
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                points=[x[i] for x in points],
                img_meta=img_metas[i])
            results.append(result)
        return results

    def forward_test(self, x, img_metas):
        bbox_preds, cls_preds, points = self(x)
        return self._get_bboxes(bbox_preds, cls_preds, points, img_metas)


@HEADS.register_module()
class TR3DHead_angle(BaseModule):
    def __init__(self,
                 n_classes,
                 in_channels,
                 n_reg_outs,
                 n_angle,
                 voxel_size,
                 assigner,
                 bbox_loss=dict(type='AxisAlignedIoULoss', reduction='none'),
                 cls_loss=dict(type='FocalLoss', reduction='none'),
                 angle_loss=None,
                 train_cfg=None,
                 test_cfg=None):
        super(TR3DHead_angle, self).__init__()
        self.voxel_size = voxel_size
        self.assigner = build_assigner(assigner)
        self.bbox_loss = build_loss(bbox_loss)
        self.cls_loss = build_loss(cls_loss)
        self.angle_loss = build_loss(angle_loss)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers(n_classes, in_channels, n_reg_outs, n_angle)

    def _init_layers(self, n_classes, in_channels, n_reg_outs, n_angle):
        self.bbox_conv = ME.MinkowskiConvolution(
            in_channels, n_reg_outs, kernel_size=1, bias=True, dimension=3)
        self.cls_conv = ME.MinkowskiConvolution(
            in_channels, n_classes, kernel_size=1, bias=True, dimension=3)
        self.angle_conv = ME.MinkowskiConvolution(
            in_channels, n_angle, kernel_size=1, bias=True, dimension=3)

    def init_weights(self):
        nn.init.normal_(self.bbox_conv.kernel, std=.01)
        nn.init.normal_(self.angle_conv.kernel, std=.01)
        nn.init.normal_(self.cls_conv.kernel, std=.01)
        nn.init.constant_(self.cls_conv.bias, bias_init_with_prob(.01))

    # per level
    def _forward_single(self, x):
        reg_final = self.bbox_conv(x).features
        reg_distance = torch.exp(reg_final[:, 3:6])
        bbox_pred = torch.cat((reg_final[:, :3], reg_distance), dim=1)
        cls_pred = self.cls_conv(x).features
        angle_pred = self.angle_conv(x).features

        bbox_preds, cls_preds, angle_preds, points = [], [], [], []
        for permutation in x.decomposition_permutations:
            bbox_preds.append(bbox_pred[permutation])
            cls_preds.append(cls_pred[permutation])
            angle_preds.append(angle_pred[permutation])
            points.append(x.coordinates[permutation][:, 1:] * self.voxel_size)

        return bbox_preds, cls_preds, angle_preds, points

    def forward(self, x):
        bbox_preds, cls_preds, angle_preds, points = [], [], [], []
        for i in range(len(x)):
            bbox_pred, cls_pred, angle_pred, point = self._forward_single(x[i])
            bbox_preds.append(bbox_pred)
            cls_preds.append(cls_pred)
            angle_preds.append(angle_pred)
            points.append(point)

        return bbox_preds, cls_preds, angle_preds, points
    
    @staticmethod
    def _bbox_to_loss(bbox):
        """Transform box to the axis-aligned or rotated iou loss format.
        Args:
            bbox (Tensor): 3D box of shape (N, 6) or (N, 7).
        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        # rotated iou loss accepts (x, y, z, w, h, l, heading)
        if bbox.shape[-1] != 6:
            return bbox

        # axis-aligned case: x, y, z, w, h, l -> x1, y1, z1, x2, y2, z2
        return torch.stack(
            (bbox[..., 0] - bbox[..., 3] / 2, bbox[..., 1] - bbox[..., 4] / 2,
             bbox[..., 2] - bbox[..., 5] / 2, bbox[..., 0] + bbox[..., 3] / 2,
             bbox[..., 1] + bbox[..., 4] / 2, bbox[..., 2] + bbox[..., 5] / 2),
            dim=-1)

    @staticmethod
    def _bbox_pred_to_bbox(points, bbox_pred):
        """Transform predicted bbox parameters to bbox.
        Args:
            points (Tensor): Final locations of shape (N, 3)
            bbox_pred (Tensor): Predicted bbox parameters of shape (N, 6)
                or (N, 8).
        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        if bbox_pred.shape[0] == 0:
            return bbox_pred

        x_center = points[:, 0] + bbox_pred[:, 0]
        y_center = points[:, 1] + bbox_pred[:, 1]
        z_center = points[:, 2] + bbox_pred[:, 2]
        base_bbox = torch.stack([
            x_center,
            y_center,
            z_center,
            bbox_pred[:, 3],
            bbox_pred[:, 4],
            bbox_pred[:, 5]], -1)

        # axis-aligned case
        if bbox_pred.shape[1] == 6:
            return base_bbox
        
        # x,y,z rotation case
        if bbox_pred.shape[1] == 9:
            x_center = points[:, 0] + bbox_pred[:, 0]
            y_center = points[:, 1] + bbox_pred[:, 1]
            z_center = points[:, 2] + bbox_pred[:, 2]
            
            width = bbox_pred[:, 3]
            length = bbox_pred[:, 4]
            height = bbox_pred[:, 5]

            roll = bbox_pred[:, 6]
            pitch = bbox_pred[:, 7]
            yaw = bbox_pred[:, 8]
            
            base_bbox = torch.stack([
                x_center,
                y_center,
                z_center,
                width,
                length,
                height,
                roll,
                pitch,
                yaw
            ], dim=-1)
            
            return base_bbox

        scale = bbox_pred[:, 3] + bbox_pred[:, 4]
        q = torch.exp(
            torch.sqrt(
                torch.pow(bbox_pred[:, 6], 2) + torch.pow(bbox_pred[:, 7], 2)))
        alpha = 0.5 * torch.atan2(bbox_pred[:, 6], bbox_pred[:, 7])
        return torch.stack(
            (x_center, y_center, z_center, scale / (1 + q), scale /
             (1 + q) * q, bbox_pred[:, 5] + bbox_pred[:, 4], alpha),
            dim=-1)

    # per scene
    def _loss_single(self,
                     bbox_preds,
                     cls_preds,
                     angle_preds,
                     points,
                     gt_bboxes,
                     gt_labels,
                     img_meta):
        assigned_ids = self.assigner.assign(points, gt_bboxes, gt_labels, img_meta)
        bbox_preds = torch.cat(bbox_preds)
        cls_preds = torch.cat(cls_preds)
        angle_preds = torch.cat(angle_preds)
        points = torch.cat(points)

        # cls loss
        n_classes = cls_preds.shape[1]
        pos_mask = assigned_ids >= 0

        if len(gt_labels) > 0:
            cls_targets = torch.where(pos_mask, gt_labels[assigned_ids], n_classes)
        else:
            cls_targets = gt_labels.new_full((len(pos_mask),), n_classes)
        cls_loss = self.cls_loss(cls_preds, cls_targets)

        # bbox loss
        pos_bbox_preds = bbox_preds[pos_mask]
        if pos_mask.sum() > 0:
            pos_points = points[pos_mask]
            pos_bbox_preds = bbox_preds[pos_mask]
            bbox_targets = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)
            pos_bbox_targets = bbox_targets.to(points.device)[assigned_ids][pos_mask]
            if pos_bbox_preds.shape[1] == 6:
                pos_bbox_targets = pos_bbox_targets[:, :6]
            if pos_bbox_preds.shape[1] == 9:
                pos_bbox_targets = pos_bbox_targets[:, :9]
            bbox_loss = self.bbox_loss(
                self._bbox_to_loss(
                    self._bbox_pred_to_bbox(pos_points, pos_bbox_preds)),
                self._bbox_to_loss(pos_bbox_targets))            
        else:
            bbox_loss = None
            
        # angle loss
        pos_angle_preds = angle_preds[pos_mask]
        if pos_mask.sum() > 0:
            pos_points = points[pos_mask]
            pos_angle_preds = angle_preds[pos_mask]
            angle_targets = gt_bboxes.tensor[:, 6:]
            pos_angle_targets = angle_targets.to(points.device)[assigned_ids][pos_mask]
            angle_loss = self.angle_loss(
                pos_angle_preds, pos_angle_targets
            )
        else:
            angle_loss = None
            
        return bbox_loss, cls_loss, angle_loss, pos_mask

    def _loss(self, bbox_preds, cls_preds, angle_preds, points,
              gt_bboxes, gt_labels, img_metas):
        bbox_losses, cls_losses, angle_losses, pos_masks = [], [], [], []
        for i in range(len(img_metas)):
            bbox_loss, cls_loss, angle_loss, pos_mask = self._loss_single(
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                angle_preds = [x[i] for x in angle_preds],
                points=[x[i] for x in points],
                img_meta=img_metas[i],
                gt_bboxes=gt_bboxes[i],
                gt_labels=gt_labels[i])
            if bbox_loss is not None:
                bbox_losses.append(bbox_loss)
            cls_losses.append(cls_loss)
            angle_losses.append(angle_loss)
            pos_masks.append(pos_mask)

        if len(bbox_losses) == 0:
            bbox_losses = torch.tensor([0], dtype=torch.float32) 
            bbox_losses = torch.mean(bbox_losses)
            
            return dict(
                bbox_loss=torch.mean(bbox_losses),
                cls_loss=torch.sum(torch.cat(cls_losses)) / torch.sum(torch.cat(pos_masks)))
        return dict(
            bbox_loss=torch.mean(torch.cat(bbox_losses)),
            cls_loss=torch.sum(torch.cat(cls_losses)) / torch.sum(torch.cat(pos_masks)),
            angle_loss=torch.mean(torch.stack(angle_losses)))

    def forward_train(self, x, gt_bboxes, gt_labels, img_metas):
        bbox_preds, cls_preds, angle_preds, points = self(x)
        return self._loss(bbox_preds, cls_preds, angle_preds, points,
                          gt_bboxes, gt_labels, img_metas)

    def _nms(self, bboxes, scores, img_meta):
        """Multi-class nms for a single scene.
        Args:
            bboxes (Tensor): Predicted boxes of shape (N_boxes, 9)
            scores (Tensor): Predicted scores of shape (N_boxes, N_classes).
            img_meta (dict): Scene meta data.
        Returns:
            Tensor: Predicted bboxes.
            Tensor: Predicted scores.
            Tensor: Predicted labels.
        """
        n_classes = scores.shape[1]
        
        if bboxes.shape[1] == 9:
            yaw_flag = bboxes.shape[1] == 9
        else:
            yaw_flag = bboxes.shape[1] == 7
        nms_bboxes, nms_scores, nms_labels = [], [], []
        for i in range(n_classes):
            ids = scores[:, i] > self.test_cfg.score_thr
            if not ids.any():
                continue

            class_scores = scores[ids, i]
            class_bboxes = bboxes[ids]
            if yaw_flag:
                if bboxes.shape[1] == 9:
                    nms_function = nms3d_angle
                else:
                    nms_function = nms3d
                    
            else:
                class_bboxes = torch.cat(
                    (class_bboxes, torch.zeros_like(class_bboxes[:, :1])),
                    dim=1)
                nms_function = nms3d_normal
            nms_ids = nms_function(class_bboxes, class_scores,
                                   self.test_cfg.iou_thr)
            nms_bboxes.append(class_bboxes[nms_ids])
            nms_scores.append(class_scores[nms_ids])
            nms_labels.append(
                bboxes.new_full(
                    class_scores[nms_ids].shape, i, dtype=torch.long))

        if len(nms_bboxes):
            nms_bboxes = torch.cat(nms_bboxes, dim=0)
            nms_scores = torch.cat(nms_scores, dim=0)
            nms_labels = torch.cat(nms_labels, dim=0)
        else:
            nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores = bboxes.new_zeros((0, ))
            nms_labels = bboxes.new_zeros((0, ))
        if yaw_flag:
            if bboxes.shape[1] == 9:
                box_dim = 9
                with_yaw = True
                
            else:
                box_dim = 7
                with_yaw = True 
        else:
            box_dim = 6
            with_yaw = False
            nms_bboxes = nms_bboxes[:, :6]
        nms_bboxes = img_meta['box_type_3d'](
            nms_bboxes,
            box_dim=box_dim,
            with_yaw=with_yaw,
            origin=(.5, .5, .5))

        return nms_bboxes, nms_scores, nms_labels

    def _get_bboxes_single(self, bbox_preds, cls_preds, angle_preds, points, img_meta):
        scores = torch.cat(cls_preds).sigmoid()
        bbox_preds = torch.cat(bbox_preds)
        points = torch.cat(points)
        angle_preds = torch.cat(angle_preds)
        max_scores, _ = scores.max(dim=1)

        if len(scores) > self.test_cfg.nms_pre > 0:
            _, ids = max_scores.topk(self.test_cfg.nms_pre)
            bbox_preds = bbox_preds[ids]
            angles = angle_preds[ids]
            scores = scores[ids]
            points = points[ids]

        boxes = self._bbox_pred_to_bbox(points, bbox_preds)
        boxes = torch.cat((boxes, angles), dim=1)
        boxes, scores, labels = self._nms(boxes, scores, img_meta)
        return boxes, scores, labels

    def _get_bboxes(self, bbox_preds, cls_preds, angle_preds, points, img_metas):
        results = []
        for i in range(len(img_metas)):
            result = self._get_bboxes_single(
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                angle_preds=[x[i] for x in angle_preds],
                points=[x[i] for x in points],
                img_meta=img_metas[i])
            results.append(result)
        return results

    def forward_test(self, x, img_metas):
        bbox_preds, cls_preds, angle_preds, points = self(x)
        
        return self._get_bboxes(bbox_preds, cls_preds, angle_preds, points, img_metas)

@BBOX_ASSIGNERS.register_module() 
class TR3DAssigner:
    def __init__(self, top_pts_threshold, label2level):
        self.top_pts_threshold = top_pts_threshold
        self.label2level = label2level

    @torch.no_grad()
    def assign(self, points, gt_bboxes, gt_labels, img_meta):
        # -> object id or -1 for each point
        float_max = points[0].new_tensor(1e8)
        levels = torch.cat([points[i].new_tensor(i, dtype=torch.long).expand(len(points[i]))
                            for i in range(len(points))])
        points = torch.cat(points)
        n_points = len(points)
        n_boxes = len(gt_bboxes)


        if len(gt_labels) == 0:
            return gt_labels.new_full((n_points,), -1)
        
        # z-axis to gravity center
        boxes = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)

        if boxes.shape[1] == 9:
            boxes = boxes.to(points.device).expand(n_points, n_boxes, 9)
        else:
            boxes = boxes.to(points.device).expand(n_points, n_boxes, 7)
        points = points.unsqueeze(1).expand(n_points, n_boxes, 3)

        # condition 1: fix level for label
        label2level = gt_labels.new_tensor(self.label2level)
        label_levels = label2level[gt_labels].unsqueeze(0).expand(n_points, n_boxes)
        point_levels = torch.unsqueeze(levels, 1).expand(n_points, n_boxes)
        level_condition = label_levels == point_levels

        # condition 2: keep topk location per box by center distance
        center = boxes[..., :3]
        center_distances = torch.sum(torch.pow(center - points, 2), dim=-1)
        center_distances = torch.where(level_condition, center_distances, float_max)
        topk_distances = torch.topk(center_distances,
                                    min(self.top_pts_threshold + 1, len(center_distances)),
                                    largest=False, dim=0).values[-1]
        topk_condition = center_distances < topk_distances.unsqueeze(0)

        # condition 3.0: only closest object to point
        center_distances = torch.sum(torch.pow(center - points, 2), dim=-1)
        _, min_inds_ = center_distances.min(dim=1)

        # condition 3: min center distance to box per point
        center_distances = torch.where(topk_condition, center_distances, float_max)
        min_values, min_ids = center_distances.min(dim=1)
        min_inds = torch.where(min_values < float_max, min_ids, -1)
        min_inds = torch.where(min_inds == min_inds_, min_ids, -1)

        return min_inds




    