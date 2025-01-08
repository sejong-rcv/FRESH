import re
from copy import deepcopy
from os import path as osp

import mmcv
import numpy as np
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmdet3d.core import (Box3DMode, CameraInstance3DBoxes, Coord3DMode,
                          DepthInstance3DBoxes, LiDARInstance3DBoxes,
                          show_multi_modality_result, show_result,
                          show_seg_result)
from mmdet3d.core.bbox import get_box_type
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger
from mmdet3d.apis import init_model

def inference_multi_modality_detector(model, pcd, image, ann_file):
    """Inference point cloud with the multi-modality detector.

    Args:
        model (nn.Module): The loaded detector.
        pcd (str): Point cloud files.
        image (str): Image files.
        ann_file (str): Annotation files.

    Returns:
        tuple: Predicted results and data from pipeline.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = deepcopy(cfg.data.test.pipeline)
    test_pipeline = Compose(test_pipeline)
    box_type_3d, box_mode_3d = get_box_type(cfg.data.test.box_type_3d)
    # get data info containing calib
    data_infos = mmcv.load(ann_file)
    image_idx = int(re.findall(r'\d+', image)[-1])  # xxx/sunrgbd_000017.jpg
    for x in data_infos:
        if int(x['image']['image_idx']) != image_idx:
            continue
        info = x
        break
    data = dict(
        pts_filename=pcd,
        img_prefix=osp.dirname(image),
        img_info=dict(filename=osp.basename(image)),
        box_type_3d=box_type_3d,
        box_mode_3d=box_mode_3d,
        img_fields=[],
        bbox3d_fields=[],
        pts_mask_fields=[],
        pts_seg_fields=[],
        bbox_fields=[],
        mask_fields=[],
        seg_fields=[])
    data = test_pipeline(data)

    # TODO: this code is dataset-specific. Move lidar2img and
    #       depth2img to .pkl annotations in the future.
    # LiDAR to image conversion
    if box_mode_3d == Box3DMode.LIDAR:
        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        P2 = info['calib']['P2'].astype(np.float32)
        lidar2img = P2 @ rect @ Trv2c
        data['img_metas'][0].data['lidar2img'] = lidar2img
    # Depth to image conversion
    elif box_mode_3d == Box3DMode.DEPTH:
        rt_mat = info['calib']['Rt']
        # follow Coord3DMode.convert_point
        rt_mat = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]
                           ]) @ rt_mat.transpose(1, 0)
        depth2img = info['calib']['K'] @ rt_mat
        data['img_metas'][0].data['depth2img'] = depth2img

    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device.index])[0]
    else:
        # this is a workaround to avoid the bug of MMDataParallel
        data['img_metas'] = data['img_metas'][0].data
        data['points'] = data['points'][0].data
        data['img'] = data['img'][0].data

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)

    return result, data

def show_proj_det_result_meshlab(data,
                                 result,
                                 out_dir,
                                 score_thr=0.0,
                                 show=False,
                                 snapshot=False):
    """Show result of projecting 3D bbox to 2D image by meshlab."""
    assert 'img' in data.keys(), 'image data is not provided for visualization'

    img_filename = data['img_metas'][0][0]['filename']
    file_name = osp.split(img_filename)[-1].split('.')[0]

    # read from file because img in data_dict has undergone pipeline transform
    img = mmcv.imread(img_filename)
    img2 = deepcopy(img)

    if 'pts_bbox' in result[0].keys():
        result[0] = result[0]['pts_bbox']
    elif 'img_bbox' in result[0].keys():
        result[0] = result[0]['img_bbox']
    pred_bboxes = result[0]['boxes_3d'].tensor.numpy()
    pred_scores = result[0]['scores_3d'].numpy()
    ##########################################################
    gt_bboxes = []
    if 'gt_boxes_upright_depth' in data.keys():
        gt_bboxes=data['gt_boxes_upright_depth'].cpu().numpy()[0]
    ##########################################################

    # filter out low score bboxes for visualization
    if score_thr > 0:
        inds = pred_scores > score_thr
        pred_bboxes = pred_bboxes[inds]

    box_mode = data['img_metas'][0][0]['box_mode_3d']
    if box_mode == Box3DMode.LIDAR:
        if 'lidar2img' not in data['img_metas'][0][0]:
            raise NotImplementedError(
                'LiDAR to image transformation matrix is not provided')

        show_bboxes = LiDARInstance3DBoxes(pred_bboxes, origin=(0.5, 0.5, 0))

        show_multi_modality_result(
            img,
            None,
            show_bboxes,
            data['img_metas'][0][0]['lidar2img'],
            out_dir,
            file_name,
            box_mode='lidar',
            show=show)
    elif box_mode == Box3DMode.DEPTH:

        show_bboxes = DepthInstance3DBoxes(pred_bboxes, origin=(1, 0, 1)) ## origin=(0.5, 0.5, 0)


        ##########################################################
        if len(gt_bboxes) !=0:
            show_gt_bboxes = DepthInstance3DBoxes(gt_bboxes, origin=(1, 0, 1))

            # cam2img=data['img_metas'][0][0]['depth2img']
            # import pdb; pdb.set_trace()
            cam2img=data['calib']['K'][0] # no .T!!!!
            # cam2img=np.array([[649.0014,   0.    , 634.287 ],
            #                     [  0.    , 649.0014, 403.859 ],
            #                     [  0.    ,   0.    ,   1.    ]])

            # cam2img=np.array([[449.965,   0.    ,  342.437],
            #                     [  0.    , 445.364, 247.496],
            #                     [  0.    ,   0.    ,   1. *0.5]])
            # import pdb;pdb.set_trace()
            cam2img[0][0]*=-1
            # print(cam2img)
            # if cam2img[0][0]==457.25 :
            #     print("L515")
            #     import pdb;pdb.set_trace()
            #     cam2img[0][0]*=-1


            if not isinstance(cam2img, torch.Tensor):
                cam2img = torch.from_numpy(np.array(cam2img))
            assert (cam2img.shape == torch.Size([3, 3])
                    or cam2img.shape == torch.Size([4, 4]))
            # import pdb;pdb.set_trace()
            # cam2img = torch.from_numpy(np.array([[649.0, 0., 0.],[0.,649.0, 0 ],[634.3, 403.9, 1]]))
            cam2img = cam2img.float().cpu()


            from mmdet3d.core.bbox import points_cam2img
            from mmdet3d.models import apply_3d_transformation
            corners_3d_gt = show_gt_bboxes.corners
            num_gt = corners_3d_gt.shape[0]
            point_3d_gt = corners_3d_gt.reshape(-1,3)

            corners_3d_pred = show_bboxes.corners
            num_pred = corners_3d_pred.shape[0]
            point_3d_pred = corners_3d_pred.reshape(-1,3)


            # xyz_gt = apply_3d_transformation(point_3d_gt, 'DEPTH', data['img_metas'])
            # uv_gt = points_cam2img(xyz_gt,data['img_metas'][0][0]['depth2img'])
            # uv_gt = (uv_gt-1).round()
            # xyz_pred = apply_3d_transformation(point_3d_pred, 'DEPTH', data['img_metas'])
            # uv_pred = points_cam2img(xyz_pred,data['img_metas'][0][0]['depth2img'])
            # uv_pred = (uv_pred-1).round()

            uv_gt = points_cam2img(point_3d_gt,cam2img)
            uv_gt = (uv_gt-1).round()
            uv_pred = points_cam2img(point_3d_pred,cam2img)
            uv_pred = (uv_pred-1).round()

            imgfov_pts_2d_gt = uv_gt[..., :2].reshape(num_gt, 8, 2).numpy()
            imgfov_pts_2d_pred = uv_pred[..., :2].reshape(num_pred, 8, 2).numpy()

            ## for GT
            img2 = plot_rect3d_on_img(img2, num_gt,imgfov_pts_2d_gt,color=(0,0,255),thickness=2)
            img2 = plot_rect3d_on_img(img2, num_pred,imgfov_pts_2d_pred,color=(255,0,0),thickness=2)
            import cv2
            cv2.imwrite("/home/rcv/workspace/shlee_workspace/vis_results/tmp.png",img2)
            # show_multi_modality_result(img,show_gt_bboxes,show_bboxes,None,out_dir,file_name,box_mode='depth',img_metas=data['img_metas'][0][0],show=show) 

            # import pdb;pdb.set_trace()

            show_multi_modality_result(
                img,
                show_gt_bboxes,
                show_bboxes,
                None,
                out_dir,
                file_name,
                box_mode='depth',
                img_metas=data['img_metas'][0][0],
                show=show) 
        else: ## original
            show_multi_modality_result(
                img,
                None,
                show_bboxes,
                None,
                out_dir,
                file_name,
                box_mode='depth',
                img_metas=data['img_metas'][0][0],
                show=show)
    elif box_mode == Box3DMode.CAM:
        if 'cam2img' not in data['img_metas'][0][0]:
            raise NotImplementedError(
                'camera intrinsic matrix is not provided')

        show_bboxes = CameraInstance3DBoxes(
            pred_bboxes, box_dim=pred_bboxes.shape[-1], origin=(0.5, 1.0, 0.5))

        show_multi_modality_result(
            img,
            None,
            show_bboxes,
            data['img_metas'][0][0]['cam2img'],
            out_dir,
            file_name,
            box_mode='camera',
            show=show)
    else:
        raise NotImplementedError(
            f'visualization of {box_mode} bbox is not supported')

    return file_name

if __name__ == "__main__":
    
    cfg_file = '/home/dgkim/workspace/tr3d/mmdetection3d/configs/tr3d/tr3d-ff-vit_smartfarm.py'
    ckpt_file = '/home/dgkim/workspace/tr3d/mmdetection3d/work_dirs/tr3d-ff-vit_smartfarm-v2/epoch_30.pth'
    image = '/home/dgkim/workspace/tr3d/data/smartfarm/smartfarm_trainval/image/000688.jpg'
    ann_file = '/home/dgkim/workspace/tr3d/data/smartfarm/smartfarm_infos_val.pkl'
    pcd = '/home/dgkim/workspace/tr3d/data/smartfarm/points/000688.bin'
    out_dir = '/home/dgkim/workspace/tr3d/mmdetection3d/visualization/'
    
    model = init_model(cfg_file, ckpt_file, device='cuda:0')

    result, data = inference_multi_modality_detector(model, pcd, image, ann_file)
    
    #import pdb;pdb.set_trace()
    
    show_proj_det_result_meshlab(data, result, out_dir)