import cv2
import argparse
import mmcv
from image_vis import (draw_camera_bbox3d_on_img, draw_depth_bbox3d_on_img,
                        draw_lidar_bbox3d_on_img)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno',
                        default='/home/dgkim/workspace/tr3d/data/smartfarm/smartfarm_infos_val.pkl')
    # parser.add_argument('--dest_path', required = True)
    # parser.add_argument('--size_float', default = 4, type=int)
    args = parser.parse_args()
    
    return args

# class AgricultureDataset(Custom3DDataset):

#     CLASSES = ('Turning', 'Ripe', 'Over-Ripe')

#     def __init__(self,
#                  data_root,
#                  ann_file,
#                  **kwargs):
#         super().__init__(
#             data_root=data_root,
#             ann_file=ann_file,
#             test_mode=test_mode,
#              **kwargs)

#     def get_data_info(self, index):

#         import pdb;pdb.set_trace()
#         info = self.data_infos[index]

# def vis_bbox_to_img():
    
    
if __name__ == '__main__':
    args = parse_args()
    anno = mmcv.load(args.anno, file_format='pkl')
    import pdb;pdb.set_trace()
    AgricultureDataset(data_root = 'data/Agriculture/',
                       ann_file = args.anno
                       )
    