import torch
import PIL
from PIL import Image
import kitti_util as util
import calibration
from depth_lidar import get_lidar_pc
#from det3d.torchie.trainer.trainer import example_to_device


def COR(infos, idx, depth, pipeline):
    
    res_batch = list()
    
    res_temp = {
            "type": "KittiDataset",
            "lidar": {
                "type": "lidar",
                "points": None,
                "ground_plane": None,
                "annotations": None,  # include centered gt_boxes and gt_names
                "names": None,        # 'Car'
                "targets": None,      # include cls_labels & reg_targets
            },
            "metadata": {
                "image_prefix": "/notebooks/KITTI_DATASET",
                "num_point_features": 4,
                "image_idx": None,
                "image_shape": None,
                "token": None,
            },
            "calib": None,            # R0_rect, Tr_velo_to_cam, P2
            "cam": {
                "annotations": None,  # include 2d bbox and gt_names
            },
            "mode": "train",
        }
    
    for i in range(depth.shape[0]):
#         print(res_list)
        res_temp['metadata']['image_idx'] = infos[idx[i]]["image"]["image_idx"]
        res_temp['metadata']['image_shape'] = infos[idx[i]]["image"]["image_shape"]
        res_temp['metadata']['token'] = str(infos[idx[i]]["image"]["image_idx"])
        
#         calib = kitti_util.Calibration(infos[idx[i]]['calib'])
        calib_info=util.Calib(calibration.Calibration(infos[idx[i]]['calib']))
        lidar = get_lidar_pc(depth[i], calib_info, res_temp['metadata']['image_shape'], max_high=1)
#         lidar = cor.project_disp_to_points(calib, depth[i],1)
#         lidar= torch.cat((lidar, torch.ones((lidar.shape[0], 1)).cuda()), dim = 1)
#         lidar= cor.gen_sparse_points(lidar)
        res_temp['lidar']['points'] = lidar
        
        res, _ = pipeline(res_temp, infos[idx[i]])
        
        
        
        res_batch.append(res)
        
        
    return res_batch