from __future__ import absolute_import
import numpy as np
import os
import torch
import sys
sys.path.append('/notebooks/aanet')
import nets 
import torch.nn.functional as F

def filter_specific_params(kv):
    specific_layer_name = ['offset_conv.weight', 'offset_conv.bias']
    for name in specific_layer_name:
        if name in kv[0]:
            return True
    return False


def filter_base_params(kv):
    specific_layer_name = ['offset_conv.weight', 'offset_conv.bias']
    for name in specific_layer_name:
        if name in kv[0]:
            return False
    return True

def load_pretrained( net , path ):
    state = torch.load(path, map_location='cuda')

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    
    weights = state['state_dict'] if 'state_dict' in state.keys() else state
    
    for k, v in weights.items():
        name = k[7:] if 'module' in k and not resume else k
        new_state_dict[name] = v
    
    net.load_state_dict(new_state_dict)  # optimizer has no argument `strict`

class aanet_interface():
    def __init__(self , pretrained_path):
        device = torch.device('cuda')
        self.aanet = nets.AANet(192,
                       num_downsample=2,
                       feature_type='ganet' ,
                       no_feature_mdconv=False,
                       feature_pyramid=True,
                       feature_pyramid_network=False,
                       feature_similarity='correlation',
                       aggregation_type='adaptive',
                       num_scales=3,
                       num_fusions=6,
                       num_stage_blocks=1,
                       num_deform_blocks=3,
                       no_intermediate_supervision=False,
                       refinement_type='hourglass' ,
                       mdconv_dilation=2,
                       deformable_groups=2).to(device)
        
        specific_params = list(filter(filter_specific_params,
                                  self.aanet.named_parameters()))
        base_params = list(filter(filter_base_params,
                              self.aanet.named_parameters()))
        specific_params = [kv[1] for kv in specific_params]  # kv is a tuple (key, value)
        base_params = [kv[1] for kv in base_params]

        specific_lr = 1e-3 * 0.1
        params_group = [
            {'params': base_params, 'lr': 1e-3},
            {'params': specific_params, 'lr': specific_lr},
        ]

        self.optimizer = torch.optim.Adam(params_group, weight_decay=1e-4)
        #load_pretrained(optimizer  ,pretrained_path  )
        milestones = [400,600,800,900]
        #need to change last epoch in case of using checkpoints
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                milestones=milestones,
                                                                gamma=0.5,
                                                                last_epoch=-1)
        
        load_pretrained(self.aanet ,pretrained_path  )
        self.aanet.train()
        
    def aanet_train_batch(self,left , right , gt_disp):
        mask = (gt_disp > 0) & (gt_disp < 192)
        pred_disp_pyramid = self.aanet(left, right)
        
        disp_loss = 0
        pyramid_loss = []
        # Loss weights
        if len(pred_disp_pyramid) == 5:
            pyramid_weight = [1 / 3, 2 / 3, 1.0, 1.0, 1.0]  # AANet and AANet+
        elif len(pred_disp_pyramid) == 4:
            pyramid_weight = [1 / 3, 2 / 3, 1.0, 1.0]
        elif len(pred_disp_pyramid) == 3:
            pyramid_weight = [1.0, 1.0, 1.0]  # 1 scale only
        elif len(pred_disp_pyramid) == 1:
            pyramid_weight = [1.0]  # highest loss only
        else:
            raise NotImplementedError
        for k in range(len(pred_disp_pyramid)):
            pred_disp = pred_disp_pyramid[k]
            weight = pyramid_weight[k]
            if pred_disp.size(-1) != gt_disp.size(-1):
                pred_disp = pred_disp.unsqueeze(1)  # [B, 1, H, W]
                pred_disp = F.interpolate(pred_disp, size=(gt_disp.size(-2), gt_disp.size(-1)),
                                              mode='bilinear', align_corners=False) * (gt_disp.size(-1) / pred_disp.size(-1))
                pred_disp = pred_disp.squeeze(1)  # [B, H, W]

            curr_loss = F.smooth_l1_loss(pred_disp[mask], gt_disp[mask],
                                             reduction='mean')
            disp_loss += weight * curr_loss
            pyramid_loss.append(curr_loss)
        self.optimizer.zero_grad()
        pred_disp = pred_disp_pyramid[-1]
        return disp_loss , pred_disp
    def opt_step(self):
        self.optimizer.step()