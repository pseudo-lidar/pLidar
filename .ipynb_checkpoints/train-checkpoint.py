## imports 
import argparse
from torch.utils.data import DataLoader
import torch
import numpy as np
from dataloader.e2e_dataset import e2e_dataset
from aanet_interface import aanet_interface as aanet_interface
import sys
sys.path.append('/notebooks/cor')
from cor_interface import COR  
from cia_interface import ODModel
def parse_args():
    msg = 'msg'
    parser = argparse.ArgumentParser(msg)
    parser.add_argument("--num_epochs", default=1)
    parser.add_argument("--batch_size", default=1)
    parser.add_argument("--num_workers", default=8)
    parser.add_argument("--seed", default=326)
    parser.add_argument('--data_dir', default='data/KITTI/kitti_2015/data_scene_flow', type=str, help='Training dataset')
    
    parser.add_argument('--pkl_path', default='/notebooks/cia/kitti_infos_train.pkl', type=str, help='infos pkl path')
    
    parser.add_argument('--aanet_pretrained_path', default='pretrained/aanet+_sceneflow-d3e13ef0.pth', type=str, help='aanet pretrained path')
    args = parser.parse_args()

    return args


def train(args):
    # For reproducibility
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = torch.device('cuda')
    # DataLoader
    train_data = e2e_dataset('train',384,1248)
    train_loader =  DataLoader(dataset=train_data, batch_size=int(args.batch_size), shuffle=True,
                              num_workers=int(args.num_workers), pin_memory=True, drop_last=True)
    infos = train_data.getinfos(args.pkl_path)
    #val_data = e2e_dataset('val')
    #val_loader = DataLoader(dataset=val_data, batch_size=args.val_batch_size, shuffle=False,
                            #num_workers=args.num_workers, pin_memory=True, drop_last=False)
    
    #print(args.batch_size)
    AANET_interface = aanet_interface(args.aanet_pretrained_path)
    CIA_interface =  ODModel( "/notebooks/cia/examples/second/configs/kitti_car_vfev3_spmiddlefhd_rpn1_mghead_syncbn.py", loader_len = len(train_loader))
    pipeline = CIA_interface.get_pipline()
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(0 , args.num_epochs):
        CIA_interface.before_train_epoch()
        for i , sample in enumerate(train_loader):
            left = sample['left'].to(device)  # [B, 3, H, W]
            right = sample['right'].to(device)
            gt_disp = sample['disp'].to(device)  # [B, H, W]
            
            disp_loss , pred_disp = AANET_interface.aanet_train_batch(left , right , gt_disp)
            #print(pred_disp)
            #print(disp_loss)
           # print(pred_disp.size)
            print("**************************************: iteration: %d" , i)
            batch = COR(infos, sample["idx"], pred_disp ,pipeline )
            print(type(batch))
            
            output =  CIA_interface.cia_forward(batch ,epoch ,i)
            #output of COR to CIA
            #get CIA loss
            total_loss = output["loss"] + 0.1*disp_loss
            total_loss.backward()
            CIA_interface.clip_grad()
            CIA_interface.optimizer_step()
            print("Total Loss: ",total_loss , "OD Loss: " , output["loss"] , "disp Loss:",disp_loss)
            AANET_interface.opt_step()
            #cia optimizer.step()
        CIA_interface.after_train_epoch()
    
if __name__ == "__main__":
    args = parse_args()
    train(args)