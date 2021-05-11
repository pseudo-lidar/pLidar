## imports 
import argparse
from torch.utils.data import DataLoader
import torch
import numpy as np
from dataloader.e2e_dataset import e2e_dataset


def parse_args():
    msg = 'msg'
    parser = argparse.ArgumentParser(msg)
    parser.add_argument("--batch_size", default='32')
    parser.add_argument("--num_workers", default='8')
    parser.add_argument("--seed", default='326')
    parser.add_argument('--data_dir', default='data/KITTI/kitti_2015/data_scene_flow', type=str, help='Training dataset')
    args = parser.parse_args()

    return args


def train(args):
    # For reproducibility
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    
    # DataLoader
    train_data = e2e_dataset('train')
    def collate_fn(data):
        return data
    train_loader =  DataLoader(dataset=train_data, batch_size=int(args.batch_size), shuffle=True,
                              num_workers=int(args.num_workers), pin_memory=True, drop_last=True , collate_fn=collate_fn)
    #val_data = e2e_dataset('val')
    #val_loader = DataLoader(dataset=val_data, batch_size=args.val_batch_size, shuffle=False,
                            #num_workers=args.num_workers, pin_memory=True, drop_last=False)
    
    #print(args.batch_size)
    for i , data in enumerate(train_loader):
        print(i,data)
        if i == 3 :
            break
    
if __name__ == "__main__":
    args = parse_args()
    train(args)