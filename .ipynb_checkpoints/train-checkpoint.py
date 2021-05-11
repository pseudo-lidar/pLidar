## imports 
import configargparse
from torch.utils.data import DataLoader
import torch




def parse_args():
    parser = configargparse.ArgParser(
        description="Train E2E")
    parser.add_argument("--batch_size", default='32')
    parser.add_argument("--num_workers", default='8')
    parser.add_argument("--seed", default='326')
    parser.add_argument('--data_dir', default='data/KITTI/kitti_2015/data_scene_flow', type=str, help='Training dataset')
    args = parser.parse_args()
    for arg in vars(args):
        if getattr(args, arg) == 'None':
            setattr(args, arg, None)
    return args


def train(args):
    # For reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # DataLoader
    train_data = e2e_dataset(args.data_dir,'train')
    train_loader =  DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_data = e2e_dataset(args.data_dir,'val')
    val_loader = DataLoader(dataset=val_data, batch_size=args.val_batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False)
if __name__ == "__main__":
    args = parse_args()
    train(args)