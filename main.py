from GAUGAN import GAUGAN
import argparse
import utils
import torch 

def parse_args():
    parser = argparse.ArgumentParser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--device', type=str, default=device)
    parser.add_argument('--img_path', type=str, default='/data1/jjm/SPADE_pytorch/DATASET/celebAHQ/train/origin')
    parser.add_argument('--seg_path', type=str, default='/data1/jjm/SPADE_pytorch/DATASET/celebAHQ/train/segmen')
    parser.add_argument('--test_img_path', type=str, default='/data1/jjm/SPADE_pytorch/DATASET/celebAHQ/test/origin')
    parser.add_argument('--test_seg_path', type=str, default='/data1/jjm/SPADE_pytorch/DATASET/celebAHQ/test/segmen')
    parser.add_argument('--batch_size', type=int, default=7, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=4, help='test batch size for image save')
    parser.add_argument('--num_workers', type=int, default=4, help='worker number for data load')
    parser.add_argument('--lr_G', type=float, default=0.0001, help='initial learning rate for Generator')
    parser.add_argument('--lr_D', type=float, default=0.0004, help='initial learning rate for Discriminator')
    parser.add_argument('--beta_1', type=float, default=0, help='beta2 for Adam')
    parser.add_argument('--beta_2', type=float, default=0.9, help='beta2 for Adam')
    parser.add_argument('--total_step', type=int, default=5_000_000, help='total step for training')
    parser.add_argument('--n_critic', type=int, default=1, help='discriminator and generator update ratio')
    parser.add_argument('--n_save', type=int, default=10000, help='save interval btw save ckeck points on step')
    parser.add_argument('--lambda_kl', type=float, default=0.01, help='weight of kl loss')
    parser.add_argument('--lambda_fm', type=float, default=10, help='weight of fm loss on generator')
    parser.add_argument('--lambda_vgg', type=float, default=10, help='weight of vgg loss')
    parser.add_argument('--grid_n_row', type=int, default=5, help='number of row of save image')
    parser.add_argument('--n_save_image', type=int, default=1000, help='save interval btw save images on step')
    parser.add_argument('--GAN_G_loss_type', type=str, default='hinge', help='Loss type of G, ["lsgan" / "hinge" / "origin"]')
    parser.add_argument('--GAN_D_loss_type', type=str, default='hinge', help='Loss type of D, ["lsgan" / "hinge" / "origin"]')
    parser.add_argument('--save_Dis', type=bool, default=True, help='whether save discriminator ckpt with generator')
    parser.add_argument('--start_annealing_epoch', type=int, default=150, help='Learning rate weight decay start point')
    parser.add_argument('--end_annealing_epoch', type=int, default=300, help='Learning rate weight decay end point')
    parser.add_argument('--end_lr', type=int, default=0, help='Final learning rate')
    ckpt_folder, image_folder, source_folder, summary_folder = utils.folder_setting()
    parser.add_argument('--ckpt_dir', type=str, default=ckpt_folder)
    parser.add_argument('--img_dir', type=str, default=image_folder)
    parser.add_argument('--scr_dir', type=str, default=source_folder)
    parser.add_argument('--use_vgg', type=bool, default=True, help='where to use vgg perceptual loss in generator')
    parser.add_argument('--n_summary', type=int, default=100, help='save interval btw save summary points on step')
    parser.add_argument('--sum_dir', type=str, default=summary_folder, help='tensorboard log write iteration')
    parser.add_argument('--seg_channel', type=int, default=3, help='# channel of segmap / celebhq : 19, city : 30')

    return parser.parse_args()

def main():
    args = parse_args()
    if not args :
        exit()
    print(args.ckpt_dir.split('/')[5])
    gaugan = GAUGAN(args)
    
    print('build model')
    gaugan.build_model()
    
    print('load datasets')
    gaugan.load_dataset()

    print('train start')
    gaugan.train()

if __name__=='__main__':
    main()
