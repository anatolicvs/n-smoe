import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ang_res", type=int, default=5, help="angular resolution")
parser.add_argument("--scale_factor", type=int, default=4, help="4, 2")
parser.add_argument('--model_name', type=str, default='network_lft', help="model name")
parser.add_argument("--channels", type=int, default=64, help="channels")
parser.add_argument("--use_pre_pth", type=bool, default=False, help="use pre model ckpt")


# parser.add_argument('--opt', type=str, default="/home/ozkan/works/n-smoe/superresolution/lft_gan_discriminator_unet_muller_resizer_v4_angx5_scalex4/options/train_lft_gan_240620_190052.json", help='Path to option JSON file.')

parser.add_argument('--opt', type=str, default="/mnt/d/LF/zoo/lft_gan_discriminator_unet_muller_resizer_v4_angx5_scalex4/train_lft_gan_240430_194704.json", help='Path to option JSON file.')


# parser.add_argument("--path_pre_pth", type=str, default='./pth/LFT_5x5_4x_epoch_50_model.pth',
#                     help="path for pre model ckpt")


# parser.add_argument("--path_pre_pth", type=str, default='/home/ozkan/works/n-smoe/superresolution/lft_gan_discriminator_unet_muller_resizer_v4_angx5_scalex4/models/5000_G.pth',
#                     help="path for pre model ckpt")

parser.add_argument("--path_pre_pth", type=str, default='/mnt/d/LF/zoo/lft_gan_discriminator_unet_muller_resizer_v4_angx5_scalex4/60000_G.pth',
                    help="path for pre model ckpt")

parser.add_argument('--data_name', type=str, default='HCI_old',
                    help='EPFL, HCI_new, HCI_old, INRIA_Lytro, Stanford_Gantry, ALL')
parser.add_argument('--path_for_train', type=str, default='./data_for_train/')
parser.add_argument('--path_for_test', type=str, default='/mnt/d/LF/data_for_test/')
parser.add_argument('--path_log', type=str, default='./log/')
parser.add_argument('--patch_size_for_test', default=32, type=int, help='patch size')
parser.add_argument('--stride_for_test', default=16, type=int, help='stride')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
parser.add_argument('--decay_rate', type=float, default=0, help='weight decay [default: 1e-4]')
parser.add_argument('--n_steps', type=int, default=15, help='number of epochs to update learning rate')
parser.add_argument('--gamma', type=float, default=0.5, help='gamma')
parser.add_argument('--epoch', default=50, type=int, help='Epoch to run [default: 50]')
parser.add_argument('--num_workers', type=int, default=8, help='num workers of the Data Loader')
parser.add_argument('--local_rank', dest='local_rank', type=int, default=0, )

args = parser.parse_args()