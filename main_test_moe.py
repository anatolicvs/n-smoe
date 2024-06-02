import math
import argparse
import torch.nn as nn
import os.path

import random
import numpy as np

import logging
from torch.utils.data import DataLoader
import torch

from utils_n import utils_logger
from utils_n import utils_image as util
from utils_n import utils_option as option
from utils_n.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model
import matplotlib.pyplot as plt
from scipy.fft import fftshift, fft2
from matplotlib.gridspec import GridSpec



from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import models.basicblock as B


"""
# --------------------------------------------
# SR network withsr/BasicSR/basicsr Residual in Residual Dense Block (RRDB)
# "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks"
# --------------------------------------------
"""
class RRDB(nn.Module):
    """
    gc: number of growth channels
    nb: number of RRDB
    """
    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=23, gc=32, upscale=4, act_mode='L', upsample_mode='upconv'):
        super(RRDB, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'

        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        m_head = B.conv(in_nc, nc, mode='C')

        m_body = [B.RRDB(nc, gc=32, mode='C'+act_mode) for _ in range(nb)]
        m_body.append(B.conv(nc, nc, mode='C'))

        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        if upscale == 3:
            m_uper = upsample_block(nc, nc, mode='3'+act_mode)
        else:
            m_uper = [upsample_block(nc, nc, mode='2'+act_mode) for _ in range(n_upscale)]

        H_conv0 = B.conv(nc, nc, mode='C'+act_mode)
        H_conv1 = B.conv(nc, out_nc, mode='C')
        m_tail = B.sequential(H_conv0, H_conv1)

        self.model = B.sequential(m_head, B.ShortcutBlock(B.sequential(*m_body)), *m_uper, m_tail)

    def forward(self, x):
        x = self.model(x)
        return x



""""
# --------------------------------------------
# modified SRResNet
#   -- MSRResNet_prior (for DPSR)
# --------------------------------------------
References:
@inproceedings{zhang2019deep,
  title={Deep Plug-and-Play Super-Resolution for Arbitrary Blur Kernels},
  author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  pages={1671--1681},
  year={2019}
}
@inproceedings{wang2018esrgan,
  title={Esrgan: Enhanced super-resolution generative adversarial networks},
  author={Wang, Xintao and Yu, Ke and Wu, Shixiang and Gu, Jinjin and Liu, Yihao and Dong, Chao and Qiao, Yu and Change Loy, Chen},
  booktitle={European Conference on Computer Vision (ECCV)},
  pages={0--0},
  year={2018}
}
@inproceedings{ledig2017photo,
  title={Photo-realistic single image super-resolution using a generative adversarial network},
  author={Ledig, Christian and Theis, Lucas and Husz{\'a}r, Ferenc and Caballero, Jose and Cunningham, Andrew and Acosta, Alejandro and Aitken, Andrew and Tejani, Alykhan and Totz, Johannes and Wang, Zehan and others},
  booktitle={IEEE conference on computer vision and pattern recognition},
  pages={4681--4690},
  year={2017}
}
# --------------------------------------------
"""

# --------------------------------------------
# MSRResNet super-resolver prior for DPSR
# https://github.com/cszn/DPSR
# https://github.com/cszn/DPSR/blob/master/models/network_srresnet.py
# --------------------------------------------
class MSRResNet_prior(nn.Module):
    def __init__(self, in_nc=4, out_nc=3, nc=96, nb=16, upscale=4, act_mode='R', upsample_mode='upconv'):
        super(MSRResNet_prior, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        m_head = B.conv(in_nc, nc, mode='C')

        m_body = [B.ResBlock(nc, nc, mode='C'+act_mode+'C') for _ in range(nb)]
        m_body.append(B.conv(nc, nc, mode='C'))

        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            m_uper = upsample_block(nc, nc, mode='3'+act_mode)
        else:
            m_uper = [upsample_block(nc, nc, mode='2'+act_mode) for _ in range(n_upscale)]

        H_conv0 = B.conv(nc, nc, mode='C'+act_mode)
        H_conv1 = B.conv(nc, out_nc, bias=False, mode='C')
        m_tail = B.sequential(H_conv0, H_conv1)

        self.model = B.sequential(m_head, B.ShortcutBlock(B.sequential(*m_body)), *m_uper, m_tail)

    def forward(self, x):
        x = self.model(x)
        return x

class SRResNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=16, upscale=4, act_mode='R', upsample_mode='upconv'):
        super(SRResNet, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        m_head = B.conv(in_nc, nc, mode='C')

        m_body = [B.ResBlock(nc, nc, mode='C'+act_mode+'C') for _ in range(nb)]
        m_body.append(B.conv(nc, nc, mode='C'))

        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            m_uper = upsample_block(nc, nc, mode='3'+act_mode)
        else:
            m_uper = [upsample_block(nc, nc, mode='2'+act_mode) for _ in range(n_upscale)]

        H_conv0 = B.conv(nc, nc, mode='C'+act_mode)
        H_conv1 = B.conv(nc, out_nc, bias=False, mode='C')
        m_tail = B.sequential(H_conv0, H_conv1)

        self.model = B.sequential(m_head, B.ShortcutBlock(B.sequential(*m_body)), *m_uper, m_tail)

    def forward(self, x):
        x = self.model(x)
        return x


def visualize_data(images, titles):
    num_images = len(images)
    fig = plt.figure(figsize=(18, 8), constrained_layout=True)
    gs = GridSpec(5, num_images * 2, figure=fig, height_ratios=[4, 1, 1, 1, 1])

    reference_title = "Ground Truth"
    noisy_title = "Noisy Low Resolution"
    reference_index = titles.index(reference_title) if reference_title in titles else -1
    noisy_index = titles.index(noisy_title) if noisy_title in titles else -1
    reference_image = images[reference_index].squeeze() if reference_index != -1 else None

    for i, (img, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(gs[0, i*2:(i+1)*2])
        ax.imshow(img, cmap='gray')
        ax.axis('on')

        if i != noisy_index and i != reference_index and reference_image is not None:
            current_psnr = psnr(reference_image, img, data_range=img.max() - img.min())
            current_ssim = ssim(reference_image, img, data_range=img.max() - img.min())
            title += f"\nPSNR: {current_psnr:.2f} dB, SSIM: {current_ssim:.4f}"
        
        ax.set_title(title, fontsize=12, family='Times New Roman', fontweight='bold')

        freq = fftshift(fft2(img))
        freq_magnitude = np.log(np.abs(freq) + 1)
        ax_freq = fig.add_subplot(gs[1:5, i*2:(i+1)*2])
        ax_freq.plot(np.sum(freq_magnitude, axis=0), color='blue')
        ax_freq.set_title(f"{title} Spectrum", fontsize=12, family='Times New Roman', fontweight='bold')
        ax_freq.set_xlabel('Frequency (pixels)', fontsize=11, family='Times New Roman')
        ax_freq.set_yticklabels([])
        ax_freq.tick_params(axis='both', which='major', labelsize=10)
        ax_freq.grid(True)

    plt.show()

#  Unet-SMoE: /mnt/e/Weights/superresolution/f_u_moe_muller__sr_gan_x2_v9_mri_rgb_gan_discriminator_unet/options/train_f_u_moe_gan_x2_240528_030700.json
#  Transformer-SMoE: /mnt/e/Weights/superresolution/transformer_moe_sr_gan_gan_x2_v2_mri_rgb_gan_discriminator_unet/options/train_transformer_x2_gan_240528_031433.json

def main(json_path='/mnt/e/Weights/superresolution/f_u_moe_muller__sr_gan_x2_v9_mri_rgb_gan_discriminator_unet/options/train_f_u_moe_gan_x2_240528_030700.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()


    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    opt['path']['pretrained_netG'] = init_path_G
    
    current_step = init_iter_G

    border = opt['scale']

    opt = option.dict_to_nonedict(opt)

    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True,collate_fn=util.custom_collate)

    model = define_Model(opt)
    model.load()
    if opt['rank'] == 0:
        logger.info(model.info_network())
        logger.info(model.info_params())

    
    directory = "/home/ozkan/works/diff-smoe/zoo/"
    esrgan = os.path.join(directory, "ESRGAN.pth") 
    dpsr = os.path.join(directory, "dpsr_x2.pth")
    # model_esrgan = torch.load(esrgan, map_location=torch.device(model.device))
    # model_dpsr = torch.load(dpsr, map_location=torch.device(model.device))

    model_dpsr = MSRResNet_prior(in_nc=3+1, out_nc=3, nc=96, nb=16, upscale=opt['scale'], act_mode='R', upsample_mode='pixelshuffle')
    model_dpsr.load_state_dict(torch.load(dpsr), strict=False)
    model_dpsr.eval()
    for k, v in model_dpsr.named_parameters():
        v.requires_grad = False
    model_dpsr = model_dpsr.to(model.device)

    model_esrgan = RRDB(in_nc=3, out_nc=3, nc=64, nb=23, gc=32, upscale=opt['scale'], act_mode='L', upsample_mode='upconv')
    model_esrgan.load_state_dict(torch.load(esrgan), strict=False)  # strict=False
    model_esrgan.eval()
    for k, v in model_esrgan.named_parameters():
        v.requires_grad = False
    model_esrgan = model_esrgan.to(model.device)

    avg_psnr = 0.0
    idx = 0

    titles = ['Noisy Low Resolution', 'Ground Truth', 'Reconstruction (UNet-SMoE)', "DPSR"]
    
    with torch.no_grad():
        for test_data in test_loader:
            if test_data is None:
                continue
            
            idx += 1
            image_name_ext = os.path.basename(test_data['L_path'][0])
            img_name, ext = os.path.splitext(image_name_ext)

            img_dir = os.path.join(opt['path']['images'], img_name)
            util.mkdir(img_dir)


            L_img_3ch = util.make_3ch(test_data['L']).cuda()

            noise_level_map = torch.ones((1, 1, L_img_3ch.size(2), L_img_3ch.size(3)), dtype=torch.float, device=L_img_3ch.device).mul_(0./255.)
            img_L = torch.cat((L_img_3ch, noise_level_map), dim=1)
            E_img_dpsr = model_dpsr(img_L)
            E_img_dpsr = util._tensor2uint(E_img_dpsr)
            E_img_dpsr = util.bgr2ycbcr(E_img_dpsr, only_y=True)

            # E_img_esrgan = model_esrgan(L_img_3ch)
            # E_img_esrgan = util._tensor2uint(E_img_esrgan)
            # E_img_esrgan = util.bgr2ycbcr(E_img_esrgan, only_y=True)
            
            model.feed_data(test_data)
            model.test()

            visuals = model.current_visuals()
            L_img = util.tensor2uint(visuals['L'])
            E_img = util.tensor2uint(visuals['E'])
            H_img = util.tensor2uint(visuals['H'])
            
            
            visualize_data([L_img, H_img, E_img, E_img_dpsr], titles)

            save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
            util.imsave(E_img, save_img_path)

            
            current_psnr = util.calculate_psnr(E_img, H_img, border=border)

            logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB'.format(idx, image_name_ext, current_psnr))

            avg_psnr += current_psnr

        avg_psnr = avg_psnr / idx

    
        logger.info('<Average PSNR : {:<.2f}dB\n'.format(avg_psnr))


if __name__ == '__main__':
    main()