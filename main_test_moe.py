import argparse
import json
import logging
import math
import os.path
import random

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import models.basicblock as B
from data.select_dataset import define_Dataset
from models.network_dpsr import MSRResNet_prior as dpsr
from models.select_model import define_Model
from utils_n import utils_image as util
from utils_n import utils_logger
from utils_n import utils_option as option
from utils_n.utils_dist import get_dist_info, init_dist

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
            raise NotImplementedError(
                'upsample mode [{:s}] is not found'.format(upsample_mode))

        if upscale == 3:
            m_uper = upsample_block(nc, nc, mode='3'+act_mode)
        else:
            m_uper = [upsample_block(nc, nc, mode='2'+act_mode)
                      for _ in range(n_upscale)]

        H_conv0 = B.conv(nc, nc, mode='C'+act_mode)
        H_conv1 = B.conv(nc, out_nc, mode='C')
        m_tail = B.sequential(H_conv0, H_conv1)

        self.model = B.sequential(m_head, B.ShortcutBlock(
            B.sequential(*m_body)), *m_uper, m_tail)

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
            raise NotImplementedError(
                'upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            m_uper = upsample_block(nc, nc, mode='3'+act_mode)
        else:
            m_uper = [upsample_block(nc, nc, mode='2'+act_mode)
                      for _ in range(n_upscale)]

        H_conv0 = B.conv(nc, nc, mode='C'+act_mode)
        H_conv1 = B.conv(nc, out_nc, bias=False, mode='C')
        m_tail = B.sequential(H_conv0, H_conv1)

        self.model = B.sequential(m_head, B.ShortcutBlock(
            B.sequential(*m_body)), *m_uper, m_tail)

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
            raise NotImplementedError(
                'upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            m_uper = upsample_block(nc, nc, mode='3'+act_mode)
        else:
            m_uper = [upsample_block(nc, nc, mode='2'+act_mode)
                      for _ in range(n_upscale)]

        H_conv0 = B.conv(nc, nc, mode='C'+act_mode)
        H_conv1 = B.conv(nc, out_nc, bias=False, mode='C')
        m_tail = B.sequential(H_conv0, H_conv1)

        self.model = B.sequential(m_head, B.ShortcutBlock(
            B.sequential(*m_body)), *m_uper, m_tail)

    def forward(self, x):
        x = self.model(x)
        return x


def visualize_with_segmentation(images, titles):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from torchvision.transforms import ToTensor
    import numpy as np
    import segmentation_models_pytorch as smp
    import torch

    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 5, height_ratios=[2, 2, 1], width_ratios=[
                  2, 1, 1, 1, 1], hspace=0, wspace=0)

    model = smp.Unet(encoder_name="resnext101_32x48d", encoder_weights="instagram",
                     in_channels=1, classes=4, activation='softmax')
    model.eval()

    ax_img = fig.add_subplot(gs[0:2, 0])
    ax_img.imshow(images[0], cmap='gray')
    ax_img.axis('off')
    ax_img.set_title(titles[0], fontsize=15, weight='bold')

    for i in range(1, len(images)):
        ax_crop = fig.add_subplot(gs[0, i])
        ax_crop.imshow(images[i], cmap='gray')
        ax_crop.axis('off')

        tensor_img = ToTensor()(images[i]).unsqueeze(0)
        with torch.no_grad():
            mask = model(tensor_img)
        mask = mask.squeeze(0).permute(1, 2, 0).cpu().numpy()

        ax_seg = fig.add_subplot(gs[1, i])
        ax_seg.imshow(np.argmax(mask, axis=2), cmap='viridis')
        ax_seg.axis('off')

        ax_title = fig.add_subplot(gs[2, i])
        ax_title.text(0.6, 0.6, titles[i], fontsize=12,
                      weight='bold', va='center', ha='center')
        ax_title.axis('off')

    plt.tight_layout(pad=0)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


# def visualize_with_segmentation(images, titles):
#     import matplotlib.pyplot as plt
#     from matplotlib.gridspec import GridSpec
#     from torchvision.transforms import ToTensor
#     import numpy as np
#     import segmentation_models_pytorch as smp
#     import torch

#     fig = plt.figure(figsize=(20, 10))
#     gs = GridSpec(3, 5, height_ratios=[1, 1, 1], width_ratios=[
#                   3, 1, 1, 1, 1], hspace=0, wspace=0)

#     model = smp.Unet(encoder_name="resnext101_32x48d",
#                      encoder_weights="instagram", in_channels=1, classes=64, activation=None)
#     model.eval()

#     ax_img = fig.add_subplot(gs[:, 0])
#     ax_img.imshow(images[0], cmap='gray')
#     ax_img.axis('off')
#     ax_img.set_title(titles[0], fontsize=15, weight='bold')

#     for i in range(1, len(images)):

#         ax_crop = fig.add_subplot(gs[0, i])
#         ax_crop.imshow(images[i], cmap='gray')
#         ax_crop.axis('off')
#         ax_crop.set_title(titles[i], fontsize=12, weight='bold')

#         tensor_img = ToTensor()(images[i]).unsqueeze(0)
#         with torch.no_grad():
#             mask = model(tensor_img)
#         mask = mask.squeeze(0).permute(1, 2, 0).cpu().numpy()

#         ax_seg = fig.add_subplot(gs[1:, i])
#         ax_seg.imshow(np.argmax(mask, axis=2), cmap='viridis')
#         ax_seg.axis('off')

#     plt.tight_layout()
#     plt.subplots_adjust(wspace=0, hspace=0)
#     plt.show()


def visualize_data(images, titles):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import numpy as np
    matplotlib.use('TkAgg')
    from numpy.fft import fft2, fftshift
    from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

    num_images = len(images)
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    gs = GridSpec(5, num_images + 1, figure=fig,
                  height_ratios=[3, 0.5, 0.5, 1, 2])

    # Scientifically appropriate axis colors
    axes_colors = ['darkslategray', 'olive',
                   'steelblue', 'darkred', 'slategray']
    reference_title = "Ground Truth"
    reference_index = titles.index(
        reference_title) if reference_title in titles else -1
    reference_image = images[reference_index].squeeze(
    ) if reference_index != -1 else None

    for i, (img, title) in enumerate(zip(images, titles)):
        ax_img = fig.add_subplot(gs[0, i])
        ax_img.imshow(img, cmap='gray')
        ax_img.axis('on')
        for spine in ax_img.spines.values():  # Apply color to each spine
            spine.set_color(axes_colors[i % len(axes_colors)])

        if title in ['N-SMoE', 'DPSR'] and reference_image is not None:
            current_psnr = psnr(reference_image, img,
                                data_range=img.max() - img.min())
            current_ssim = ssim(reference_image, img,
                                data_range=img.max() - img.min())
            title += f"\nPSNR: {current_psnr:.2f} dB, SSIM: {current_ssim:.4f}"

        ax_img.set_title(title, fontsize=12,
                         family='Times New Roman', fontweight='bold')

        freq = fftshift(fft2(img))
        freq_magnitude = np.log(np.abs(freq) + 1)

        ax_x_spectrum = fig.add_subplot(gs[1, i])
        ax_x_spectrum.plot(np.sum(freq_magnitude, axis=0), color='blue')
        ax_x_spectrum.set_title("X-Spectrum", fontsize=12,
                                family='Times New Roman', fontweight='bold')
        ax_x_spectrum.set_xlabel('Frequency (pixels)',
                                 fontsize=11, family='Times New Roman')
        ax_x_spectrum.set_yticklabels([])
        ax_x_spectrum.tick_params(axis='both', which='major', labelsize=10)
        ax_x_spectrum.grid(True)

        ax_y_spectrum = fig.add_subplot(gs[2, i])
        ax_y_spectrum.plot(np.sum(freq_magnitude, axis=1), color='blue')
        ax_y_spectrum.set_title("Y-Spectrum", fontsize=12,
                                family='Times New Roman', fontweight='bold')
        ax_y_spectrum.set_xlabel('Frequency (pixels)',
                                 fontsize=11, family='Times New Roman')
        ax_y_spectrum.set_yticklabels([])
        ax_y_spectrum.tick_params(axis='both', which='major', labelsize=10)
        ax_y_spectrum.grid(True)

        ax_2d_spectrum = fig.add_subplot(gs[3, i])
        ax_2d_spectrum.imshow(freq_magnitude, cmap='gray')
        ax_2d_spectrum.set_title(
            "2D Spectrum", fontsize=12, family='Times New Roman', fontweight='bold')
        ax_2d_spectrum.axis('on')

    nsmoe_idx = titles.index('N-SMoE') if 'N-SMoE' in titles else -1
    dpsr_idx = titles.index('DPSR') if 'DPSR' in titles else -1

    if nsmoe_idx != -1 and dpsr_idx != -1:
        rec_image = images[nsmoe_idx].squeeze()
        dpsr_image = images[dpsr_idx].squeeze()
        error_map = np.abs(rec_image - dpsr_image)

        # Placing error map in the new column right after DPSR
        ax_error_map = fig.add_subplot(gs[0, dpsr_idx + 1])
        # Changed to 'viridis' colormap
        ax_error_map.imshow(error_map, cmap='viridis')
        ax_error_map.set_title("Error Map (N-SMoE - DPSR)",
                               fontsize=12, family='Times New Roman', fontweight='bold')
        ax_error_map.axis('off')

    plt.show()


#  Unet-SMoE: /mnt/e/Weights/superresolution/f_u_moe_muller__sr_gan_x2_v9_mri_rgb_gan_discriminator_unet/options/train_f_u_moe_gan_x2_240528_030700.json
#  Transformer-SMoE: /mnt/e/Weights/superresolution/transformer_moe_sr_gan_gan_x2_v2_mri_rgb_gan_discriminator_unet/options/train_transformer_x2_gan_240528_031433.json


def main(json_path='/home/ozkan/works/n-smoe/options/train_unet_moex1_psnr_local.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path,
                        help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    # init_iter_G, init_path_G = option.find_last_checkpoint(
    #     opt['path']['models'], net_type='G')

    init_path_G = "/mnt/e/Weights/superresolution/unet_unet_moex1_sr_plain_v5_x2_mri_rgb_act_gelu/models/25000_G.pth"
    init_iter_G = 25000

    opt['path']['pretrained_netG'] = init_path_G

    current_step = init_iter_G

    border = opt['scale']

    opt = option.dict_to_nonedict(opt)

    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(
            opt['path']['log'], logger_name+'.log'))
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
                                     drop_last=False, pin_memory=True, collate_fn=util.custom_collate)

    model = define_Model(opt)
    model.load()
    if opt['rank'] == 0:
        logger.info(model.info_network())
        logger.info(model.info_params())

    directory = "/home/ozkan/works/diff-smoe/zoo/"
    esrgan = os.path.join(directory, "ESRGAN.pth")

    dpsr_state_path = "/home/ozkan/works/n-smoe/superresolution/dpsr/models/10000_G.pth"

    # model_esrgan = torch.load(esrgan, map_location=torch.device(model.device))
    # model_dpsr = torch.load(dpsr, map_location=torch.device(model.device))

    # model_dpsr = MSRResNet_prior(in_nc=1, out_nc=1, nc=96, nb=16,
    #                              upscale=opt['scale'], act_mode='R', upsample_mode='pixelshuffle')

    json_dpsr = '''
        {
        "netG": {
            "net_type": "dpsr",
            "in_nc": 1,
            "out_nc": 1,
            "nc": 96,
            "nb": 16,
            "gc": 32,
            "ng": 2,
            "reduction": 16,
            "act_mode": "R",
            "upsample_mode": "pixelshuffle",
            "downsample_mode": "strideconv",
            "init_type": "orthogonal",
            "init_bn_type": "uniform",
            "init_gain": 0.2,
            "scale": 2,
            "n_channels": 1,
            "ang_res": 5,
            "phw": 16,
            "overlap": 10
            }
         }
        '''

    netG_dpsr = json.loads(json_dpsr)["netG"]

    model_dpsr = dpsr(
        in_nc=netG_dpsr["in_nc"],
        out_nc=netG_dpsr["out_nc"],
        nc=netG_dpsr["nc"],
        nb=netG_dpsr["nb"],
        upscale=netG_dpsr["scale"],
        act_mode=netG_dpsr["act_mode"],
        upsample_mode=netG_dpsr["upsample_mode"],
    )

    model_dpsr.load_state_dict(torch.load(
        dpsr_state_path, weights_only=True), strict=True)
    model_dpsr.eval()
    for k, v in model_dpsr.named_parameters():
        v.requires_grad = False
    model_dpsr = model_dpsr.to(model.device)

    model_esrgan = RRDB(in_nc=3, out_nc=3, nc=64, nb=23, gc=32,
                        upscale=opt['scale'], act_mode='L', upsample_mode='upconv')
    model_esrgan.load_state_dict(torch.load(
        esrgan, weights_only=True), strict=False)  # strict=False
    model_esrgan.eval()
    for k, v in model_esrgan.named_parameters():
        v.requires_grad = False
    model_esrgan = model_esrgan.to(model.device)

    avg_psnr = 0.0
    idx = 0

    # titles = ['Noisy Low Resolution', 'Ground Truth',
    #           'N-SMoE', "DPSR"]

    titles = ["High Resolution", "Low Resolution Crop", "High Resolution Crop",
              'N-SMoE', "DPSR"]

    with torch.no_grad():
        for test_data in test_loader:
            if test_data is None:
                continue

            idx += 1
            image_name_ext = os.path.basename(test_data['L_path'][0])
            img_name, ext = os.path.splitext(image_name_ext)

            img_dir = os.path.join(opt['path']['images'], img_name)
            util.mkdir(img_dir)

            # L_img_3ch = util.make_3ch(test_data['L']).cuda()

            # noise_level_map = torch.ones((1, 1, L_img_3ch.size(2), L_img_3ch.size(
            #     3)), dtype=torch.float, device=L_img_3ch.device).mul_(0./255.)
            # img_L = torch.cat((L_img_3ch, noise_level_map), dim=1)

            E_img_dpsr = model_dpsr(test_data['L'].to(model.device))
            E_img_dpsr = util._tensor2uint(E_img_dpsr)
            # E_img_dpsr = util.bgr2ycbcr(E_img_dpsr, only_y=True)

            # E_img_esrgan = model_esrgan(L_img_3ch)
            # E_img_esrgan = util._tensor2uint(E_img_esrgan)
            # E_img_esrgan = util.bgr2ycbcr(E_img_esrgan, only_y=True)

            model.feed_data(test_data)
            model.test()

            visuals = model.current_visuals()
            L_crop_img = util.tensor2uint(visuals['L'])
            E_crop_img = util.tensor2uint(visuals['E'])
            H_crop_img = util.tensor2uint(visuals['H'])

            img_H = util.imread_uint(test_data['H_path'][0], n_channels=1)
            img_H = util.modcrop(img_H, border)

            # visualize_data(
            #     [L_crop_img, H_crop_img, E_crop_img, E_img_dpsr], titles)

            visualize_with_segmentation(
                [img_H, L_crop_img, H_crop_img, E_crop_img, E_img_dpsr], titles)

            save_img_path = os.path.join(
                img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
            util.imsave(E_crop_img, save_img_path)

            current_psnr = util.calculate_psnr(
                E_crop_img, H_crop_img, border=border)

            logger.info(
                '{:->4d}--> {:>10s} | {:<4.2f}dB'.format(idx, image_name_ext, current_psnr))

            avg_psnr += current_psnr

        avg_psnr = avg_psnr / idx

        logger.info('<Average PSNR : {:<.2f}dB\n'.format(avg_psnr))


if __name__ == '__main__':
    main()
