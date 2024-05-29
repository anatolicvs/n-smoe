import argparse
import math
import os.path

import random
import numpy as np

import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

from utils_n import utils_logger
from utils_n import utils_image as util
from utils_n import utils_option as option
from utils_n.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model
import matplotlib.pyplot as plt
from scipy.fft import fftshift, fft2

def visualize_data(L, H, E):
    fig = plt.figure(figsize=(18, 12))  # Overall figure size
    gs = plt.GridSpec(2, 3)  # 2 rows, 3 columns grid

    titles = ['Noisy Low Resolution', 'Ground Truth', 'Reconstruction (UNet-SMoE)']
    images = [L, H, E]

    for i, (img, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img, cmap='gray') 
        ax.set_title(title, fontsize=12, family='Times New Roman', fontweight='bold')
        ax.axis('on')  # Turn off axis to focus on the image

        # Compute frequency spectrum
        freq = fftshift(fft2(img))
        freq_magnitude = np.log(np.abs(freq) + 1)
        signal = np.sum(freq_magnitude, axis=0)  # Projection of the frequency magnitude along x-axis

        ax_freq = fig.add_subplot(gs[1, i])
        ax_freq.plot(signal, color='blue')  
        ax_freq.set_title(f'{title} Spectrum', fontsize=12, family='Times New Roman', fontweight='bold')
        ax_freq.set_xlim([0, len(signal)])  
        ax_freq.set_xlabel('Frequency (pixels)', fontsize=11, family='Times New Roman')
        ax_freq.set_ylabel('Magnitude (log scale)', fontsize=11, family='Times New Roman')
        ax_freq.tick_params(axis='both', which='major', labelsize=10)
        ax_freq.grid(True)  

    plt.tight_layout()
    plt.show()

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

    avg_psnr = 0.0
    idx = 0

    for test_data in test_loader:
        if test_data is None:
            continue
        
        idx += 1
        image_name_ext = os.path.basename(test_data['L_path'][0])
        img_name, ext = os.path.splitext(image_name_ext)

        img_dir = os.path.join(opt['path']['images'], img_name)
        util.mkdir(img_dir)

        model.feed_data(test_data)
        model.test()

        visuals = model.current_visuals()
        L_img = util.tensor2uint(visuals['L'])
        E_img = util.tensor2uint(visuals['E'])
        H_img = util.tensor2uint(visuals['H'])
        
        visualize_data(L_img, H_img, E_img)

        save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
        util.imsave(E_img, save_img_path)

        
        current_psnr = util.calculate_psnr(E_img, H_img, border=border)

        logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB'.format(idx, image_name_ext, current_psnr))

        avg_psnr += current_psnr

    avg_psnr = avg_psnr / idx

   
    logger.info('<Average PSNR : {:<.2f}dB\n'.format(avg_psnr))


if __name__ == '__main__':
    main()