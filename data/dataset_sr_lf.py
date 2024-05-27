import random
import numpy as np
import torch.utils.data as data
import utils_n.utils_image as util
import torch.nn.functional as F
import h5py

class DatasetSRLF(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for SISR.
    # If only "paths_H" is provided, sythesize bicubicly downsampled L on-the-fly.
    # -----------------------------------------
    # e.g., SRResNet
    # -----------------------------------------
    '''

    def __init__(self, opt):
        super(DatasetSRLF, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.sf = opt['scale'] if opt['scale'] else 4
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 96
        self.L_size = self.patch_size // self.sf
        self.phw = opt['phw']
        self.stride = opt['stride']
        # ------------------------------------
        # get paths of L/H
        # ------------------------------------
        self.paths_H = util.get_lf_image_paths(opt['dataroot_H'], opt['ang_res'], opt['scale'], opt['data_name'])
        
        # assert self.paths_H, 'Error: H path is empty.'
        # if self.paths_L and self.paths_H:
        #     assert len(self.paths_L) == len(self.paths_H), 'L/H mismatch - {}, {}.'.format(len(self.paths_L), len(self.paths_H))

    def __getitem__(self, index):

        L_path = None
        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        with h5py.File(H_path, 'r') as hf:
            img_H_SAI_y = np.array(hf.get('Hr_SAI_y'))  
            img_H_SAI_y = np.expand_dims(img_H_SAI_y, axis=-1)      
           
        # ------------------------------------
        # modcrop
        # ------------------------------------
        img_H = util.modcrop(img_H_SAI_y, self.sf)

        # ------------------------------------
        # get L image
        # ------------------------------------
        
        H, W = img_H.shape[:2]
        img_L = util.imresize_np(img_H, 1 / self.sf, True)

        H, W, C = img_L.shape
        # --------------------------------
        # randomly crop the L patch
        # --------------------------------
        rnd_h = random.randint(0, max(0, H - self.L_size))
        rnd_w = random.randint(0, max(0, W - self.L_size))
        img_L = img_L[rnd_h:rnd_h + self.L_size, rnd_w:rnd_w + self.L_size, :]

        # --------------------------------
        # crop corresponding H patch
        # --------------------------------
        rnd_h_H, rnd_w_H = int(rnd_h * self.sf), int(rnd_w * self.sf)
        img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.opt['phase'] == 'train':
            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            mode = random.randint(0, 7)
            img_L, img_H = util.augment_img(img_L, mode=mode), util.augment_img(img_H, mode=mode)
     
        # ------------------------------------
        # L/H pairs, HWC to CHW, numpy to tensor
        # ------------------------------------
        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)
    
        if L_path is None:
            L_path = H_path

        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)
