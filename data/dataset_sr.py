import random
import numpy as np
import torch.utils.data as data
import utils_n.utils_image as util
import torch.nn.functional as F
from utils_n import utils_blindsr as blindsr
from scipy.io import loadmat
import torch


class DatasetSR(data.Dataset):
    """
    # -----------------------------------------
    # Get L/H for SISR.
    # If only "paths_H" is provided, sythesize bicubicly downsampled L on-the-fly.
    # -----------------------------------------
    # e.g., SRResNet
    # -----------------------------------------
    """

    def __init__(self, opt):
        super(DatasetSR, self).__init__()
        self.opt = opt
        self.n_channels = opt["n_channels"] if opt["n_channels"] else 3
        self.sf = opt["scale"] if opt["scale"] else 4
        self.patch_size = self.opt["H_size"] if self.opt["H_size"] else 96
        self.L_size = self.patch_size // self.sf

        self.phw: int = opt.get("phw", 32)
        self.overlap: int = opt.get("overlap", 4)
        self.length: int = opt.get("length", -1)

        self.stride = opt["stride"]
        self.k = loadmat(opt["kernel_path"])
        # ------------------------------------
        # get paths of L/H
        # ------------------------------------
        self.paths_H = util.get_image_paths(opt["dataroot_H"])[: self.length]
        # self.paths_H = random.sample(
        #     util.get_image_paths(opt["dataroot_H"]), self.length
        # )
        self.paths_L = util.get_image_paths(opt["dataroot_L"])

        assert self.paths_H, "Error: H path is empty."
        if self.paths_L and self.paths_H:
            assert len(self.paths_L) == len(
                self.paths_H
            ), "L/H mismatch - {}, {}.".format(len(self.paths_L), len(self.paths_H))

    def __getitem__(self, index):

        L_path = None
        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)
        img_H = util.uint2single(img_H)

        # ------------------------------------
        # modcrop
        # ------------------------------------
        img_H = util.modcrop(img_H, self.sf)

        # ------------------------------------
        # get L image
        # ------------------------------------
        if self.paths_L:
            # --------------------------------
            # directly load L image
            # --------------------------------
            L_path = self.paths_L[index]
            img_L = util.imread_uint(L_path, self.n_channels)
            img_L = util.uint2single(img_L)
        else:
            # --------------------------------
            # sythesize L image via matlab's bicubic
            # --------------------------------
            H, W = img_H.shape[:2]
            img_L = util.imresize_np(img_H, 1 / self.sf, True)
            # img_L = blindsr.dpsr_degradation(
            #     img_H, k=self.k["kernels"][0][1], sf=self.sf
            # )[0]

        H, W, C = img_L.shape
        # --------------------------------
        # randomly crop the L patch
        # --------------------------------
        rnd_h = random.randint(0, max(0, H - self.L_size))
        rnd_w = random.randint(0, max(0, W - self.L_size))
        img_L = img_L[rnd_h : rnd_h + self.L_size, rnd_w : rnd_w + self.L_size, :]

        # --------------------------------
        # crop corresponding H patch
        # --------------------------------
        rnd_h_H, rnd_w_H = int(rnd_h * self.sf), int(rnd_w * self.sf)
        img_H = img_H[
            rnd_h_H : rnd_h_H + self.patch_size, rnd_w_H : rnd_w_H + self.patch_size, :
        ]

        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.opt["phase"] == "train":
            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            mode = random.randint(0, 7)
            img_L, img_H = util.augment_img(img_L, mode=mode), util.augment_img(
                img_H, mode=mode
            )

        # ------------------------------------
        # L/H pairs, HWC to CHW, numpy to tensor
        # ------------------------------------
        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)

        img_L_p = self.extract_blocks(img_L, self.phw, self.overlap)
        # img_L_p = img_L.unfold(1, self.phw, self.stride).unfold(
        #     2, self.phw, self.stride
        # )
        # img_L_p = F.max_pool3d(img_L_p, kernel_size=1, stride=1)
        # img_L_p = img_L_p.view(
        #     img_L_p.shape[1] * img_L_p.shape[2],
        #     img_L_p.shape[0],
        #     img_L_p.shape[3],
        #     img_L_p.shape[4],
        # )

        if L_path is None:
            L_path = H_path

        return {
            "L": img_L,
            "L_p": img_L_p,
            "H": img_H,
            "L_path": L_path,
            "H_path": H_path,
        }

    @staticmethod
    def extract_blocks(img_tensor, block_size, overlap) -> torch.Tensor:
        blocks = []
        step = block_size - overlap
        for i in range(0, img_tensor.shape[1] - block_size + 1, step):
            for j in range(0, img_tensor.shape[2] - block_size + 1, step):
                blocks.append(img_tensor[:, i : i + block_size, j : j + block_size])
        return torch.stack(blocks)

    def __len__(self):
        return len(self.paths_H)
