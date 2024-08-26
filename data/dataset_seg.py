import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from sam2.utils.transforms import SAM2Transforms
import random


join = os.path.join


class NpyDataset(Dataset):
    def __init__(self, opt):

        data_root: str = opt.get("dataroot_H", None)

        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.gt_path_files = sorted(
            glob.glob(join(self.gt_path, "*.npy"), recursive=True)
        )
        self.gt_path_files = self.gt_path_files
        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if os.path.isfile(join(self.img_path, os.path.basename(file)))
        ]
        self.bbox_shift: int = opt.get("bbox_shift", 20)
        self._transform = SAM2Transforms(resolution=1024, mask_threshold=0)
        print(f"number of images: {len(self.gt_path_files)}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        # load npy image (1024, 1024, 3), [0,1]
        img_name = os.path.basename(self.gt_path_files[index])
        img = np.load(
            join(self.img_path, img_name), "r", allow_pickle=True
        )  # (1024, 1024, 3)
        # convert the shape to (3, H, W)
        img_1024 = self._transform(img.copy())
        gt = np.load(
            self.gt_path_files[index], "r", allow_pickle=True
        )  # multiple labels [0, 1,4,5...], (256,256)
        assert img_name == os.path.basename(self.gt_path_files[index]), (
            "img gt name error" + self.gt_path_files[index] + self.npy_files[index]
        )
        assert gt.shape == (256, 256), "ground truth should be 256x256"
        label_ids = np.unique(gt)[1:]
        gt2D = np.uint8(
            gt == random.choice(label_ids.tolist())
        )  # only one label, (256, 256)
        assert np.max(gt2D) == 1 and np.min(gt2D) == 0.0, "ground truth should be 0, 1"
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))

        bboxes = (
            np.array([x_min, y_min, x_max, y_max]) * 4
        )  ## scale bbox from 256 to 1024

        return {
            "img": img_1024,
            "label": torch.tensor(gt2D[None, :, :]).long(),
            "box": torch.tensor(bboxes).float(),
            "L_path": str(img_name),
            "H_path": str(img_name),
        }
