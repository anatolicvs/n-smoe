# type: ignore
import logging
import os
import pickle
import random
import xml.etree.ElementTree as etree
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Sequence

import cv2
import h5py
import nibabel
import numpy as np
import torch
import torch.cuda.amp as amp
from scipy.io import loadmat
from torch.utils.data import Dataset

from PIL import Image
import utils_n.utils_image as util
from utils_n import utils_blindsr as blindsr


class FastMRIRawDataSample(NamedTuple):
    fname: Path
    slice_ind: int
    metadata: Dict[str, Any]


class MedicalDatasetSR(Dataset):
    def __init__(self, opt):
        self.n_channels: int = opt.get("n_channels", 1)
        self.roots: list[str] = opt.get("dataroot_H", [])
        self.exclude_dirs: list[str] = opt.get("exclude_dirs", [])
        self.challenge = opt.get("challenge", "multicoil")
        self.use_dataset_cache: bool = opt.get("use_dataset_cache", True)
        self.dataset_cache_file: str = opt.get(
            "dataset_cache_file", "dataset_cache.pkl"
        )
        self.sf: int = opt.get("scale", 2)
        self.phase: str = opt.get("phase", "train")
        self.h_size: int = opt.get("H_size", 96)
        self.lq_patchsize: int = opt.get("lq_patchsize", 64)
        self.crop_method: str = opt.get("crop_method", "high_texture")
        self.phw: int = opt.get("phw", 32)
        self.overlap: int = opt.get("overlap", 4)
        self.length: int = opt.get("length", -1)
        self.degradation_methods: List[str] = opt.get("degradation_methods", ["dpsr"])
        self.k = loadmat(opt.get("kernel_path")) if "kernel_path" in opt else None
        self.raw_samples = self.load_samples()
        self.use_imgH = opt.get("use_imgH", False)

    def load_samples(self) -> List[FastMRIRawDataSample]:
        if self.use_dataset_cache and os.path.exists(self.dataset_cache_file):
            try:
                with open(self.dataset_cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logging.error(f"Error loading cached data: {e}")

        files = [self.load_sample(f) for f in util.get_m_image_paths(self.roots)]
        samples = list(chain.from_iterable(filter(None, files)))
        filtered_samples = self.filter_low_content_images(samples, self.exclude_dirs)

        if self.length > 0 and len(filtered_samples) > self.length:
            filtered_samples = random.sample(filtered_samples, self.length)

        if self.use_dataset_cache:
            try:
                with open(self.dataset_cache_file, "wb") as f:
                    pickle.dump(filtered_samples, f)
            except Exception as e:
                logging.error(f"Error writing to cache: {e}")

        return filtered_samples

    def filter_low_content_images(
        self, samples: List[FastMRIRawDataSample], exclude_dirs: List[str] = None
    ) -> List[FastMRIRawDataSample]:
        std_devs = []
        sample_indices = []
        filtered_samples = []

        for s in samples:
            if exclude_dirs and any(
                str(s.fname).startswith(exclude_dir.rstrip("/"))
                for exclude_dir in exclude_dirs
            ):
                filtered_samples.append(s)
                continue

            img_data = self.load_image_data(str(s.fname), s.slice_ind)
            if img_data is None or not isinstance(img_data, np.ndarray):
                continue
            try:
                tensor = torch.from_numpy(img_data).float().to("cuda")

                with torch.amp.autocast("cuda"):
                    std_dev: float = torch.std(tensor).item()

                std_devs.append(std_dev)
                sample_indices.append(s)

            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
        if std_devs:
            std_devs_tensor = torch.tensor(std_devs)
            threshold = torch.quantile(std_devs_tensor, 0.25)
            filtered_samples.extend(
                s
                for s, std_dev in zip(sample_indices, std_devs)
                if std_dev >= threshold
            )
        return filtered_samples

    def load_sample(self, fname) -> None | List[FastMRIRawDataSample]:
        fname_path = Path(fname)

        if not fname_path.exists() or not os.access(fname_path, os.R_OK):
            logging.warning(f"Access issue with file: {fname}")
            return None

        try:
            if str(fname_path).endswith((".jpg", ".jpeg")):
                with Image.open(fname_path) as img:
                    img.verify()

            if str(fname_path).endswith((".h5", ".gz", ".npy")):
                return self.handle_special_formats(fname_path)

            return [FastMRIRawDataSample(fname_path, 0, {})]

        except (OSError, ValueError) as e:
            logging.error(f"Error processing file {fname}: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error processing file {fname}: {e}")
            return None

    def handle_special_formats(self, fname_path) -> List[FastMRIRawDataSample] | None:
        if fname_path.suffix == ".h5":
            metadata, num_slices = self._retrieve_metadata(str(fname_path))
            return [
                FastMRIRawDataSample(fname_path, slice_ind, metadata)
                for slice_ind in range(num_slices)
            ]
        if fname_path.suffix in [".gz", ".npy"]:
            return [FastMRIRawDataSample(fname_path, 0, {})]

    @staticmethod
    def et_query(
        root: etree.Element,
        qlist: Sequence[str],
        namespace: str = "http://www.ismrm.org/ISMRMRD",
    ) -> str:
        s = "."
        prefix = "ismrmrd_namespace"
        ns = {prefix: namespace}
        for el in qlist:
            s += f"//{prefix}:{el}"
        value = root.find(s, ns)
        if value is None:
            raise RuntimeError("Element not found")
        return str(value.text)

    def _retrieve_metadata(self, fname: str):
        if not os.access(fname, os.R_OK):
            logging.error(f"Cannot read file: {fname}")
            return {}, 0

        with h5py.File(fname, "r") as hf:
            et_root = etree.fromstring(hf["ismrmrd_header"][()])

            enc = ["encoding", "encodedSpace", "matrixSize"]
            enc_size = (
                int(self.et_query(et_root, enc + ["x"])),
                int(self.et_query(et_root, enc + ["y"])),
                int(self.et_query(et_root, enc + ["z"])),
            )
            rec = ["encoding", "reconSpace", "matrixSize"]
            recon_size = (
                int(self.et_query(et_root, rec + ["x"])),
                int(self.et_query(et_root, rec + ["y"])),
                int(self.et_query(et_root, rec + ["z"])),
            )

            lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
            enc_limits_center = int(self.et_query(et_root, lims + ["center"]))
            enc_limits_max = int(self.et_query(et_root, lims + ["maximum"])) + 1

            padding_left = enc_size[1] // 2 - enc_limits_center
            padding_right = padding_left + enc_limits_max

            num_slices = hf["kspace"].shape[0]  # type: ignore

            metadata = {
                "padding_left": padding_left,
                "padding_right": padding_right,
                "encoding_size": enc_size,
                "recon_size": recon_size,
                **hf.attrs,
            }

        return metadata, num_slices

    def load_image_data(self, fname: str, slice_ind: int):
        try:
            if isinstance(fname, Path):
                fname = str(fname)

            if not os.path.exists(fname) or not os.access(fname, os.R_OK):
                logging.warning(f"File does not exist or is not accessible: {fname}")
                return None

            img = None
            if fname.endswith(".h5"):
                with h5py.File(fname, "r") as hf:
                    img = np.array(hf["reconstruction_rss"])[slice_ind]
            elif fname.endswith(".gz") and "t1n" in fname:
                volume = nibabel.load(fname).get_fdata()
                best_slice_index = self._get_best_slice(volume)
                img = volume[:, :, best_slice_index]
            elif fname.endswith(".npy"):
                img = np.load(fname) if slice_ind == 0 else np.load(fname)[slice_ind]
            elif fname.endswith(".gz") and "4CH_ES.nii" in fname:
                img = nibabel.load(fname).get_fdata()
            else:
                img = util.imread_uint(fname, self.n_channels)
                img = util.uint2single(img)

            return img if img is not None and img.ndim >= 2 else None

        except PermissionError as e:
            logging.warning(f"Skipping file {fname} due to PermissionError: {e}")
            return None
        except Exception as e:
            logging.warning(f"Skipping file {fname} due to error: {e}")
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
            return None

    def _get_best_slice(self, volume):
        variances = [np.var(volume[:, :, i]) for i in range(volume.shape[2])]
        return np.argmax(variances)

    def center_crop(self, img, crop_size):
        center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
        crop_half = crop_size // 2
        start_x = max(center_x - crop_half, 0)
        start_y = max(center_y - crop_half, 0)
        end_x = min(start_x + crop_size, img.shape[1])
        end_y = min(start_y + crop_size, img.shape[0])
        return img[start_y:end_y, start_x:end_x]

    def crop_high_texture(self, img, crop_size):
        img = np.squeeze(img)
        channels = img.shape[2] if img.ndim == 3 else 1

        grad_magnitude = np.sqrt(
            sum(
                cv2.Sobel(
                    img[:, :, c] if channels > 1 else img, cv2.CV_64F, 1, 0, ksize=3
                )
                ** 2
                + cv2.Sobel(
                    img[:, :, c] if channels > 1 else img, cv2.CV_64F, 0, 1, ksize=3
                )
                ** 2
                for c in range(channels)
            )
        )

        texture_map = cv2.filter2D(
            grad_magnitude, -1, np.ones((crop_size, crop_size), dtype=np.float64)
        )
        center_y, center_x = np.unravel_index(np.argmax(texture_map), texture_map.shape)

        start_y = max(center_y - crop_size // 2, 0)
        start_x = max(center_x - crop_size // 2, 0)
        end_y = min(start_y + crop_size, img.shape[0])
        end_x = min(start_x + crop_size, img.shape[1])

        if end_y - start_y < crop_size:
            start_y = max(end_y - crop_size, 0)
        if end_x - start_x < crop_size:
            start_x = max(end_x - crop_size, 0)

        end_y = start_y + crop_size
        end_x = start_x + crop_size

        return img[start_y:end_y, start_x:end_x]

    def preprocess(self, img):
        if img is None or img.ndim < 2 or np.max(img) == np.min(img):
            return None

        if self.n_channels == 1 and img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = img[:, :, np.newaxis]

        img_H = util.modcrop(img, self.sf)
        img_H = img_H.astype(np.float32)

        if img_H.shape[0] < self.h_size or img_H.shape[1] < self.h_size:
            return None

        normalized_diff = np.ptp(img_H)
        if normalized_diff == 0:
            return None

        img_H = (img_H - np.min(img_H)) / (normalized_diff + np.finfo(np.float32).eps)

        crop_methods = {
            "high_texture": self.crop_high_texture,
            "center": self.center_crop,
        }

        img_crop_H = crop_methods.get(self.crop_method, self.center_crop)(
            img_H, self.h_size
        )

        if self.use_imgH:
            img_H = torch.from_numpy(img_H).float()

        img_crop_H = img_crop_H.astype(np.float32)

        if img_crop_H.ndim == 2:
            img_crop_H = img_crop_H[:, :, np.newaxis]

        return img_crop_H, img_H

    def __getitem__(self, index):
        sample = self.raw_samples[index]
        img = self.load_image_data(sample.fname, sample.slice_ind)

        if img is None:
            return None

        processed_data = self.preprocess(img)

        if processed_data is None:
            return None

        img_crop_H, img_H = processed_data

        return self.apply_degradation(img_crop_H, img_H, sample.fname)

    def apply_degradation(self, img_crop_H, img_oH, fname):
        chosen_model = random.choice(self.degradation_methods)
        kernel = self.select_kernel()
        img_L, img_H = {
            "dpsr": lambda x: blindsr.dpsr_degradation(
                x, kernel, self.sf, self.lq_patchsize
            ),
            "bsrgan_plus": lambda x: blindsr.degradation_bsrgan_plus(
                img=x, sf=self.sf, use_sharp=True, lq_patchsize=self.lq_patchsize
            ),
            "classic_sr": lambda x: blindsr.classical_degradation(
                x, kernel, self.sf, self.lq_patchsize
            ),
            "bicubic_degradation": lambda x: blindsr.bicubic_degradation(
                x, self.sf, self.lq_patchsize
            ),
        }[chosen_model](img_crop_H)

        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)
        img_H = img_H.contiguous()
        img_L = img_L.contiguous()
        img_L_p = self.extract_blocks(img_L, self.phw, self.overlap)

        return {
            "L": img_L,
            "L_p": img_L_p,
            "H": img_H,
            "O": img_oH if self.use_imgH else None,
            "L_path": str(fname),
            "H_path": str(fname),
        }

    def select_kernel(self):
        if self.k and "kernels" in self.k and len(self.k["kernels"][0]) > 0:
            kernel_index = random.randint(0, len(self.k["kernels"][0]) - 1)
            return self.k["kernels"][0][kernel_index]
        else:
            logging.warning(
                "No kernels found or empty kernel list, using default kernel"
            )
            return np.ones((5, 5), dtype=np.float32) / 25

    @staticmethod
    def extract_blocks(img_tensor, block_size, overlap) -> torch.Tensor:
        blocks = []
        step = block_size - overlap
        for i in range(0, img_tensor.shape[1] - block_size + 1, step):
            for j in range(0, img_tensor.shape[2] - block_size + 1, step):
                blocks.append(img_tensor[:, i : i + block_size, j : j + block_size])
        return torch.stack(blocks)

    def __len__(self) -> int:
        return len(self.raw_samples)

    def is_valid_image(self, file_path) -> bool:
        return not file_path.split("/")[-1].startswith(".")
