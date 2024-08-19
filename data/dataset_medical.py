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

import utils_n.utils_image as util
from utils_n import utils_blindsr as blindsr


class FastMRIRawDataSample(NamedTuple):
    fname: Path
    slice_ind: int
    metadata: Dict[str, Any]


class MedicalDatasetSR(Dataset):
    def __init__(self, opt):
        self.n_channels = opt["n_channels"] if "n_channels" in opt else 3
        self.roots = opt["dataroot_H"]
        self.challenge = opt["challenge"] if "challenge" in opt else "multicoil"
        self.use_dataset_cache = (
            opt["use_dataset_cache"] if "use_dataset_cache" in opt else True
        )
        self.dataset_cache_file = (
            opt["dataset_cache_file"]
            if "dataset_cache_file" in opt
            else "dataset_cache.pkl"
        )
        self.sf = opt["scale"] if "scale" in opt else 2
        self.phase = opt["phase"] if "phase" in opt else "train"
        self.h_size = opt["H_size"] if "H_size" in opt else 96
        self.lq_patchsize = opt["lq_patchsize"] if "lq_patchsize" in opt else 64
        self.crop_method = (
            opt["crop_method"] if "crop_method" in opt else "high_texture"
        )
        self.phw = opt["phw"] if "phw" in opt else 32
        self.overlap = opt["overlap"] if "overlap" in opt else 4

        # self.recons_keys = ["reconstruction_rss"]  # "kspace",
        self.k = loadmat(opt["kernel_path"]) if "kernel_path" in opt else None
        self.raw_samples = self.load_samples()

    def load_samples(self) -> List[FastMRIRawDataSample]:
        if self.use_dataset_cache and os.path.exists(self.dataset_cache_file):
            try:
                with open(self.dataset_cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logging.error(f"Error loading cached data: {e}")
                pass

        files = [self.load_sample(f) for f in util.get_m_image_paths(self.roots)]
        samples = list(chain.from_iterable(filter(None, files)))

        filtered_samples = self.filter_low_content_images(samples)

        if self.use_dataset_cache:
            try:
                with open(self.dataset_cache_file, "wb") as f:
                    pickle.dump(filtered_samples, f)
            except Exception as e:
                logging.error(f"Error writing to cache: {e}")

        return filtered_samples

    def filter_low_content_images(
        self, samples: List[FastMRIRawDataSample]
    ) -> List[FastMRIRawDataSample]:
        std_devs = []
        sample_indices = []

        max_memory_usage = torch.cuda.get_device_properties(0).total_memory * 0.9
        scaler = amp.GradScaler(enabled=True)

        for s in samples:
            try:
                img_data = self.load_image_data(str(s.fname), s.slice_ind)
                if img_data is not None:
                    tensor = torch.from_numpy(img_data).float().to("cuda")

                    with amp.autocast():
                        std_dev: int | float | bool = torch.std(tensor).item()

                    std_devs.append(std_dev)
                    sample_indices.append(s)

                torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                else:
                    raise e

        if not std_devs:
            return []

        std_devs_tensor: torch.Tensor = torch.tensor(std_devs)
        threshold: torch.Tensor = torch.quantile(std_devs_tensor, 0.25)

        filtered_samples = [
            s for s, std_dev in zip(sample_indices, std_devs) if std_dev >= threshold
        ]

        return filtered_samples

        sample, img_data = sample_data
        tensor = torch.from_numpy(img_data).float().unsqueeze(0)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(1)

        with amp.autocast():
            std_dev = torch.std(tensor).item()

        std_devs.append(std_dev)
        sample_indices.append(sample)

        torch.cuda.empty_cache()

    def load_sample(self, fname) -> None | List[FastMRIRawDataSample]:
        fname_path = Path(fname)
        if not fname_path.exists() or not os.access(fname, os.R_OK):
            logging.warning(f"Access issue with file: {fname}")
            return None

        try:
            if fname_path.suffix in [".h5", ".gz", ".npy"]:
                return self.handle_special_formats(fname)
            return [FastMRIRawDataSample(fname, 0, {})]
        except Exception as e:
            logging.error(f"Error processing file {fname}: {e}")
            return None

    def handle_special_formats(self, fname_path) -> List[FastMRIRawDataSample] | None:
        fname_path = Path(fname_path)
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
            s = s + f"//{prefix}:{el}"
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

    def __getitem__(self, index):
        sample = self.raw_samples[index]
        img = self.load_image_data(sample.fname, sample.slice_ind)

        if img is None:
            return None

        img = self.preprocess(img)

        if img is None:
            return None

        return self.apply_degradation(img, sample.fname)

    def load_image_data(self, fname: str, slice_ind: int):
        try:
            fname = str(fname)
            if not os.path.exists(fname):
                logging.warning(f"File does not exist: {fname}")
                return None
            if not os.access(fname, os.R_OK):
                logging.warning(f"Cannot read file: {fname}")
                return None

            if fname.endswith(".h5"):
                with h5py.File(fname, "r") as hf:
                    img = np.array(hf["reconstruction_rss"])[slice_ind]
            elif fname.endswith(".gz") and "t1n" in fname:
                volume = nibabel.load(fname).get_fdata()
                best_slice_index = self._get_best_slice(volume)
                img = volume[:, :, best_slice_index]
            elif fname.endswith(".npy"):
                img = np.load(fname)[slice_ind]
            elif fname.endswith(".gz") and "4CH_ES.nii" in fname:
                img = nibabel.load(fname).get_fdata()
            else:
                img = util.imread_uint(fname, self.n_channels)
                img = util.uint2single(img)
        except PermissionError as e:
            logging.warning(f"Skipping file {fname} due to PermissionError: {e}")
            return None
        except Exception as e:
            logging.warning(f"Skipping file {fname} due to error: {e}")
            return None

        return img if img is not None and img.ndim >= 2 else None

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
        cropped_img = img[start_y:end_y, start_x:end_x]
        return cropped_img

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

        img_H = util.modcrop(img, self.sf)

        if img_H.shape[0] < self.h_size or img_H.shape[1] < self.h_size:
            return None

        crop_methods = {
            "high_texture": self.crop_high_texture,
            "center": self.center_crop,
        }

        img_H = crop_methods.get(self.crop_method, self.center_crop)(img_H, self.h_size)

        img_H = img_H.astype(np.float32)

        normalized_diff = np.ptp(img_H)
        if normalized_diff == 0:
            return None

        img_H = (img_H - np.min(img_H)) / (normalized_diff + np.finfo(np.float32).eps)

        if img_H.ndim == 2:
            img_H = img_H[:, :, np.newaxis]

        return img_H

    def apply_degradation(self, img, fname):  # -> dict[str, Any]:
        # "bicubic_degradation", "dpsr", "bsrgan_plus"
        chosen_model = random.choice(["dpsr", "bsrgan_plus"])
        kernel = self.select_kernel()
        img_L, img_H = {
            "dpsr": lambda x: blindsr.dpsr_degradation(
                x, kernel, self.sf, self.lq_patchsize
            ),
            "bsrgan_plus": lambda x: blindsr.degradation_bsrgan_plus(
                x, self.sf, self.lq_patchsize
            ),
            "classic_sr": lambda x: blindsr.classical_degradation(
                x, kernel, self.sf, self.lq_patchsize
            ),
            "bicubic_degradation": lambda x: blindsr.bicubic_degradation(
                x, self.sf, self.lq_patchsize
            ),
        }[chosen_model](img)

        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)
        img_L_p = self.extract_blocks(img_L, self.phw, self.overlap)
        return {
            "L": img_L,
            "L_p": img_L_p,
            "H": img_H,
            "L_path": str(fname),
            "H_path": str(fname),
        }

    def select_kernel(self):
        if self.k and "kernels" in self.k:
            kernel_index = random.randint(0, len(self.k["kernels"][0]) - 1)
            return self.k["kernels"][0][kernel_index]
        else:
            logging.warning("No kernels found, using default kernel")
            return (
                np.ones((5, 5), dtype=np.float32) / 25
            )  # Default to average blur kernel

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
