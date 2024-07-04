# import asyncio
# import concurrent.futures
import logging
import os
import pickle
import random
import xml.etree.ElementTree as etree

# from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, NamedTuple, Sequence

import h5py
import nibabel
import numpy as np
import torch
from scipy.io import loadmat

import utils_n.utils_image as util
from utils_n import utils_blindsr as blindsr


class FastMRIRawDataSample(NamedTuple):
    fname: Path
    slice_ind: int
    metadata: Dict[str, Any]


class MedicalDatasetSR(torch.utils.data.Dataset):
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
        self.phw = opt["phw"] if "phw" in opt else 32
        self.overlap = opt["overlap"] if "overlap" in opt else 4
        self.recons_key = (
            "reconstruction_esc"
            if self.challenge == "singlecoil"
            else "reconstruction_rss"
        )
        self.k = loadmat(opt["kernel_path"]) if "kernel_path" in opt else None
        self.raw_samples = self.load_samples()
        # self.executor = concurrent.futures.ThreadPoolExecutor(
        #     max_workers=min(16, cpu_count() * 2)
        # )
        # self.loop = asyncio.get_event_loop()
        # self.raw_samples = self.loop.run_until_complete(self.async_load_samples())

    # async def async_load_samples(self):

    #     if self.use_dataset_cache and os.path.exists(self.dataset_cache_file):
    #         with open(self.dataset_cache_file, "rb") as f:
    #             return pickle.load(f)

    #     if self.use_dataset_cache:
    #         files = util.get_m_image_paths(self.roots)
    #         tasks = [self.loop.run_in_executor(self.executor, self.load_sample, fname) for fname in files]
    #         results = await asyncio.gather(*tasks)
    #         results = [sample for result in results if result for sample in result]

    #         with open(self.dataset_cache_file, "wb") as f:
    #             pickle.dump(results, f)
    #     return results

    def load_samples(self):
        if self.use_dataset_cache and os.path.exists(self.dataset_cache_file):
            with open(self.dataset_cache_file, "rb") as f:
                return pickle.load(f)

        files = util.get_m_image_paths(self.roots)
        results = []
        for fname in files:
            sample = self.load_sample(fname)
            if sample:
                results.extend(sample)

        if self.use_dataset_cache:
            with open(self.dataset_cache_file, "wb") as f:
                pickle.dump(results, f)
        return results

    def load_sample(self, fname):
        try:
            if not self.is_valid_image(fname):
                logging.warning(f"Skipping non-image or system file: {fname}")
                return None
            if fname.endswith(".h5"):
                metadata, num_slices = self._retrieve_metadata(fname)
                return [
                    FastMRIRawDataSample(fname, slice_ind, metadata)
                    for slice_ind in range(num_slices)
                ]
            elif fname.endswith(".gz") or fname.endswith(".npy"):
                return [FastMRIRawDataSample(fname, 0, {})]
            elif any(fname.lower().endswith(ext) for ext in util.IMG_EXTENSIONS):
                return [FastMRIRawDataSample(fname, 0, {})]
            else:
                logging.warning(f"Unsupported file type: {fname}")
                return None
        except Exception as e:
            logging.error(f"Error processing file {fname}: {e}")
            return None

    @staticmethod
    def et_query(
        root: etree.Element,
        qlist: Sequence[str],
        namespace: str = "http://www.ismrm.org/ISMRMRD",
    ) -> str:
        """
        ElementTree query function.

        This can be used to query an xml document via ElementTree. It uses qlist
        for nested queries.

        Args:
            root: Root of the xml to search through.
            qlist: A list of strings for nested searches, e.g. ["Encoding",
                "matrixSize"]
            namespace: Optional; xml namespace to prepend query.

        Returns:
            The retrieved data as a string.
        """
        s = "."
        prefix = "ismrmrd_namespace"

        ns = {prefix: namespace}

        for el in qlist:
            s = s + f"//{prefix}:{el}"

        value = root.find(s, ns)
        if value is None:
            raise RuntimeError("Element not found")

        return str(value.text)

    def _retrieve_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            et_root = etree.fromstring(hf["ismrmrd_header"][()])

            enc = ["encoding", "encodedSpace", "matrixSize"]
            enc_size = (
                int(et_query(et_root, enc + ["x"])),
                int(et_query(et_root, enc + ["y"])),
                int(et_query(et_root, enc + ["z"])),
            )
            rec = ["encoding", "reconSpace", "matrixSize"]
            recon_size = (
                int(et_query(et_root, rec + ["x"])),
                int(et_query(et_root, rec + ["y"])),
                int(et_query(et_root, rec + ["z"])),
            )

            lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
            enc_limits_center = int(et_query(et_root, lims + ["center"]))
            enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1

            padding_left = enc_size[1] // 2 - enc_limits_center
            padding_right = padding_left + enc_limits_max

            num_slices = hf["kspace"].shape[0]

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

    def load_image_data(self, fname, slice_ind):
        try:
            if fname.endswith(".h5"):
                with h5py.File(fname, "r") as hf:
                    img = (
                        hf[self.recons_key][slice_ind]
                        if self.recons_key in hf
                        else None
                    )
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

    def preprocess(self, img):
        if img is None or img.ndim < 2 or np.max(img) == np.min(img):
            return None

        img_H = util.modcrop(img, self.sf)

        if img_H.shape[0] < self.h_size or img_H.shape[1] < self.h_size:
            return None

        img_H = self.center_crop(img_H, self.h_size)

        img_H = img_H.astype(np.float32)

        normalized_diff = np.ptp(img_H)
        if normalized_diff == 0:
            return None

        img_H = (img_H - np.min(img_H)) / (normalized_diff + np.finfo(np.float32).eps)

        if img_H.ndim == 2:
            img_H = img_H[:, :, np.newaxis]

        return img_H

    def apply_degradation(self, img, fname):
        chosen_model = random.choice(["dpsr", "bsrgan_plus"])
        img_L, img_H = {
            # 'bsrgan': lambda x: blindsr.degradation_bsrgan(x, self.sf, self.lq_patchsize),
            "dpsr": lambda x: blindsr.dpsr_degradation(
                x, self.k["kernels"][0][1], self.sf
            ),
            "bsrgan_plus": lambda x: blindsr.degradation_bsrgan_plus(
                x, self.sf, self.lq_patchsize
            ),
        }[chosen_model](img)

        # img_L, img_H = blindsr.degradation_bsrgan_plus(img, self.sf, self.lq_patchsize)

        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)
        img_L_p = self.extract_blocks(img_L, self.phw, self.overlap)
        return {
            "L": img_L,
            "L_p": img_L_p,
            "H": img_H,
            "L_path": str(fname),
            "H_path": str(fname),
        }

    @staticmethod
    def extract_blocks(img_tensor, block_size, overlap):
        blocks = []
        step = block_size - overlap
        for i in range(0, img_tensor.shape[1] - block_size + 1, step):
            for j in range(0, img_tensor.shape[2] - block_size + 1, step):
                blocks.append(img_tensor[:, i : i + block_size, j : j + block_size])
        return torch.stack(blocks)

    def __len__(self):
        return len(self.raw_samples)

    def is_valid_image(self, file_path):
        return not file_path.split("/")[-1].startswith(".")
