"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import os
import pickle
import random
import xml.etree.ElementTree as etree
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from warnings import warn

import h5py
import numpy as np
import pandas as pd
import requests
import torch
import yaml
import matplotlib.pyplot as plt
import utils_n.utils_image as util
import torch.nn.functional as F
from utils_n import utils_blindsr as blindsr
from scipy.io import loadmat
import nibabel

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


def fetch_dir(
    key: str, data_config_file: Union[str, Path, os.PathLike] = "fastmri_dirs.yaml"
) -> Path:
    """
    Data directory fetcher.

    This is a brute-force simple way to configure data directories for a
    project. Simply overwrite the variables for `knee_path` and `brain_path`
    and this function will retrieve the requested subsplit of the data for use.

    Args:
        key: key to retrieve path from data_config_file. Expected to be in
            ("knee_path", "brain_path", "log_path").
        data_config_file: Optional; Default path config file to fetch path
            from.

    Returns:
        The path to the specified directory.
    """
    data_config_file = Path(data_config_file)
    if not data_config_file.is_file():
        default_config = {
            "knee_path": "/path/to/knee",
            "brain_path": "/path/to/brain",
            "log_path": ".",
        }
        with open(data_config_file, "w") as f:
            yaml.dump(default_config, f)

        data_dir = default_config[key]

        warn(
            f"Path config at {data_config_file.resolve()} does not exist. "
            "A template has been created for you. "
            "Please enter the directory paths for your system to have defaults."
        )
    else:
        with open(data_config_file, "r") as f:
            data_dir = yaml.safe_load(f)[key]

    return Path(data_dir)


class FastMRIRawDataSample(NamedTuple):
    fname: Path
    slice_ind: int
    metadata: Dict[str, Any]


class CombinedSliceDataset(torch.utils.data.Dataset):
    """
    A container for combining slice datasets.
    """

    def __init__(
        self,
        roots: Sequence[Path],
        challenges: Sequence[str],
        transforms: Optional[Sequence[Optional[Callable]]] = None,
        sample_rates: Optional[Sequence[Optional[float]]] = None,
        volume_sample_rates: Optional[Sequence[Optional[float]]] = None,
        use_dataset_cache: bool = False,
        dataset_cache_file: Union[str, Path, os.PathLike] = "dataset_cache.pkl",
        num_cols: Optional[Tuple[int]] = None,
        raw_sample_filter: Optional[Callable] = None,
    ):
        """
        Args:
            roots: Paths to the datasets.
            challenges: "singlecoil" or "multicoil" depending on which
                challenge to use.
            transforms: Optional; A sequence of callable objects that
                preprocesses the raw data into appropriate form. The transform
                function should take 'kspace', 'target', 'attributes',
                'filename', and 'slice' as inputs. 'target' may be null for
                test data.
            sample_rates: Optional; A sequence of floats between 0 and 1.
                This controls what fraction of the slices should be loaded.
                When creating subsampled datasets either set sample_rates
                (sample by slices) or volume_sample_rates (sample by volumes)
                but not both.
            volume_sample_rates: Optional; A sequence of floats between 0 and 1.
                This controls what fraction of the volumes should be loaded.
                When creating subsampled datasets either set sample_rates
                (sample by slices) or volume_sample_rates (sample by volumes)
                but not both.
            use_dataset_cache: Whether to cache dataset metadata. This is very
                useful for large datasets like the brain data.
            dataset_cache_file: Optional; A file in which to cache dataset
                information for faster load times.
            num_cols: Optional; If provided, only slices with the desired
                number of columns will be considered.
            raw_sample_filter: Optional; A callable object that takes an raw_sample
                metadata as input and returns a boolean indicating whether the
                raw_sample should be included in the dataset.
        """
        if sample_rates is not None and volume_sample_rates is not None:
            raise ValueError(
                "either set sample_rates (sample by slices) or volume_sample_rates (sample by volumes) but not both"
            )
        if transforms is None:
            transforms = [None] * len(roots)
        if sample_rates is None:
            sample_rates = [None] * len(roots)
        if volume_sample_rates is None:
            volume_sample_rates = [None] * len(roots)
        if not (
            len(roots)
            == len(transforms)
            == len(challenges)
            == len(sample_rates)
            == len(volume_sample_rates)
        ):
            raise ValueError(
                "Lengths of roots, transforms, challenges, sample_rates do not match"
            )

        self.datasets = []
        self.raw_samples: List[FastMRIRawDataSample] = []
        for i in range(len(roots)):
            self.datasets.append(
                SliceDataset(
                    root=roots[i],
                    transform=transforms[i],
                    challenge=challenges[i],
                    sample_rate=sample_rates[i],
                    volume_sample_rate=volume_sample_rates[i],
                    use_dataset_cache=use_dataset_cache,
                    dataset_cache_file=dataset_cache_file,
                    num_cols=num_cols,
                    raw_sample_filter=raw_sample_filter,
                )
            )

            self.raw_samples = self.raw_samples + self.datasets[-1].raw_samples

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, i):
        for dataset in self.datasets:
            if i < len(dataset):
                return dataset[i]
            else:
                i = i - len(dataset)


class SliceDatasetSR(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        roots = opt["dataroot_H"]
        challenge = opt['challenge'] if 'challenge' in opt else 'multicoil'
        transform: Optional[Callable] = None
        use_dataset_cache = opt["use_dataset_cache"] if "use_dataset_cache" in opt else False
        sample_rate = opt["sample_rate"] if "sample_rate" in opt else None
        volume_sample_rate = opt["volume_sample_rate"] if "volume_sample_rate" in opt else None
        dataset_cache_file = opt["dataset_cache_file"] if "dataset_cache_file" in opt else "dataset_cache.pkl"
        raw_sample_filter = opt["raw_sample_filter"] if "raw_sample_filter" in opt else None
        sf = opt["scale"] if "scale" in opt else 2
        phase = opt["phase"] if "phase" in opt else 'train'
        phw = opt["phw"] if "phw" in opt else 32
        overlap = opt["overlap"] if "overlap" in opt else 4
        self.h_size = opt['H_size'] if 'H_size' in opt else 96
        self.lq_patchsize = opt['lq_patchsize'] if 'lq_patchsize' in opt else 64
        self.degradation_type = opt['degradation_type'] if 'degradation_type' in opt else 'bsrgan'
        self.sf = opt['scale'] if 'scale' in opt else 4
        self.sf = sf
        self.phase = phase
        self.phw = phw
        self.overlap = overlap
        self.k = loadmat(opt['kernel_path'])

        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        if sample_rate is not None and volume_sample_rate is not None:
            raise ValueError(
                "either set sample_rate (sample by slices) or volume_sample_rate (sample by volumes) but not both"
            )

        self.dataset_cache_file = Path(dataset_cache_file)

        self.transform = transform
        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )
        self.raw_samples = []
        if raw_sample_filter is None:
            self.raw_sample_filter = lambda raw_sample: True
        else:
            self.raw_sample_filter = raw_sample_filter

        if self.dataset_cache_file.exists() and use_dataset_cache:
            with open(self.dataset_cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
        else:
            dataset_cache = {}

        files = util.get_m_image_paths(roots)

        if dataset_cache.get(tuple(files)) is None or not use_dataset_cache:
            for fname in sorted(files):

                if fname.endswith('.h5'):
                    try:
                        metadata, num_slices = self._retrieve_metadata(fname)
                    except (OSError, KeyError, ValueError) as e:
                        logging.warning(f"Skipping file {fname} due to error: {e}")
                        continue

                    new_raw_samples = []
                    for slice_ind in range(num_slices):
                        raw_sample = FastMRIRawDataSample(fname, slice_ind, metadata)
                        if self.raw_sample_filter(raw_sample):
                            new_raw_samples.append(raw_sample)

                    self.raw_samples += new_raw_samples

                elif fname.endswith('.gz'):
                    self.raw_samples.append(FastMRIRawDataSample(fname, 0, {}))
                
                else:
                    self.raw_samples.append(FastMRIRawDataSample(fname, 0, {}))
                    

        else:
            logging.info(f"Using dataset cache from {self.dataset_cache_file}.")
            self.raw_samples = dataset_cache[tuple(files)]
    
    def _get_best_slice(self, volume):
        variances = [np.var(volume[:, :, i]) for i in range(volume.shape[2])]
        return np.argmax(variances)
    
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

    def __len__(self):
        return len(self.raw_samples)

    def center_crop(self, img, crop_size):
        center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
        crop_x, crop_y = crop_size[1] // 2, crop_size[0] // 2

        start_x = max(center_x - crop_x, 0)
        start_y = max(center_y - crop_y, 0)

        end_x = min(start_x + crop_size[1], img.shape[1])
        end_y = min(start_y + crop_size[0], img.shape[0])

        cropped_img = img[start_y:end_y, start_x:end_x]
        return cropped_img
    
    @staticmethod
    def extract_blocks(img_tensor, block_size, overlap):
        blocks = []
        step = block_size - overlap
        for i in range(0, img_tensor.shape[1] - block_size + 1, step):
            for j in range(0, img_tensor.shape[2] - block_size + 1, step):
                block = img_tensor[:, i:i+block_size, j:j+block_size]
                blocks.append(block)
        return torch.stack(blocks)

    def __getitem__(self, i: int):
        img_H = None
        fname, dataslice, metadata = self.raw_samples[i]
        try:
            if fname.endswith('.h5'):
                with h5py.File(fname, "r") as hf:
                    img_H = hf[self.recons_key][dataslice] if self.recons_key in hf else None
            elif fname.endswith('.gz') and 't1n' in fname:
                nib_img = nibabel.load(fname)
                volume = nib_img.get_fdata()
                best_slice_index = self._get_best_slice(volume)
                img_H = volume[:, :, best_slice_index]
            elif fname.endswith('.gz') and '4CH_ES.nii' in fname:
                nib_img = nibabel.load(fname)
                img_H = nib_img.get_fdata()
            else:
                img_H = util.imread_uint(fname, self.n_channels)
                img_H = util.uint2single(img_H)
        except PermissionError as e:
            logging.warning(f"Skipping file {fname} due to PermissionError: {e}")
            return None
        except Exception as e:
            logging.warning(f"Skipping file {fname} due to error: {e}")
            return None

        img_H = np.clip(img_H, np.quantile(img_H, 0.001), np.quantile(img_H, 0.999))
        img_H = (img_H - np.min(img_H)) / (np.max(img_H) - np.min(img_H))

        if img_H is None or img_H.ndim == 0:
            logging.warning(f"Skipping file {fname} due to zero dimensions.")
            return None

        img_H = util.modcrop(img_H, self.sf)
        img_H = self.center_crop(img_H, (self.h_size, self.h_size))

        if img_H.ndim >= 2:
            h, w = img_H.shape[:2]
            if h < self.lq_patchsize * self.sf or w < self.lq_patchsize * self.sf:
                return None
        else:
            return None

        if img_H.ndim == 2:
            img_H = img_H[:, :, np.newaxis]

        if self.phase == 'train':
            mode = random.randint(0, 7)
            img_H = util.augment_img(img_H, mode=mode)

        if self.degradation_type == 'bsrgan':
            img_L, img_H = blindsr.degradation_bsrgan(img_H, sf=self.sf, lq_patchsize=self.lq_patchsize)
        elif self.degradation_type == 'bsrgan_plus':
            img_L, img_H = blindsr.degradation_bsrgan_plus(img_H, sf=self.sf, lq_patchsize=self.lq_patchsize)
        elif self.degradation_type == 'dpsr':
            img_L = blindsr.dpsr_degradation(img_H, k=self.k['kernels'][0][1], sf=self.sf)

        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)

        img_L_p = self.extract_blocks(img_L, self.phw, self.overlap)

        return {'L': img_L, 'L_p': img_L_p, 'H': img_H, 'L_path': str(fname), 'H_path': str(fname)}


class SliceDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        challenge: str,
        transform: Optional[Callable] = None,
        use_dataset_cache: bool = False,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        dataset_cache_file: Union[str, Path, os.PathLike] = "dataset_cache.pkl",
        num_cols: Optional[Tuple[int]] = None,
        raw_sample_filter: Optional[Callable] = None,
        sf :int = 2,
        phase : str = 'train',
        phw : int = 32,
        stride: int = 4
    ):
        """
        Args:
            root: Path to the dataset.
            challenge: "singlecoil" or "multicoil" depending on which challenge
                to use.
            transform: Optional; A callable object that pre-processes the raw
                data into appropriate form. The transform function should take
                'kspace', 'target', 'attributes', 'filename', and 'slice' as
                inputs. 'target' may be null for test data.
            use_dataset_cache: Whether to cache dataset metadata. This is very
                useful for large datasets like the brain data.
            sample_rate: Optional; A float between 0 and 1. This controls what fraction
                of the slices should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            volume_sample_rate: Optional; A float between 0 and 1. This controls what fraction
                of the volumes should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            dataset_cache_file: Optional; A file in which to cache dataset
                information for faster load times.
            num_cols: Optional; If provided, only slices with the desired
                number of columns will be considered.
            raw_sample_filter: Optional; A callable object that takes an raw_sample
                metadata as input and returns a boolean indicating whether the
                raw_sample should be included in the dataset.
        """
        self.sf = sf
        self.phase = phase
        self.phw = phw
        self.stride = stride
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        if sample_rate is not None and volume_sample_rate is not None:
            raise ValueError(
                "either set sample_rate (sample by slices) or volume_sample_rate (sample by volumes) but not both"
            )

        self.dataset_cache_file = Path(dataset_cache_file)

        self.transform = transform
        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )
        self.raw_samples = []
        if raw_sample_filter is None:
            self.raw_sample_filter = lambda raw_sample: True
        else:
            self.raw_sample_filter = raw_sample_filter

        # set default sampling mode if none given
        if sample_rate is None:
            sample_rate = 1.0
        if volume_sample_rate is None:
            volume_sample_rate = 1.0

        # load dataset cache if we have and user wants to use it
        if self.dataset_cache_file.exists() and use_dataset_cache:
            with open(self.dataset_cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
        else:
            dataset_cache = {}

        # check if our dataset is in the cache
        # if there, use that metadata, if not, then regenerate the metadata
        if dataset_cache.get(root) is None or not use_dataset_cache:
            files = list(Path(root).iterdir())
            for fname in sorted(files):
                metadata, num_slices = self._retrieve_metadata(fname)

                new_raw_samples = []
                for slice_ind in range(num_slices):
                    raw_sample = FastMRIRawDataSample(fname, slice_ind, metadata)
                    if self.raw_sample_filter(raw_sample):
                        new_raw_samples.append(raw_sample)

                self.raw_samples += new_raw_samples

            if dataset_cache.get(root) is None and use_dataset_cache:
                dataset_cache[root] = self.raw_samples
                logging.info(f"Saving dataset cache to {self.dataset_cache_file}.")
                with open(self.dataset_cache_file, "wb") as cache_f:
                    pickle.dump(dataset_cache, cache_f)
        else:
            logging.info(f"Using dataset cache from {self.dataset_cache_file}.")
            self.raw_samples = dataset_cache[root]

        # subsample if desired
        if sample_rate < 1.0:  # sample by slice
            random.shuffle(self.raw_samples)
            num_raw_samples = round(len(self.raw_samples) * sample_rate)
            self.raw_samples = self.raw_samples[:num_raw_samples]
        elif volume_sample_rate < 1.0:  # sample by volume
            vol_names = sorted(list(set([f[0].stem for f in self.raw_samples])))
            random.shuffle(vol_names)
            num_volumes = round(len(vol_names) * volume_sample_rate)
            sampled_vols = vol_names[:num_volumes]
            self.raw_samples = [
                raw_sample
                for raw_sample in self.raw_samples
                if raw_sample[0].stem in sampled_vols
            ]

        if num_cols:
            self.raw_samples = [
                ex
                for ex in self.raw_samples
                if ex[2]["encoding_size"][1] in num_cols  # type: ignore
            ]

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

    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, i: int):
        fname, dataslice, metadata = self.raw_samples[i]

        with h5py.File(fname, "r") as hf:
            kspace = hf["kspace"][dataslice]

            img_H = hf[self.recons_key][dataslice] if self.recons_key in hf else None

            k_image = np.abs(np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(kspace))))
            
            if img_H is not None:
                target_shape = img_H.shape
                start_row = (k_image.shape[0] - target_shape[0]) // 2
                start_col = (k_image.shape[1] - target_shape[1]) // 2

                img_L = k_image[start_row:start_row + target_shape[0], start_col:start_col + target_shape[1]]

            img_H = util.uint2single(img_H)
            img_H = util.modcrop(img_H, self.sf)

            img_H = img_H[:, :, np.newaxis]
            img_L = img_L.transpose(1, 2, 0)
            img_L = util.modcrop(img_L, self.sf)
            img_L = util.imresize_np(img_L, 1 / self.sf, True)
        
            if self.phase == 'train':
                mode = random.randint(0, 7)
                img_L, img_H = util.augment_img(img_L, mode=mode), util.augment_img(img_H, mode=mode)

            img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)

            img_L_p = img_L.unfold(1, self.phw, self.stride).unfold(
                2, self.phw, self.stride
            )
            img_L_p = F.max_pool3d(img_L_p, kernel_size=1, stride=1)
            img_L_p = img_L_p.view(
                img_L_p.shape[1] * img_L_p.shape[2],
                img_L_p.shape[0],
                img_L_p.shape[3],
                img_L_p.shape[4],
            )

            # mask = np.asarray(hf["mask"]) if "mask" in hf else None

            # attrs = dict(hf.attrs)
            # attrs.update(metadata)

        # if self.transform is None:
        #     sample = (img_L, mask, target, attrs, fname.name, dataslice)
        # else:
        #     sample = self.transform(k_image, mask, target, attrs, fname.name, dataslice)

        return {'L': img_L,'L_p': img_L_p, 'H': img_H, 'L_path': str(fname), 'H_path': str(fname)}


class AnnotatedSliceDataset(SliceDataset):
    """
    A PyTorch Dataset that provides access to MR image slices with annotation.

    This is a subclass from SliceDataset that incorporates functionality of the fastMRI+ dataset.
    It can be used to download the csv file from fastMRI+ based on the specified version using git hash.
    It parses the csv and links it to samples in SliceDataset as annotated_raw_samples.

    Github: https://github.com/microsoft/fastmri-plus
    Paper: https://arxiv.org/abs/2109.03812
    """

    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        challenge: str,
        subsplit: str,
        multiple_annotation_policy: str,
        transform: Optional[Callable] = None,
        use_dataset_cache: bool = False,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        dataset_cache_file: Union[str, Path, os.PathLike] = "dataset_cache.pkl",
        num_cols: Optional[Tuple[int]] = None,
        annotation_version: Optional[str] = None,
    ):
        """
        Args:
            root: Path to the dataset.
            challenge: "singlecoil" or "multicoil" depending on which challenge
                to use.
            subsplit: 'knee' or 'brain' depending on which dataset to use.
            multiple_annotation_policy: 'first', 'random' or 'all'.
                If 'first', then only use the first annotation.
                If 'random', then pick an annotation at random.
                If 'all' then two or more copies of the same slice for each annotation
                will be extended.
            transform: Optional; A callable object that pre-processes the raw
                data into appropriate form. The transform function should take
                'kspace', 'target', 'attributes', 'filename', and 'slice' as
                inputs. 'target' may be null for test data.
            use_dataset_cache: Whether to cache dataset metadata. This is very
                useful for large datasets like the brain data.
            sample_rate: Optional; A float between 0 and 1. This controls what fraction
                of the slices should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            volume_sample_rate: Optional; A float between 0 and 1. This controls what fraction
                of the volumes should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            dataset_cache_file: Optional; A file in which to cache dataset
                information for faster load times.
            num_cols: Optional; If provided, only slices with the desired
                number of columns will be considered.
            annotation_version: Optional; If provided, a specific version of csv file will be downloaded based on its git hash.
                Default value is None, then the latest version will be used.
        """

        # subclass SliceDataset
        super().__init__(
            root,
            challenge,
            transform,
            use_dataset_cache,
            sample_rate,
            volume_sample_rate,
            dataset_cache_file,
            num_cols,
        )

        self.annotated_raw_samples = []

        if subsplit not in ("knee", "brain"):
            raise ValueError('subsplit should be either "knee" or "brain"')
        if multiple_annotation_policy not in ("first", "random", "all"):
            raise ValueError(
                'multiple_annotation_policy should be "single", "random", or "all"'
            )

        # download csv file from github using git hash to find certain version
        annotation_name = f"{subsplit}{annotation_version}.csv"
        annotation_path = Path(os.getcwd(), ".annotation_cache", annotation_name)
        if not annotation_path.is_file():
            annotation_path = self.download_csv(
                annotation_version, subsplit, annotation_path
            )
        annotations_csv = pd.read_csv(annotation_path)

        for raw_sample in self.raw_samples:
            fname, slice_ind, metadata = raw_sample

            # using filename and slice to find desired annotation
            annotations_df = annotations_csv[
                (annotations_csv["file"] == fname.stem)
                & (annotations_csv["slice"] == slice_ind)
            ]
            annotations_list = annotations_df.itertuples(index=True, name="Pandas")

            # if annotation (filename or slice) not found, fill in empty values
            if len(annotations_df) == 0:
                annotation = self.get_annotation(True, None)
                metadata["annotation"] = annotation
                self.annotated_raw_samples.append(
                    list([fname, slice_ind, metadata.copy()])
                )

            elif len(annotations_df) == 1:
                rows = list(annotations_list)[0]
                annotation = self.get_annotation(False, rows)
                metadata["annotation"] = annotation
                self.annotated_raw_samples.append(
                    list([fname, slice_ind, metadata.copy()])
                )

            else:
                # only use the first annotation
                if multiple_annotation_policy == "first":
                    rows = list(annotations_list)[0]
                    annotation = self.get_annotation(False, rows)
                    metadata["annotation"] = annotation
                    self.annotated_raw_samples.append(
                        list([fname, slice_ind, metadata.copy()])
                    )

                # use an annotation at random
                elif multiple_annotation_policy == "random":
                    random_number = torch.randint(len(annotations_df) - 1, (1,))
                    rows = list(annotations_list)[random_number]
                    annotation = self.get_annotation(False, rows)
                    metadata["annotation"] = annotation
                    self.annotated_raw_samples.append(
                        list([fname, slice_ind, metadata.copy()])
                    )

                # extend raw samples to have tow copies of the same slice, one for each annotation
                elif multiple_annotation_policy == "all":
                    for rows in annotations_list:
                        annotation = self.get_annotation(False, rows)
                        metadata["annotation"] = annotation
                        self.annotated_raw_samples.append(
                            list([fname, slice_ind, metadata.copy()])
                        )

    def get_annotation(self, empty_value, row):
        if empty_value is True:
            annotation = {
                "fname": "",
                "slice": "",
                "study_level": "",
                "x": -1,
                "y": -1,
                "width": -1,
                "height": -1,
                "label": "",
            }
        elif row.study_level == "Yes":
            annotation = {
                "fname": str(row.file),
                "slice": "",
                "study_level": "Yes",
                "x": -1,
                "y": -1,
                "width": -1,
                "height": -1,
                "label": str(row.label),
            }
        else:
            annotation = {
                "fname": str(row.file),
                "slice": int(row.slice),
                "study_level": str(row.study_level),
                "x": int(row.x),
                "y": 320 - int(row.y) - int(row.height) - 1,
                "width": int(row.width),
                "height": int(row.height),
                "label": str(row.label),
            }
        return annotation

    def download_csv(self, version, subsplit, path):
        # request file by git hash and mri type
        if version is None:
            url = f"https://raw.githubusercontent.com/microsoft/fastmri-plus/main/Annotations/{subsplit}.csv"
        else:
            url = f"https://raw.githubusercontent.com/microsoft/fastmri-plus/{version}/Annotations/{subsplit}.csv"
        request = requests.get(url, timeout=10, stream=True)

        # create temporary folders
        Path(".annotation_cache").mkdir(parents=True, exist_ok=True)

        # download csv from github and save it locally
        with open(path, "wb") as fh:
            for chunk in request.iter_content(1024 * 1024):
                fh.write(chunk)
        return path
