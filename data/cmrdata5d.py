"""
Dataset classes for fastMRI, Calgary-Campinas, CMRxRecon datasets
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
import h5py
import numpy as np
import torch
import torch.utils

from mri_utils.utils import load_shape
from mri_utils import load_kdata, load_mask

from .blacklist import FileBoundBlacklist
from .subsample import MaskFunc, FixedLowEquiSpacedMaskFunc, FixedLowRandomMaskFunc, temp_seed, CmrxRecon24MaskFunc, PoissonDiscMaskFunc
from .transforms import to_tensor, PromptMRSample
from einops import rearrange

#########################################################################################################
# Common functions
#########################################################################################################


class RawDataSample(NamedTuple):
    """
    A container for raw data samples.
    """
    fname: Path
    slice_ind: int
    metadata: Dict[str, Any]


class BalanceSampler:
    def __init__(self, ratio_dict={'a':1, 'b':2}):
        self.ratio_dict = ratio_dict
        
    def __call__(self, raw_sample: List[RawDataSample]):
        # create dict, keys with empty list
        dict_list = {key: [] for key in self.ratio_dict.keys()}
        
        # for key, value in self.ratio_dict.items():
        for raw_i in raw_sample:
            for key in dict_list.keys():
                if key in str(raw_i.fname):
                    dict_list[key].append(raw_i)
                    break
        # combine to final list multiply with ratio 
        final_list = []
        for key, value in self.ratio_dict.items():
            final_list += dict_list[key] * value

        return final_list

#########################################################################################################
# CMRxRecon Transform
#########################################################################################################

class CmrxRecon5dTransform:
    """
    CmrxRecon23&24 Data Transformer for training
    """

    def __init__(self, mask_func: Optional[MaskFunc] = None, uniform_resolution= None, use_seed: bool = True, mask_type: Optional[str] = None, test_num_low_frequencies: Optional[int] = None):
        """
        Args:
            mask_func: Optional; A function that can create a mask of
                appropriate shape. Defaults to None.
            use_seed: If True, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if mask_func is None and mask_type is None:
            raise ValueError("Either `mask_func` or `mask_type` must be specified.")
        if mask_func is not None and mask_type is not None:
            raise ValueError("Both `mask_func` and `mask_type` cannot be set at the same time.")
    
        self.mask_func = mask_func
        self.use_seed = use_seed
        self.uniform_resolution = uniform_resolution
        # when training, mask_type will be returned by mask_func
        # when inference, we need to specify mask_type
        # so should check mask_func and mask_type not set at the same time or none at the same time
        if mask_func is None:
            self.mask_type = mask_type
            self.num_low_frequencies = test_num_low_frequencies

    def apply_mask(
        self,
        data: torch.Tensor,
        mask_func: MaskFunc,
        offset: Optional[int] = None,
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
        padding: Optional[Sequence[int]] = None,
        slice_idx: Optional[int] = None,
        num_t: Optional[int] = None,
        num_slc: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Subsample given k-space by multiplying with a mask.

        Args:
            data: The input k-space data. This should have at least 3 dimensions,
                where dimensions -3 and -2 are the spatial dimensions, and the
                final dimension has size 2 (for complex values).
            mask_func: A function that takes a shape (tuple of ints) and a random
                number seed and returns a mask.
            seed: Seed for the random number generator.
            padding: Padding value to apply for mask.

        Returns:
            tuple containing:
                masked data: Subsampled k-space data.
                mask: The generated mask.
                num_low_frequencies: The number of low-resolution frequency samples
                    in the mask.
        """
        shape = (1,) * len(data.shape[:-3]) + tuple(data.shape[-3:])
        if isinstance(mask_func, CmrxRecon24MaskFunc):
            if num_t is not None:
                mask, num_low_frequencies, mask_type = mask_func(shape, offset, seed, slice_idx,num_t,num_slc) # here
            else:
                mask, num_low_frequencies, mask_type = mask_func(shape, offset, seed)
            if mask.size(0) == 1:
                mask = mask.expand(data.size(0), -1, -1, -1, -1)
        else:
            if isinstance(mask_func, PoissonDiscMaskFunc):
                mask_type = 'poisson_disc'
            else:
                mask_type = 'cartesian'
            mask, num_low_frequencies = mask_func(shape, offset, seed)
        if padding is not None:
            mask[..., : padding[0], :] = 0
            mask[..., padding[1] :, :] = 0  # padding value inclusive on right of zeros
        
        # if mask.shape[0]!=1: # repeat for coil [cmr24 data]
        #     mask = mask.repeat_interleave(data.shape[0]//mask.shape[0], dim=0)
        
        if mask.ndim == 4: # radial
            mask = mask.unsqueeze(1)
        masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

        return masked_data, mask, num_low_frequencies, mask_type
    
    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
        num_t: int,
        num_slc: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            A tuple containing, zero-filled input image, the reconstruction
            target, the mean used for normalization, the standard deviations
            used for normalization, the filename, and the slice number.
        """

        if target is not None:
            target_torch = to_tensor(target)
            max_value = attrs["max"]
        else:
            target_torch = torch.tensor(0)
            max_value = 0.0
        kspace_torch = to_tensor(kspace)
        seed = None if not self.use_seed else tuple(map(ord, fname)) # so in validation, the same fname (volume) will have the same acc
        acq_start = attrs["padding_left"]
        acq_end = attrs["padding_right"]
        crop_size = (attrs["recon_size"][0], attrs["recon_size"][1]) 

        if self.mask_func is not None:
            t, s, c, h, w, two = kspace_torch.shape
            kspace_torch = rearrange(kspace_torch, "t s c h w two -> t (s c) h w two")
            masked_kspace, mask_torch, num_low_frequencies,mask_type = self.apply_mask(
                kspace_torch, self.mask_func, seed=seed, padding=(acq_start, acq_end), slice_idx=slice_num, num_t=num_t,num_slc=num_slc
            )
            masked_kspace = rearrange(masked_kspace, "t (s c) h w two -> t s c h w two", s = s, c = c)
            mask_torch = rearrange(mask_torch, "t (s c) h w two -> t s c h w two", s = 1, c = 1)
        else:
            masked_kspace = kspace_torch
            mask_torch = to_tensor(mask)
            mask_torch[:, :, :acq_start] = 0
            mask_torch[:, :, acq_end:] = 0
            if 'ktRadial' in fname:
                mask_type = 'kt_radial'
            else:
                mask_type = 'cartesian'
            num_low_frequencies = self.num_low_frequencies

        masked_kspace = masked_kspace.float()
        target_torch = target_torch.float()
        sample = PromptMRSample(
            masked_kspace=masked_kspace,
            mask=mask_torch.to(torch.bool),
            num_low_frequencies=num_low_frequencies,
            target=target_torch,
            fname=fname,
            slice_num=slice_num,
            max_value=max_value,
            crop_size=crop_size,
            mask_type=mask_type,
            num_t=num_t,
            num_slc=num_slc
            # attrs=attrs,
        )

        return sample

#########################################################################################################
# CMRxRecon dataset
#########################################################################################################


class CmrxRecon5dDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for the CMRxRecon 2025 challenge.
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
        data_balancer: Optional[Callable] = None,
        n_adj_frame: int = 3,
        n_adj_slice: int = 3,
    ):
        self.root = root
        if 'train' in str(root):
            self._split = 'train'
        elif 'val' in str(root):
            self._split = 'val'
        else:
            self._split = 'test'

        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError(
                'challenge should be either "singlecoil" or "multicoil"')

        if sample_rate is not None and volume_sample_rate is not None:
            raise ValueError(
                "either set sample_rate (sample by slices) or volume_sample_rate (sample by volumes) but not both"
            )

        self.dataset_cache_file = Path(dataset_cache_file)

        self.transform = transform

        assert n_adj_frame % 2 == 1, "Number of adjacent frames must be odd in SliceDataset"
        assert n_adj_slice % 2 == 1, "Number of adjacent slices must be odd in SliceDataset"

        # max temporal slice number is 12
        assert n_adj_frame <= 5 and n_adj_slice <= 5, "Number of adjacent must be less than 11 in CMRxRecon SliceDataset"

        self.n_adj_frame, self.n_adj_slice = n_adj_frame, n_adj_slice

        self.adj_frame_offset = - (self.n_adj_frame//2), self.n_adj_frame//2+1
        self.adj_slice_offset = - (self.n_adj_slice//2), self.n_adj_slice//2+1

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
                with h5py.File(fname, 'r') as hf:
                    # print('load debug: ', fname, hf.keys())
                    attrs = dict(hf.attrs)
                    if len(attrs['shape']) == 5:
                        num_slices = attrs['shape'][0]*attrs['shape'][1]
                    elif len(attrs['shape']) == 4:
                        num_slices = attrs['shape'][0]
                    else:
                        raise ValueError(f"Unsupported data formats: {fname} with shape {attrs['shape']}")
                    metadata = {**hf.attrs}
                new_raw_samples = []
                for slice_ind in range(num_slices):
                    raw_sample = RawDataSample(fname, slice_ind, metadata)
                    if self.raw_sample_filter(raw_sample):
                        new_raw_samples.append(raw_sample)
                self.raw_samples += new_raw_samples

            if dataset_cache.get(root) is None and use_dataset_cache:
                dataset_cache[root] = self.raw_samples
                logging.info(
                    "Saving dataset cache to %s.", self.dataset_cache_file)
                with open(self.dataset_cache_file, "wb") as cache_f:
                    pickle.dump(dataset_cache, cache_f)
        else:
            logging.info(
                "Using dataset cache from %s.", self.dataset_cache_file)
            self.raw_samples = dataset_cache[root]

        if 'train' in str(root) and data_balancer is not None:
            self.raw_samples = data_balancer(self.raw_samples)

        # subsample if desired
        if sample_rate < 1.0:  # sample by slice
            random.shuffle(self.raw_samples)
            num_raw_samples = round(len(self.raw_samples) * sample_rate)
            self.raw_samples = self.raw_samples[:num_raw_samples]
        elif volume_sample_rate < 1.0:  # sample by volume
            vol_names = sorted(
                list(set([f[0].stem for f in self.raw_samples])))
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

    # def _get_frames_indices(self, ti, num_frames):
    #     if not 0 <= ti < num_frames:
    #         raise ValueError(f"Invalid temporal index {ti} for volume with only one time point.")
    #     if num_frames == 1:
    #         return [0] * self.n_adj_frame
    #     start_lim, end_lim = -(num_frames//2), (num_frames//2+1)
    #     start, end = max(self.adj_frame_offset[0], start_lim), min(self.adj_frame_offset[1], end_lim)
    #     # Generate initial list of indices
    #     ti_idx_list = [(i + ti) % num_frames for i in range(start, end)]
    #     # duplicate padding if necessary
    #     replication_prefix = max(start_lim-self.adj_frame_offset[0], 0) * ti_idx_list[0:1]
    #     replication_suffix = max(self.adj_frame_offset[0]-end_lim, 0) * ti_idx_list[-1:]
    #     indice = replication_prefix + ti_idx_list + replication_suffix

    #     # if len(indice) == 1:
    #     #     indice = indice * self.n_adj_frame

    def _get_frames_indices(self, ti, num_frames):
        if not 0 <= ti < num_frames:
            raise ValueError(f"Invalid temporal index {ti} for volume with only one time point.")
        if num_frames == 1:
            return [0] * self.n_adj_frame

        ti_idx_list = [(i + ti) % num_frames for i in range(self.adj_frame_offset[0], self.adj_frame_offset[1])]

        return ti_idx_list
        
    def _get_slices_indices(self, zi, num_slices):
        z_list = [min(max(i+zi, 0), num_slices-1)
                  for i in range(self.adj_slice_offset[0], self.adj_slice_offset[1])]
        return z_list

    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, i: int):
        fname, data_slice, metadata = self.raw_samples[i]
        kspace = []
        with h5py.File(str(fname), 'r') as hf:
            kspace_volume = hf["kspace"]
            attrs = dict(hf.attrs)
            if len(attrs['shape']) == 5:
                num_t = int(attrs['shape'][0])
                num_slices = int(attrs['shape'][1])
                ti = data_slice // num_slices
                zi = data_slice - ti*num_slices
                mask = np.asarray(hf["mask"]) if "mask" in hf else None
                target = hf[self.recons_key][ti,zi] if self.recons_key in hf else None
                indices = self._get_frames_indices(ti, num_t), self._get_slices_indices(zi, num_slices)

                kspace = []
                for adjti in indices[0]:
                    for adjzi in indices[1]:
                        kspace.append(kspace_volume[adjti, adjzi])
                kspace = np.stack(kspace, axis = 0)
                kspace = rearrange(kspace, "(t s) ... -> t s ...", t = len(indices[0]), s = len(indices[1]))

            elif len(attrs['shape']) == 4: # 
                num_t = 1
                num_slices = int(attrs['shape'][0])
                # kspace_volume = kspace_volume[np.newaxis,:,:,:,:]
                ti = 0
                zi = data_slice

                mask = np.asarray(hf["mask"]) if "mask" in hf else None
                target = hf[self.recons_key][data_slice] if self.recons_key in hf else None
                indices = self._get_frames_indices(ti, num_t), self._get_slices_indices(zi, num_slices)

                kspace = []
                for adjzi in indices[1]:
                        kspace.append(kspace_volume[adjzi])
                kspace = np.stack(kspace, axis = 0)
                kspace = kspace[np.newaxis, :, :, :, :]

            else:
                raise ValueError(f"Unsupported data formats: {fname} with shape {attrs['shape']}")

            if mask is not None:
                mask = mask[indices[0]]
            
        if self.transform is None:
            sample = (kspace, mask, target, attrs, fname.name, data_slice, num_t)
        else:
            sample = self.transform(kspace, mask, target, attrs, fname.name, data_slice, num_t, num_slices)
        return sample


class CmrxReconInferenceSliceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        challenge: str,
        raw_sample_filter: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        num_adj_slices: int = 5
    ):
        self.root = root
        # get all the kspace mat files from root, under folder or its subfolders
        volume_paths = root.glob('**/*.mat')

        if '2023' in str(self.root):
            self.year = 2023 
        elif '2024' in str(self.root):
            self.year = 2024
        else:
            raise ValueError('Invalid dataset root')
        #
        if self.year == 2023:
            # filter out files contains '_mask.mat'
            self.volume_paths = [str(path) for path in volume_paths if '_mask.mat' not in str(path)]
            
        elif self.year == 2024:
            self.volume_paths = [str(path) for path in volume_paths if '_mask_' not in str(path)]
        
        self.volume_paths = [pp for pp in self.volume_paths if raw_sample_filter(pp)]
        print('number of inference paths: ', len(self.volume_paths))
            

        self.transform = transform
        
        assert num_adj_slices % 2 == 1, "Number of adjacent slices must be odd."
        assert num_adj_slices <= 11, "Number of adjacent slices must be <= 11."
        self.num_adj_slices = num_adj_slices
        self.start_adj = -(num_adj_slices // 2)
        self.end_adj = num_adj_slices // 2 + 1
        self.volume_shape_dict = self._get_volume_shape_info()
        # add the fisrt element in each dict
        self.len_dataset = sum([v[0]*v[1] for v in self.volume_shape_dict.values()])
        
        
        self.current_volume = None
        self.current_file_index = -1
        self.current_num_slices = None
        self.slices_offset = 0  # Track the starting index of the slices in the current volume

        # New attributes
        self.index_to_volume_idx = {}
        self.index_to_slice_idx = {}
        self.volume_start_indices = []
        self.volume_indices = []  # Add this line

        self.current_volume = None
        self.current_volume_index = None

        self._build_index_mappings()

    def _build_index_mappings(self):
        global_idx = 0
        for volume_idx, path in enumerate(self.volume_paths):
            shape = self.volume_shape_dict[path]
            num_slices = shape[0] * shape[1]
            self.volume_start_indices.append(global_idx)

            volume_indices = []
            for slice_idx in range(num_slices):
                self.index_to_volume_idx[global_idx] = volume_idx
                self.index_to_slice_idx[global_idx] = slice_idx
                volume_indices.append(global_idx)
                global_idx += 1
            self.volume_indices.append(volume_indices)

        self.len_dataset = global_idx  # Update dataset length
        
    def _get_volume_shape_info(self):
        shape_dict = {} #defaultdict(dict)
        for path in self.volume_paths:
            shape_dict[path]=load_shape(path)
        return shape_dict
 
    def _get_ti_adj_idx_list(self, ti, num_t_in_volume):
        """
        Get circular adjacent indices for temporal axis.
        """
        if num_t_in_volume == 1:
            return [ti]
        start_lim, end_lim = -(num_t_in_volume // 2), (num_t_in_volume // 2 + 1)
        start, end = max(self.start_adj, start_lim), min(self.end_adj, end_lim)
        ti_idx_list = [(i + ti) % num_t_in_volume for i in range(start, end)]

        replication_prefix = max(start_lim - self.start_adj, 0) * ti_idx_list[0:1]
        replication_suffix = max(self.end_adj - end_lim, 0) * ti_idx_list[-1:]

        return replication_prefix + ti_idx_list + replication_suffix
    
    def _load_volume(self, path):
        """
        Load the k-space volume and mask for the given path.
        Modify this function based on your `load_kdata` and `load_mask` functions.
        """
        kspace_volume = load_kdata(path)
        kspace_volume = kspace_volume[None] if len(kspace_volume.shape) != 5 else kspace_volume # blackblood has no time dimension
        kspace_volume = kspace_volume.transpose(0, 1, 2, 4, 3)
        
        if self.year==2023:
            mask_path = path.replace('.mat', '_mask.mat')
            mask = load_mask(mask_path).T[0:1]
            mask=mask[None,:,:,None]
        elif self.year==2024:
            mask_path = path.replace('UnderSample_Task', 'Mask_Task').replace('_kus_', '_mask_')
            if 'UnderSample_Task1' in path:
                mask = load_mask(mask_path).T[0:1]
                mask=mask[None,:,:,None]
            else:
                mask = load_mask(mask_path).transpose(0,2,1)
                mask=mask[:,:,:,None]

        attrs = {
            'encoding_size': [kspace_volume.shape[3], kspace_volume.shape[4], 1],
            'padding_left': 0,
            'padding_right': kspace_volume.shape[-1],
            'recon_size': [kspace_volume.shape[3], kspace_volume.shape[4], 1],
        }
        return kspace_volume, mask, attrs
    
    def _load_next_volume(self):
        """Loads the next volume in the dataset."""
        self.current_file_index += 1
        if self.current_file_index < len(self.volume_paths):
            self.current_path = self.volume_paths[self.current_file_index]
            self.current_volume, self.mask, self.attrs = self._load_volume(self.current_path)  # Shape: (D, H, W)
            self.current_num_t = self.current_volume.shape[0]
            self.current_num_z = self.current_volume.shape[1]
            self.current_num_slices = self.current_num_t * self.current_num_z
            self.slices_offset += self.current_num_slices  # Update offset
        else:
            self.current_volume = None
            self.current_num_slices = None
            
    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        volume_idx = self.index_to_volume_idx[idx]
        slice_idx = self.index_to_slice_idx[idx]

        # Load the volume if not already loaded
        if self.current_volume_index != volume_idx:
            self._load_volume_by_index(volume_idx)

        # Compute temporal and spatial indices
        ti = slice_idx // self.current_num_z
        zi = slice_idx % self.current_num_z

        # Get temporal indices
        ti_idx_list = self._get_ti_adj_idx_list(ti, self.current_num_t)

        # Gather k-space data for adjacent slices
        nc = self.current_volume.shape[2]
        kspace = [self.current_volume[idx, zi] for idx in ti_idx_list]
        kspace = np.concatenate(kspace, axis=0)
        
        _path = self.current_path.replace(str(self.root)+'/', '')
        # gather mask data for adjacent slices
        if self.year==2023 or (self.year==2024 and 'UnderSample_Task1' in _path): 
            mask = self.mask
        else:
            mask = [self.mask[idx] for idx in ti_idx_list]
            mask = np.stack(mask, axis=0)
            mask = mask.repeat(nc, axis=0)

        # Prepare the sample
        if self.transform is None:
            sample = (kspace, mask, None, self.attrs, _path, slice_idx, self.current_num_t, self.current_num_z)
        else:
            sample = self.transform(kspace, mask, None, self.attrs, _path, slice_idx, self.current_num_t, self.current_num_z)

        return sample

    def _load_volume_by_index(self, volume_idx):
        self.current_volume_index = volume_idx
        self.current_path = self.volume_paths[volume_idx]
        self.current_volume, self.mask, self.attrs = self._load_volume(self.current_path)
        self.current_num_t = self.current_volume.shape[0]
        self.current_num_z = self.current_volume.shape[1]

