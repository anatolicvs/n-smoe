import pathlib
from data.subsample import RandomMaskFunc
from data.transforms import to_tensor, apply_mask
from data.dataset_mri import SliceDataset
import matplotlib.pyplot as plt


mask_func = RandomMaskFunc(
    center_fractions=[0.08, 0.04],
    accelerations=[4, 8]
)

def data_transform(kspace, mask, target, data_attributes, filename, slice_num):
    # Transform the data into appropriate format
    # Here we simply mask the k-space and return the result
    kspace = to_tensor(kspace)
    target = to_tensor(target)
    # masked_kspace,_,_ = apply_mask(kspace, mask_func)
    return kspace,target

dataset = SliceDataset(
    root=pathlib.Path('/mnt/d/Medical/multicoil_train'),
    transform=data_transform,
    challenge='multicoil'
)

for kspace,target in dataset:

    kspace_np = kspace.cpu().numpy()
    
    pass
