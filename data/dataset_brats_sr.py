import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import os
import os.path
import nibabel
from torch.utils.data import DataLoader


# def __getitem__(self, x):
    #     filedict = self.database[x]
    #     nib_img = nibabel.load(filedict['t1n'])  # We only use t1 weighted images
    #     out = nib_img.get_fdata()

    #     # CLip and normalize the images
    #     out_clipped = np.clip(out, np.quantile(out, 0.001), np.quantile(out, 0.999))
    #     out_normalized = (out_clipped - np.min(out_clipped)) / (np.max(out_clipped) - np.min(out_clipped))
    #     out = torch.tensor(out_normalized)

    #     # Zero pad images
    #     image = torch.zeros(1, 256, 256, 256)
    #     image[:, 8:-8, 8:-8, 50:-51] = out

    #     # Downsampling
    #     if self.img_size == 128:
    #         downsample = nn.AvgPool3d(kernel_size=2, stride=2)
    #         image = downsample(image)

    #     # Normalization
    #     image = self.normalize(image)

    #     return image


class BRATSVolumes(torch.utils.data.Dataset):
    def __init__(self, directory, test_flag=False, normalize=None, mode='train', img_size=256):
        super().__init__()
        self.mode = mode
        self.directory = os.path.expanduser(directory)
        self.normalize = normalize or (lambda x: x)
        self.test_flag = test_flag
        self.img_size = img_size
        self.seqtypes = ['t1n', 't1c', 't2w', 't2f'] if test_flag else ['t1n', 't1c', 't2w', 't2f', 'seg']
        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        for root, dirs, files in os.walk(self.directory):
            if not dirs:
                files.sort()
                datapoint = {}
                for f in files:
                    seqtype = f.split('-')[4].split('.')[0]
                    datapoint[seqtype] = os.path.join(root, f)
                if len(datapoint) == len(self.seqtypes):
                    self.database.append(datapoint)

    def __getitem__(self, index, get_full_volume=False):
        filedict = self.database[index]
        nib_img = nibabel.load(filedict['t1n'])
        volume = nib_img.get_fdata()

        if get_full_volume:
            return volume

        # Use get_best_slice to find the most informative slice
        best_slice_index = self.get_best_slice(volume)
        slice_data = volume[:, :, best_slice_index]

        slice_clipped = np.clip(slice_data, np.quantile(slice_data, 0.001), np.quantile(slice_data, 0.999))
        slice_normalized = (slice_clipped - np.min(slice_clipped)) / (np.max(slice_clipped) - np.min(slice_clipped))
        slice_tensor = torch.tensor(slice_normalized, dtype=torch.float32)

        image = torch.zeros(1, self.img_size, self.img_size)
        image[0, :, :] = slice_tensor
        image = self.normalize(image)

        return image

    def get_best_slice(self, volume):
        variances = [np.var(volume[:, :, i]) for i in range(volume.shape[2])]
        return np.argmax(variances)

    def __len__(self):
        return len(self.database)




class NewDataset(torch.utils.data.Dataset):
    def __init__(self, directory, options):
        self.directory = directory
        self.options = options
        self.transform = self.options.get('transform', lambda x: x)

        
        if self.contains_h5_files(directory):
            self.init_slice_sr()
        else:
            self.init_brats()

    def contains_h5_files(self, directory):
   
        for subdir, dirs, files in os.walk(directory):
            print("Checking in:", subdir)  # Debug print to show where the search is happening
            for file in files:
                print("Found file:", file)  # Debug print to show files being checked
                if file.endswith('.h5'):
                    return True
        return False


    def init_brats(self):
        self.img_size = self.options.get('img_size', 256)
        self.seqtypes = ['t1n', 't1c', 't2w', 't2f', 'seg']
        self.database = []
        for root, dirs, files in os.walk(self.directory):
            if not dirs:
                files.sort()
                datapoint = {seq: None for seq in self.seqtypes}
                for file in files:
                    seqtype = file.split('-')[4].split('.')[0]
                    if seqtype in datapoint:
                        datapoint[seqtype] = os.path.join(root, file)
                if all(datapoint.values()):
                    self.database.append(datapoint)

    def init_slice_sr(self):
        # List .h5 files across the directory
        self.h5_files = [os.path.join(dirpath, file)
                         for dirpath, dirnames, files in os.walk(self.directory)
                         for file in files if file.endswith('.h5')]

    def __getitem__(self, index):
        if hasattr(self, 'database'):
            return self.get_brats_item(index)
        elif hasattr(self, 'h5_files'):
            return self.get_slice_sr_item(index)

    def get_brats_item(self, index):
        filedict = self.database[index]
        nib_img = nibabel.load(filedict['t1n'])
        volume = nib_img.get_fdata()
        best_slice_index = np.argmax([np.var(volume[:, :, i]) for i in range(volume.shape[2])])
        slice_data = volume[:, :, best_slice_index]
        return torch.tensor(slice_data, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

    def get_slice_sr_item(self, index):
        file_name = self.h5_files[index]
        with h5py.File(file_name, 'r') as hf:
            img_reconstructed = hf['reconstruction_rss'][()]
            img_normalized = (img_reconstructed - np.min(img_reconstructed)) / (np.max(img_reconstructed) - np.min(img_reconstructed))
            img_tensor = torch.tensor(img_normalized, dtype=torch.float32).unsqueeze(0)  # Ensure it has a channel dimension
            if self.transform:
                img_tensor = self.transform(img_tensor)
            return img_tensor

    def __len__(self):
        if hasattr(self, 'database'):
            return len(self.database)
        elif hasattr(self, 'h5_files'):
            return len(self.h5_files)

# class BRATSVolumes(torch.utils.data.Dataset):
#     def __init__(self, directory, test_flag=False, normalize=None, mode='train', img_size=256):
#         '''
#         directory is expected to contain some folder structure:
#                   if some subfolder contains only files, all of these
#                   files are assumed to have a name like
#                   brats_train_NNN_XXX_123_w.nii.gz
#                   where XXX is one of t1n, t1c, t2w, t2f, seg
#                   we assume these five files belong to the same image
#                   seg is supposed to contain the segmentation
#         '''
#         super().__init__()
#         self.mode = mode
#         self.directory = os.path.expanduser(directory)
#         self.normalize = normalize or (lambda x: x)
#         self.test_flag = test_flag
#         self.img_size = img_size
#         if test_flag:
#             self.seqtypes = ['t1n', 't1c', 't2w', 't2f']
#         else:
#             self.seqtypes = ['t1n', 't1c', 't2w', 't2f', 'seg']
#         self.seqtypes_set = set(self.seqtypes)
#         self.database = []

#         for root, dirs, files in os.walk(self.directory):
#             # if there are no subdirs, we have a datadir
#             if not dirs:
#                 files.sort()
#                 datapoint = dict()
#                 # extract all files as channels
#                 for f in files:
#                     seqtype = f.split('-')[4].split('.')[0]
#                     datapoint[seqtype] = os.path.join(root, f)
#                 self.database.append(datapoint)

#     def __getitem__(self, index, get_full_volume=False):
#         filedict = self.database[index]
#         nib_img = nibabel.load(filedict['t1n'])  # assuming t1n is representative for all slices
#         volume = nib_img.get_fdata()

#         if get_full_volume:
#             return volume

        
#         middle_slice_index = volume.shape[2] // 2
#         slice_data = volume[:, :, middle_slice_index]

        
#         slice_clipped = np.clip(slice_data, np.quantile(slice_data, 0.001), np.quantile(slice_data, 0.999))
#         slice_normalized = (slice_clipped - np.min(slice_clipped)) / (np.max(slice_clipped) - np.min(slice_clipped))
#         slice_tensor = torch.tensor(slice_normalized, dtype=torch.float32)

        
#         image = torch.zeros(1, self.img_size, self.img_size)
#         image[0, :, :] = slice_tensor
#         image = self.normalize(image)

#         return image

#     def __len__(self):
#         return len(self.database)
    

if __name__ == '__main__':
   
#    data_dir = '/mnt/e/Medical/mri/data_for_train/'
#    renormalize = False
#    image_size = 240
#    ds = BRATSVolumes(directory=data_dir, test_flag=False,
#                           normalize=(lambda x: 2*x - 1) if renormalize else None,
#                           mode='train',
#                           img_size=image_size)

    
    dataroot = '/mnt/e/Medical/mri/data_for_train/'
    options = {'img_size': 256, 'transform': None}
    ds = NewDataset(dataroot, options)

    loader = torch.utils.data.DataLoader(ds, batch_size=10, shuffle=True)

    dataloader = DataLoader(ds, batch_size=1, shuffle=True)

    for data in dataloader:
       print(data.shape)
   

   