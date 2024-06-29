import os
import h5py
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor


def MultiTestSetDataLoader(args):
    # get testdataloader of every test dataset
    dataset_dir = args.path_for_test + 'SR_' + str(args.ang_res) + 'x' + str(args.ang_res) + '_' + \
                  str(args.scale_factor) + 'x/'
    data_list = os.listdir(dataset_dir)

    test_Loaders = []
    length_of_tests = 0
    for data_name in data_list:
        test_Dataset = TestSetDataLoader(args, data_name)
        length_of_tests += len(test_Dataset)

        test_Loaders.append(DataLoader(dataset=test_Dataset, num_workers=args.num_workers, batch_size=1, shuffle=False))

    return data_list, test_Loaders, length_of_tests

class TestSetDataLoader(Dataset):
    def __init__(self, args, data_name='ALL'):
        super(TestSetDataLoader, self).__init__()
        self.dataset_dir = args.path_for_test + 'SR_' + str(args.ang_res) + 'x' + str(args.ang_res) + '_' + \
                           str(args.scale_factor) + 'x/'
        self.data_list = [data_name]

        self.file_list = []
        for data_name in self.data_list:
            tmp_list = os.listdir(self.dataset_dir + data_name)
            for index, _ in enumerate(tmp_list):
                tmp_list[index] = data_name + '/' + tmp_list[index]

            self.file_list.extend(tmp_list)

        self.item_num = len(self.file_list)

    def __getitem__(self, index):
        file_name = [self.dataset_dir + self.file_list[index]]
        with h5py.File(file_name[0], 'r') as hf:
            Lr_SAI_y = np.array(hf.get('Lr_SAI_y'))
            Hr_SAI_y = np.array(hf.get('Hr_SAI_y'))
            # Lr_SAI_cb = np.array(hf.get('Lr_SAI_cb'))
            # Hr_SAI_cb = np.array(hf.get('Hr_SAI_cb'))
            # Lr_SAI_cr = np.array(hf.get('Lr_SAI_cr'))
            # Hr_SAI_cr = np.array(hf.get('Hr_SAI_cr'))

            Lr_SAI_y = np.transpose(Lr_SAI_y, (1, 0))
            Hr_SAI_y = np.transpose(Hr_SAI_y, (1, 0))
            # Lr_SAI_cb = np.transpose(Lr_SAI_cb, (1, 0))
            # Hr_SAI_cb = np.transpose(Hr_SAI_cb, (1, 0))
            # Lr_SAI_cr = np.transpose(Lr_SAI_cr, (1, 0))
            # Hr_SAI_cr = np.transpose(Hr_SAI_cr, (1, 0))

        Lr_SAI_y = ToTensor()(Lr_SAI_y.copy())
        Hr_SAI_y = ToTensor()(Hr_SAI_y.copy())
        # Lr_SAI_cb = ToTensor()(Lr_SAI_cb.copy())
        # Hr_SAI_cb = ToTensor()(Hr_SAI_cb.copy())
        # Lr_SAI_cr = ToTensor()(Lr_SAI_cr.copy())
        # Hr_SAI_cr = ToTensor()(Hr_SAI_cr.copy())

        return Lr_SAI_y, Hr_SAI_y  #, Lr_SAI_cb, Hr_SAI_cb, Lr_SAI_cr, Hr_SAI_cr

    def __len__(self):
        return self.item_num
