from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import os
import scipy.io as sio


class BlockSet(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        # return the database size
        return len(self.file_list)

    def __getitem__(self, idx):
        # return a single entity from database
        filename = self.file_list[idx]
        data = sio.loadmat(filename)
        data = data['data']
        ac, label = np.real(data[0][0][0]), np.real(data[0][0][1][0][0])
        return {'ac': ac.astype(np.float32), 'label': label.astype(np.float32)}


class FHSet(Dataset):
    def __init__(self, data_into, data_path, index=None):
        self.data_path = data_path
        self.file_list, self.label = self.extract_info(data_into)
        if index is not None:
            self.file_list = self.file_list[index]
            self.label = self.label[index]

    def __len__(self):
        return self.file_list.shape[0]

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        fh_mat = sio.loadmat(os.path.join(self.data_path, filename))
        fh = fh_mat['fh']

        return {'fh': fh.astype(np.float32), 'label': self.label[idx]}

    def extract_info(self, data_info, file_name_key='file_name', label_key='dmos_all'):
        mat_file = sio.loadmat(data_info)
        raw_file_list = mat_file[file_name_key]
        raw_dmos = mat_file[label_key]
        file_list, dmos = [], []
        for i in range(raw_file_list.shape[0]):
            file_list.append(raw_file_list[i][0][0])
            dmos.append(raw_dmos[i][0])

        dmos = np.array(dmos, dtype=np.float32)
        raw_max, raw_min = np.max(dmos), np.min(dmos)
        dmos = np.divide(dmos - raw_min, raw_max - raw_min).astype(np.float32)

        return np.array(file_list, np.str), dmos


if __name__ == '__main__':
    file_list = glob.glob(os.path.join('data', '*.mat'))

    dataset = BlockSet(file_list)
    loader = DataLoader(dataset, batch_size=2)

    for batch in loader:
        print(batch)
