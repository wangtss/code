import scipy.io as sio
import numpy as np
import os
import glob
import shutil
import random
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from models import ArcNet
from data_processor import BlockSet

class Utils:
    def display_train_val_loss(self, train_loss, val_loss, val_interval):
        """
        Plot train and val loss
        :param train_loss:
        :param val_loss:
        :param val_interval:
        :return:
        """
        if isinstance(train_loss, str):
            train_loss = np.fromfile(train_loss, dtype=np.float32)
            train_loss = train_loss[50:]
            val_loss = np.fromfile(val_loss, dtype=np.float32)

        datapoint_num = train_loss.shape[0]

        train_x = np.arange(0, datapoint_num, 1)
        val_x = np.arange(0, datapoint_num, val_interval)

        plt.plot(train_x, train_loss, color='b', label='train loss')
        plt.plot(val_x, val_loss, 'r--', label='val loss')
        plt.legend()

        plt.show()

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

    def block_filter(self, data_path, data_info, new_path='filtered_block'):
        mat = sio.loadmat(data_info)
        ref_name_list, dmos = mat['ref_name'][0], mat['dmos_all']
        group_size, group_num = [4, 3, 4, 4], 4
        if not os.path.exists(new_path):
            os.mkdir(new_path)

        self.count, self.total = 0, 0

        e_num = lambda x1, x2: pre + str(sum(group_size[:x1]) + 2 + x2)
        e_label = lambda x1, x2: sum(group_size[:x1]) + 1 + x2

        def check(pre, h, w, t):
            name = os.path.join(data_path, '{}_{}_{}_{}_{}.mat'.format(pre, frame_rate, h, w, t))
            return os.path.exists(name)
        def rank_check(block_label, video_label):
            new_block_label, new_video_label = [], []
            for i in range(block_label.shape[0]):
                new_block_label.append([i, block_label[i]])
                new_video_label.append([i, video_label[i]])
            new_video_label.sort(key=lambda x: x[1])
            new_block_label.sort(key=lambda x: x[1])
            # print(new_video_label, new_block_label)
            for i in range(block_label.shape[0]):
                if new_block_label[i][0] != new_video_label[i][0]:
                    return False
            return True
        def std_check(block_label, video_label):
            block_std, video_std = np.std(block_label), np.std(video_label)
            return np.abs(block_std - video_std) <= 0.02

        def verify_block(name_list, video_label, frame_rate, h, w, t):
            block_name, block_label = [], []
            for name in name_list:
                block = os.path.join(data_path, '{}_{}_{}_{}_{}.mat'.format(name, frame_rate, h, w, t))
                block_name.append(block)
                data = sio.loadmat(block)
                data = data['data']
                block_label.append(data[0][0][1][0][0])
            # print(block_name)
            block_label, video_label = np.array(block_label, np.float32), np.array(video_label, np.float32)
            video_label = 1 - np.divide(video_label, 100)
            if rank_check(block_label, video_label) and std_check(block_label, video_label):
                for name in block_name:
                    new_name = os.path.join(new_path, os.path.basename(name))
                    shutil.copyfile(name, new_name)
                self.count += block_label.shape[0]
            self.total += block_label.shape[0]
            print('Total {}, real {}'.format(self.total, self.count), end='\r')

        def filter(name_list, label_list, frame_rate):
            h, w, t = 1, 1, 1
            stride = [48, 48, 3]
            while True:
                if not check(name_list[0], h, 1, 1):
                    break
                while True:
                    if not check(name_list[0], 1, w, 1):
                        w = 1
                        break
                    while True:
                        if not check(name_list[0], 1, 1, t):
                            t = 1
                            break
                        # print(name_list)
                        verify_block(name_list, label_list, frame_rate, h, w, t)
                        t += stride[2]
                    w += stride[1]
                h += stride[0]

        for index_ref in range(ref_name_list.shape[0]):
            parts = ref_name_list[index_ref][0].split('_')
            pre, frame_rate = parts[0][:-1], parts[1][:-4]
            for ig in range(group_num):
                group_name_list = [e_num(ig, i) for i in range(group_size[ig])]
                label_list = [dmos[e_label(ig, i)][0] for i in range(group_size[ig])]

                filter(group_name_list, label_list, frame_rate)

    def inference(self, trained_params, data_info, data_path, output_path):
        """
        Predict block score using trained model
        :param trained_params:
        :param data_info:
        :return:
        """
        # get database file list
        file_list, _ = self.extract_info(data_info)
        file_num = file_list.shape[0]

        # define model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = ArcNet().to(device)
        model.load_state_dict(torch.load(trained_params))
        model.eval()
        batch_size = 1024

        if not os.path.exists(output_path):
            os.mkdir(output_path)

        def get_dataloader(video_filename):
            pattern = video_filename.split('_')[0] + '_'
            block_file_list = glob.glob(os.path.join(data_path, '{}*.mat'.format(pattern)))
            random.shuffle(block_file_list)
            block_file_list = block_file_list[:int(round(0.8 * len(block_file_list)))]
            dataset = BlockSet(block_file_list)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
            return loader

        def compute_frequency_histogram(vec):
            vec = np.concatenate(vec, axis=0).astype(np.float32)
            his = np.histogram(vec, bins=13)
            return his[0]
        print('Total video num {}'.format(file_num))
        for i in range(file_num):
            filename = file_list[i]
            print('******************************')
            print('Compute fh for {}..'.format(filename))
            loader = get_dataloader(filename)
            pred = []
            count = 0
            for batch in loader:
                output = model(batch['ac'].to(device))
                pred.append(output.cpu().detach().numpy())
                count += batch_size
                print('Generate block score {}'.format(count), end='\r')

            fh = compute_frequency_histogram(pred)
            print('\nSave fh..')
            save_name = os.path.join(output_path, filename)
            sio.savemat(save_name, {'fh': fh})
            print('******************************')
        print('Complete!')



if __name__ == '__main__':
    utils = Utils()

    # utils.block_filter(data_path='data', data_info='LIVEVIDEOData.mat')

    # utils.display_train_val_loss('train_loss', 'val_loss', 100)

    utils.inference('params/3500-params.pkl', 'LIVEVIDEOData.mat', 'data', 'fhs')