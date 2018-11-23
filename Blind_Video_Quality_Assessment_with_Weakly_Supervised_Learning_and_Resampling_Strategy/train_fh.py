import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import glob
import os
import random
from data_processor import BlockSet, FHSet
from models import ArcNet, LinearRegression
from scipy.stats import spearmanr, pearsonr


class Trainer:
    def __init__(self, data_path, batch_size, train_ratio=0.8, trained_params=None, params_path=None):
        # define train info
        self.batch_size = batch_size
        self.max_epoch = 10000
        self.lr = 0.01
        self.global_step = 0
        self.val_interval = 10
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.params_path = params_path

        # output training info
        print('******************************')
        print('*initialization')
        print('*batch size {}'.format(self.batch_size))
        print('*max epoch {}'.format(self.max_epoch))
        print('*learning rate {}'.format(self.lr))
        print('*validation interval {}'.format(self.val_interval))
        print('*device {}'.format(self.device))
        print('*parameter path {}'.format(self.params_path))

        self.train_loss, self.val_loss = [], []
        if not os.path.exists(self.params_path):
            os.mkdir(self.params_path)

        # define model
        self.model = LinearRegression(100).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if trained_params is not None:
            self.model.load_state_dict(torch.load(trained_params))

        # get data loader
        self.train_loader, self.val_loader = self.get_train_test_dataloader(data_path, train_ratio)

        print('******************************')

    def get_train_test_dataloader(self, data_path, train_ratio, shuffle=True):
        """
        Get train and test data loader
        :param data_path: database path
        :param train_ratio: specific train data ratio
        :param shuffle: shuffle
        :return: train and test data loader
        """
        train_dataset = FHSet('LIVEVIDEOData.mat', 'fhs')
        test_dataset = FHSet('LIVEVIDEOData.mat', 'fhs')

        # output database info
        print('*train dataset size {}'.format(train_dataset.__len__()))
        print('*validation dataset size {}'.format(test_dataset.__len__()))

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=4)

        return train_loader, test_loader

    def calculate_loss(self, prediction, label):
        """
        Calculate mean square error loss
        :param prediction: model prediction
        :param label: ground truth label
        :return: mse loss
        """
        return F.mse_loss(prediction, label)

    def calculate_fit(self, prediction, label):
        pred = prediction.cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        srcc = spearmanr(pred, label)
        plcc = pearsonr(pred, label)
        return srcc[0], plcc[0]

    def saver(self):
        """
        Save model parameters and train val loss
        """
        torch.save(
            self.model.state_dict(),
            os.path.join(self.params_path, '{}-params.pkl'.format(self.global_step))
        )
        train_loss = np.array(self.train_loss, dtype=np.float32)
        train_loss.tofile('train_loss')
        val_loss = np.array(self.val_loss, dtype=np.float32)
        val_loss.tofile('val_loss')

    def run(self):
        """
        Run train and validate process
        """
        # initialize validation data loader
        val_iter = iter(self.val_loader)
        for i in range(self.max_epoch):
            for batch in self.train_loader:
                self.global_step += 1
                self.optim.zero_grad()

                y = self.model(batch['fh'].to(self.device))
                loss = self.calculate_loss(y, batch['label'].to(self.device))
                srcc, plcc = self.calculate_fit(y, batch['label'])

                loss.backward()
                self.optim.step()

                loss_val = loss.item()
                self.train_loss.append(loss_val)
                print('Epoch {}, step {}, loss {:.5f}, SRCC {:.5f}, PLCC {:.5f}'.format(
                    i, self.global_step, loss_val, srcc, plcc
                ))

                if self.global_step % self.val_interval == 0:
                    try:
                        val_batch = next(val_iter)
                    except StopIteration:
                        val_iter = iter(self.val_loader)
                        val_batch = next(val_iter)
                    val_y = self.model(val_batch['fh'].to(self.device))
                    loss = self.calculate_loss(val_y, val_batch['label'].to(self.device))
                    srcc, plcc = self.calculate_fit(val_y, val_batch['label'])

                    loss_val = loss.item()
                    self.val_loss.append(loss_val)

                    print('========================================')
                    print('validation loss {:.5f}, SRCC {:.5f}, PLCC {:.5f}'.format(loss_val, srcc, plcc))
                    print('========================================')

                    self.saver()
                    torch.cuda.empty_cache()


if __name__ == '__main__':
    data_path = r'fhs'
    batch_size = 160
    params_path = 'fh_params'
    trained_params = None

    trainer = Trainer(
        data_path=data_path,
        batch_size=batch_size,
        trained_params=trained_params,
        params_path=params_path)

    trainer.run()