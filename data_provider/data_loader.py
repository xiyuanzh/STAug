import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
from PyEMD import EMD
from tqdm import tqdm
import random

warnings.filterwarnings('ignore')

def emd_augment_scale(data, sequence_length, data_path):
    n_imf, channel_num = 100, data.shape[1]
    emd_data = np.zeros((n_imf,data.shape[0],channel_num))
    max_imf = 0
    for ci in tqdm(range(channel_num)):
        s = data[:, ci]
        IMF = EMD().emd(s)
        r_s = np.zeros((n_imf, data.shape[0]))
        if len(IMF) > max_imf:
            max_imf = len(IMF)
        for i in range(len(IMF)):
            r_s[i] = IMF[len(IMF)-1-i] #first trend, then seasonal pattern
        if(len(IMF)==0): r_s[0] = s
        emd_data[:,:,ci] = r_s
    if max_imf < n_imf:
        print('max_imf: ', max_imf)
        emd_data = emd_data[:max_imf,:,:]
    train_data_new = np.zeros((len(data)-sequence_length+1,max_imf,sequence_length,channel_num))
    for i in range(len(data)-sequence_length+1):
        train_data_new[i] = emd_data[:,i:i+sequence_length,:]
    return train_data_new

def emd_augment(train_data, sequence_length, data_path):

    channel_num, n_imf = train_data.shape[1], 100
    if not os.path.isfile('./dataset/emd_%s_%d_%d_%d.npy'%(data_path, len(train_data)-sequence_length+1,sequence_length,channel_num)):
        train_data_new = np.zeros((len(train_data)-sequence_length+1,n_imf,sequence_length,channel_num))
        max_imf = 0
        for sequencei in tqdm(range(len(train_data)-sequence_length+1)):
            for ci in range(channel_num):
                s = train_data[sequencei:sequencei+sequence_length,ci]
                IMF = EMD().emd(s)
                r_s = np.zeros((n_imf, sequence_length))
                if len(IMF) > max_imf:
                    max_imf = len(IMF)
                for i in range(len(IMF)):
                    r_s[i] = IMF[len(IMF)-1-i] #first trend, then seasonal pattern
                if(len(IMF)==0): r_s[0] = s
                train_data_new[sequencei,:,:,ci] = r_s
        if max_imf < n_imf:
            print('max_imf: ', max_imf)
            train_data_new = train_data_new[:,:max_imf,:,:]
        np.save('./dataset/emd_%s_%d_%d_%d.npy'%(data_path, len(train_data)-sequence_length+1,sequence_length,channel_num), train_data_new)
    else:
        train_data_new = np.load('./dataset/emd_%s_%d_%d_%d.npy'%(data_path, len(train_data)-sequence_length+1,sequence_length,channel_num))
    
    return train_data_new

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='ETTh1.csv', scale=True, timeenc=0, freq='h', aug='none', emd_scale=1):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.aug = aug
        self.emd_scale = emd_scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        if self.set_type == 0:
            if self.emd_scale == 1:
                self.aug_data = emd_augment_scale(data[:len(train_data)], self.seq_len+self.pred_len, self.data_path[:-4])
            else:
                self.aug_data = emd_augment(data[:len(train_data)], self.seq_len+self.pred_len, self.data_path[:-4])
        else:
            ### just a placeholder, no augmentation in validation and test phase
            self.aug_data = np.zeros_like(data[border1:border2])

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        aug_data = self.aug_data[s_begin]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark, aug_data
        

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='ETTm1.csv', scale=True, timeenc=0, freq='t', aug='none', emd_scale=1):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.aug = aug
        self.emd_scale = emd_scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        if self.set_type == 0:
            if self.emd_scale == 1:
                self.aug_data = emd_augment_scale(data[:len(train_data)], self.seq_len+self.pred_len, self.data_path[:-4])
            else:
                self.aug_data = emd_augment(data[:len(train_data)], self.seq_len+self.pred_len, self.data_path[:-4])
            
        else:
            ### just a placeholder, no augmentation in validation and test phase
            self.aug_data = np.zeros_like(data[border1:border2])

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        aug_data = self.aug_data[s_begin]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, aug_data

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='ETTh1.csv', scale=True, timeenc=0, freq='h', aug='none', emd_scale=1):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.aug = aug
        self.emd_scale = emd_scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        cols = list(df_raw.columns)
        cols.remove('OT')
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + ['OT']]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        if self.set_type == 0:
            if self.emd_scale == 1:
                self.aug_data = emd_augment_scale(data[:len(train_data)], self.seq_len+self.pred_len, self.data_path[:-4])
            else:
                self.aug_data = emd_augment(data[:len(train_data)], self.seq_len+self.pred_len, self.data_path[:-4])
        else:
            ### just a placeholder, no augmentation in validation and test phase
            self.aug_data = np.zeros_like(data[border1:border2])

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        aug_data = self.aug_data[s_begin]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, aug_data

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)