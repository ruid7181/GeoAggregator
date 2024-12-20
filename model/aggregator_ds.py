import os

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.neighbors import KDTree
import torch
from torch.utils.data import Dataset, DataLoader

from tools.utils import get_config
from configurations.model_config import TabDataColumns

config = get_config()


class TabDataSampler(Dataset):
    def __init__(self,
                 tab_data: pd.DataFrame,
                 mode='train',
                 sample_radius=0.014,
                 radius_estimate_ratio=0.5,
                 sequence_len=25,
                 atr_columns=None,
                 spa_columns=None,
                 y_columns=None,
                 get_sample_loc=False):
        self.mode = mode
        self.s_radius = sample_radius
        self.radius_estimate_ratio = radius_estimate_ratio
        self.seq_len = sequence_len
        self.atr_cols = atr_columns
        self.spa_cols = spa_columns
        self.y_cols = y_columns
        self.feat_cols = list(self.atr_cols) + list(self.spa_cols) + list(self.y_cols)
        self.get_sample_loc = get_sample_loc
        self.df_len = len(tab_data)

        # Train/val/test split. Default ratio is 7:1:2
        self.trn_n_sample = int(0.7 * self.df_len)
        self.val_n_sample = int(0.1 * self.df_len)
        self.tst_n_sample = int(0.2 * self.df_len)

        tab_data = tab_data.sample(frac=1.0, random_state=41)
        if not self.mode == 'all':
            self.trn_pool = tab_data.iloc[: self.trn_n_sample]
            self.query_tree = KDTree(self.trn_pool[self.spa_cols])

        if self.mode == 'train':
            self.query_pool = self.trn_pool
        elif self.mode == 'val':
            self.query_pool = tab_data.iloc[self.trn_n_sample:].iloc[: self.val_n_sample]
        elif self.mode == 'test':
            self.query_pool = tab_data.iloc[self.trn_n_sample + self.val_n_sample:]
        elif self.mode == 'all':
            self.query_pool = tab_data
            self.trn_pool = tab_data
            self.query_tree = KDTree(self.trn_pool[self.spa_cols])

        self.trn_pool_data = torch.FloatTensor(self.trn_pool[self.feat_cols].to_numpy())

        if self.s_radius is None:
            print(f'Estimating the searching radius for ContextQuery.')
            self.s_radius = self.__estimate_radius(
                all_points=tab_data[self.spa_cols],
                seq_len=self.seq_len,
                sample_ratio=self.radius_estimate_ratio
            )
            print(f'Estimated radius: {self.s_radius}')

    def __getitem__(self, item):
        """
        :return: input sequence (dists) and target element position.
        """
        indices, dists = self.query_tree.query_radius(
            X=self.query_pool[self.spa_cols][item: item + 1].to_numpy(),
            r=self.s_radius,
            return_distance=True
        )
        indices = indices[0]
        dists = dists[0]
        n_neighbor = indices.shape[0]

        if self.mode == 'all' or self.mode == 'train':
            dists = dists[indices != item]
            indices = indices[indices != item]
            n_neighbor -= 1

        query_point = torch.FloatTensor(self.query_pool[self.feat_cols][item: item + 1].to_numpy())

        if n_neighbor <= self.seq_len:
            sample = torch.zeros(size=(self.seq_len, len(self.feat_cols)), dtype=torch.float) + torch.nan
            sample[:n_neighbor] = self.trn_pool_data[indices]
            sample[-1] = query_point

            sample_dist = torch.zeros(size=(self.seq_len,), dtype=torch.float) + torch.nan
            sample_dist[:n_neighbor] = torch.FloatTensor(dists)
            sample_dist = torch.where(sample_dist == 0, 1e-3, sample_dist)
            sample_dist[-1] = 0.
        else:
            indices = shuffle(indices, random_state=item)[:self.seq_len]
            sample = self.trn_pool_data[indices]
            sample[-1] = query_point

            sample_dist = shuffle(dists, random_state=item)[:self.seq_len]
            sample_dist = torch.FloatTensor(sample_dist)
            sample_dist = torch.where(sample_dist == 0, 1e-3, sample_dist)
            sample_dist[-1] = 0.

        if self.get_sample_loc:
            return (sample, sample_dist), item
        else:
            return sample, sample_dist

    def __len__(self):
        return len(self.query_pool)

    def __count_avg_neighbors(self, all_points, radius, sample_size):
        tree = self.query_tree
        N = all_points.shape[0]
        if sample_size < N:
            idx = np.random.choice(N, sample_size, replace=False)
            all_points = all_points[idx]

        neighbors_idx = tree.query_radius(all_points, radius)
        counts = [len(nbrs) - 1 for nbrs in neighbors_idx]

        return np.mean(counts)

    def __estimate_radius(self,
                          all_points: pd.DataFrame,
                          seq_len=81,
                          sample_ratio=0.3,
                          max_iter=30,
                          tolerance=0.02):
        """
        Find a radius such that the average number of neighbors is approximately 'seq_len'.
        A binary search approach.
        :param all_points: a df containing only 'spa' columns (spatial coordinates).
                           Typically, a df get from 'get_original_data_df(norm=True)[spa_cols]'
        :param seq_len: expected sequence length.
        :param sample_ratio: sample part of the data set for estimation.
        :param max_iter: maximum number of iterations.
        :param tolerance: stopping criteria.
        """
        sample_size = int(sample_ratio * len(all_points))
        all_points = all_points.values

        left, right = 1e-6, 1
        for i in range(max_iter):
            mid = (left + right) / 2.
            avg_seq_len = self.__count_avg_neighbors(all_points=all_points,
                                                     radius=mid,
                                                     sample_size=sample_size)

            if abs(avg_seq_len - seq_len) <= tolerance:
                print(f'Radius estimation ends after {i} iterations.')
                return mid

            if avg_seq_len > seq_len:
                right = mid
            else:
                left = mid

        print(f'Radius estimation ends after {max_iter} iterations.')
        return (left + right) / 2.


class TabDataLoaderWrapper:
    def __init__(self,
                 ds='housing',
                 shuffle=False,
                 batch_size=8,
                 sample_radius=None,
                 radius_estimate_ratio=0.5,
                 sequence_len=25,
                 split_mode='train',
                 get_sample_loc=False):
        self.ds = ds
        self.ds_path = None
        self.to_shuffle = shuffle
        self.batch_size = batch_size
        self.sample_radius = sample_radius
        self.radius_estimate_ratio = radius_estimate_ratio
        self.sequence_len = sequence_len
        self.split_mode = split_mode
        self.get_sample_loc = get_sample_loc

        self.tabDS = None
        self.col_repository = TabDataColumns

        if ds == 'housing':
            atr_cols, spa_cols, y_cols = self.col_repository.housing_atr, self.col_repository.housing_spa, self.col_repository.housing_y
            self.atr_dims = tuple(range(0, len(self.col_repository.housing_atr)))
            self.spa_dims = tuple(
                range(len(atr_cols), len(atr_cols) + len(spa_cols))
            )
            self.target_dims = (-1,)
            self.ds_path = os.path.join(config.local_data_tabular_dir, 'seattle_house_price_ds.csv')
            self.tab_df = pd.read_csv(self.ds_path)
            self.tab_df = self.tab_df.sample(frac=1., random_state=41)
            self.tab_df[spa_cols] = self.tab_df[spa_cols].apply(
                lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
            )
            self.tab_df[y_cols] = self.tab_df[y_cols].apply(
                lambda x: 10 ** x / 1e5
            )
        elif ds == 'pm25':
            atr_cols, spa_cols, y_cols = self.col_repository.pm25_atr, self.col_repository.pm25_spa, self.col_repository.pm25_y
            self.atr_dims = tuple(range(0, len(atr_cols)))
            self.spa_dims = tuple(
                range(len(atr_cols), len(atr_cols) + len(spa_cols))
            )
            self.target_dims = (-1,)
            self.ds_path = os.path.join(config.local_data_tabular_dir, 'pm25_data_scaled.csv')
            self.tab_df = pd.read_csv(self.ds_path)
            self.tab_df = self.tab_df[atr_cols + spa_cols + y_cols]
            self.tab_df[spa_cols] = self.tab_df[spa_cols].apply(
                lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
            )
        elif 'syn' in ds:
            atr_cols, spa_cols, y_cols = self.col_repository.syn_atr, self.col_repository.syn_spa, self.col_repository.syn_y
            self.atr_dims = tuple(range(0, len(atr_cols)))
            self.spa_dims = tuple(
                range(len(atr_cols), len(atr_cols) + len(spa_cols))
            )
            self.target_dims = (-1,)
            self.ds_path = os.path.join(config.local_data_tabular_dir, f'{ds}_ds.csv')
            self.tab_df = pd.read_csv(self.ds_path)
            self.tab_df = self.tab_df.sample(frac=1., random_state=42)
        elif ds == 'sdoh':
            atr_cols, spa_cols, y_cols = self.col_repository.sdoh_atr, self.col_repository.sdoh_spa, self.col_repository.sdoh_y
            self.atr_dims = tuple(range(0, len(atr_cols)))
            self.spa_dims = tuple(
                range(len(atr_cols), len(atr_cols) + len(spa_cols))
            )
            self.target_dims = (-1,)
            self.ds_path = os.path.join(config.local_data_tabular_dir, 'us_sdoh_2014.csv')
            self.tab_df = pd.read_csv(self.ds_path)
            self.tab_df = self.tab_df[atr_cols + spa_cols + y_cols]
            self.tab_df[spa_cols] = self.tab_df[spa_cols].apply(
                lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
            )
        else:
            raise TypeError(f'Illegal dataset \'{ds}\'.')

        self.tabDS = TabDataSampler(
            tab_data=self.tab_df,
            atr_columns=atr_cols,
            spa_columns=spa_cols,
            y_columns=y_cols,
            get_sample_loc=self.get_sample_loc,
            mode=self.split_mode,
            sample_radius=self.sample_radius,
            sequence_len=self.sequence_len,
            radius_estimate_ratio=radius_estimate_ratio
        )

    def get_dataloader(self):
        return DataLoader(
            dataset=self.tabDS,
            batch_size=self.batch_size,
            shuffle=self.to_shuffle
        )

    def get_dataiter(self):
        return self.tabDS

    def get_original_data_df(self, norm=False):
        if norm:
            return self.tab_df
        else:
            return pd.read_csv(self.ds_path)
