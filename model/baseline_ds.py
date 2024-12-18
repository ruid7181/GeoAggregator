import numpy as np
import geopandas as gpd
import torch

from pysal.lib import weights

from model.aggregator_ds import TabDataLoaderWrapper
from configurations.model_config import TabDataColumns


col_repository = TabDataColumns


def load_tab_as_df(ds, shuffle=False):
    wrapper = TabDataLoaderWrapper(
        ds=ds, shuffle=shuffle, batch_size=1, split_mode='all', get_sample_loc=False
    )
    tab_df = wrapper.get_original_data_df(norm=True).rename(columns={col_repository.pm25_y[0]: 'y'})

    return tab_df


def load_tab_as_graph(ds, shuffle=False):
    def _reNormalized_trick_laplacian_(mx_tilde):
        degree_tilde = np.diag(np.sum(mx_tilde, axis=1))
        D_tilde_inv_sqrt = np.linalg.inv(np.sqrt(degree_tilde))
        return np.dot(D_tilde_inv_sqrt, mx_tilde).dot(D_tilde_inv_sqrt)

    def _reNormalized_trick_mx_(mx):
        return mx + np.eye(mx.shape[0])

    wrapper = TabDataLoaderWrapper(
        ds=ds, shuffle=shuffle, batch_size=1,
        split_mode='all', get_sample_loc=False,
        sample_radius=1, sequence_len=1
    )

    tab_df = wrapper.get_original_data_df(norm=True)
    tab_df = gpd.GeoDataFrame(data=tab_df,
                              geometry=gpd.points_from_xy(tab_df[col_repository.pm25_spa[0]],
                                                          tab_df[col_repository.pm25_spa[1]]),
                              crs='EPSG:4326')   # seattle UTM 26910

    knn = weights.KNN.from_dataframe(tab_df, geom_col='geometry', k=9)
    A_20nn_sym = 1 * np.logical_or(knn.full()[0], knn.full()[0].T)
    A_tilde_20nn_sym = _reNormalized_trick_mx_(A_20nn_sym)
    Laplacian_knn = _reNormalized_trick_laplacian_(A_tilde_20nn_sym)
    adj = torch.FloatTensor(Laplacian_knn)

    x_tensor = torch.FloatTensor(
        tab_df[col_repository.pm25_atr + col_repository.pm25_spa].values.tolist()
    )
    y_tensor = torch.FloatTensor(tab_df[col_repository.pm25_y].values.tolist())

    idx_trn = torch.LongTensor(tab_df.index.values[: int(0.7 * len(tab_df))])
    idx_val = torch.LongTensor(tab_df.index.values[int(0.7 * len(tab_df)): int(0.8 * len(tab_df))])
    idx_tst = torch.LongTensor(tab_df.index.values[int(0.8 * len(tab_df)):])

    return x_tensor, y_tensor, adj, idx_trn, idx_val, idx_tst
