import os
import itertools

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F

from tools.utils import get_config
from configurations.model_config import TabDataColumns

config = get_config()

def calc_adjacency_map(order=1, h=50, w=50):
    """Calculate W matrix with Adj matrix"""
    from libpysal.weights import lat2W

    W = torch.zeros(size=(h**2, w**2))
    pysal_w = lat2W(nrows=h, ncols=w, rook=False).neighbors
    for k in pysal_w.keys():
        for v in pysal_w[k]:
            W[k, v] = 1

    return W / W.sum(dim=1, keepdim=True)


class SynGenerator:
    def __init__(self,
                 h=50,
                 w=50,
                 noise_cancel_level=1,
                 x_type='dem',
                 pipeline_option='produce'):
        self.GLOBAL_NOISE_CANCEL_LEVEL = noise_cancel_level
        self.h, self.w = h, w
        self.theta, self.rho = 0.8, 0.8
        self.pipeline_option = pipeline_option
        self.x_type = x_type

        print(f'[Notice]:\tUsing X type: \'{self.x_type}\'.')

        coord_mat = torch.FloatTensor(list(itertools.product(
            torch.arange(-self.h // 2, self.w // 2, 1),
            torch.arange(-self.h // 2, self.w // 2, 1)
        ))).transpose(1, 0).reshape(2, self.h, self.w)
        coord_mean, coord_std = coord_mat.mean(dim=(1, 2), keepdim=True), coord_mat.std(dim=(1, 2), keepdim=True)
        self.coord_mat = (coord_mat - coord_mean) / coord_std

    def _load_gwr_beta_mat_(self):
        beta_df = pd.read_csv(os.path.join(config.local_data_tabular_dir, 'synthetic_data_2-variables.csv'))
        cols = ['b0', 'b1', 'b2']
        beta_mat = torch.FloatTensor(beta_df[cols].to_numpy())   # [625, 5]
        beta_mat = beta_mat.T.contiguous().view(3, self.h // 2, self.w // 2)
        beta_mat = F.interpolate(beta_mat.unsqueeze(0), size=(self.h, self.w), mode='bilinear').squeeze()

        return beta_mat

    def _load_linear_beta_mat_(self):
        beta_mat = torch.ones(size=(3, self.h, self.w))
        beta_mat[0] *= 3
        beta_mat[1] *= 1
        beta_mat[2] *= 2

        return beta_mat

    def _load_x_mat_(self):
        if self.x_type == 'dem':
            dem = cv2.imread(os.path.join(config.local_data_raster_syn_dir, 'SRTM_n38_w119_1arc_v2_50by50.tif'), -1)
            dem = dem.transpose(2, 0, 1)
            dem = torch.FloatTensor(dem)[:2]
            x_mat = (dem - dem.min()) / (dem.max() - dem.min())
            x_mat = 3 * x_mat - 1.5
        elif self.x_type == 'rand':
            x_mat = torch.rand(size=(2, self.h, self.w)) * 3 - 1.5
        else:
            raise TypeError('Unknown x mat type.')

        return x_mat

    def _load_noise_term_(self):
        return torch.randn((self.h, self.w)) / self.GLOBAL_NOISE_CANCEL_LEVEL

    def get_spatial_lag(self):
        if self.pipeline_option == 'produce':
            x_mat = self._load_x_mat_()
            noise_mat = self._load_noise_term_()
            beta_mat = self._load_gwr_beta_mat_()

            res_mat = torch.concat((
                x_mat[:2],                                # x1, x2
                self.coord_mat,                         # x_coord, y_coord
                torch.zeros(size=(1, self.h, self.w))   # y
            ))

            res_mat[-1] += beta_mat[0]                            # add beta0
            res_mat[-1] += torch.mul(beta_mat[1], res_mat[0])     # add beta1 * X1
            res_mat[-1] += torch.mul(beta_mat[2], res_mat[1])     # add beta2 * X2
            res_mat[-1] += noise_mat                              # add noise

            W = calc_adjacency_map(h=self.h, w=self.w)
            temp = torch.linalg.inv(torch.eye(self.h * self.w) - self.rho * W)
            res_mat[-1] = torch.matmul(
                temp,
                res_mat[-1].view(1, self.h * self.w).transpose(1, 0)
            ).view(self.h, self.w)

            noise_star_mat = torch.matmul(
                temp,
                noise_mat.reshape(1, self.h * self.w).transpose(1, 0)
            ).view(self.h, self.w)

            res_mat = (res_mat.numpy() * 5e3).astype(np.int32)
            noise_star_mat = (noise_star_mat.numpy() * 5e3).astype(np.int32)

            for bid, band in enumerate(res_mat):
                cv2.imwrite(
                    os.path.join(config.local_data_raster_syn_dir, f'sl_ds_{bid}_{self.x_type}.tif'), band
                )
            cv2.imwrite(
                os.path.join(config.local_data_raster_syn_dir, f'sl_noise-term_{self.x_type}.tif'), noise_star_mat
            )

        elif self.pipeline_option == 'load':
            print('[Notice]:\tloading dataset \'SL\'')
            data_mat = torch.zeros(size=(5, self.h, self.w))
            for b in range(5):   # 5 bands
                temp = cv2.imread(
                    os.path.join(config.local_data_raster_syn_dir, f'sl_ds_{b}_{self.x_type}.tif'), -1
                )
                data_mat[b, :, :] = torch.FloatTensor(
                    temp
                ) / 5e3

            return data_mat

    def get_spatial_lagged_x(self):
        if self.pipeline_option == 'produce':
            x_mat = self._load_x_mat_()
            noise_mat = self._load_noise_term_()
            beta_mat = self._load_linear_beta_mat_()
            res_mat = torch.concat((
                x_mat[:2],                              # x1, x2
                self.coord_mat,                         # x_coord, y_coord
                torch.zeros(size=(1, self.h, self.w))   # y
            ))

            res_mat[-1] += beta_mat[0]                          # add beta0
            res_mat[-1] += torch.mul(beta_mat[1], res_mat[0])   # add beta1 * X1
            res_mat[-1] += torch.mul(beta_mat[2], res_mat[1])   # add beta2 * X2
            res_mat[-1] += noise_mat                            # add noise

            W = calc_adjacency_map(order=1, h=self.h, w=self.w)
            res_mat[-1] += self.theta * torch.matmul(
                W,
                res_mat[0].view(1, self.h * self.w).transpose(1, 0)
            ).view(self.h, self.w)
            res_mat[-1] += self.theta * torch.matmul(
                W,
                res_mat[1].view(1, self.h * self.w).transpose(1, 0)
            ).view(self.h, self.w)

            res_mat = (res_mat.numpy() * 5e3).astype(np.int32)
            noise_star_mat = (noise_mat.numpy() * 5e3).astype(np.int32)

            for bid, band in enumerate(res_mat):
                cv2.imwrite(
                    os.path.join(config.local_data_raster_syn_dir, f'slx_ds_{bid}_{self.x_type}.tif'), band
                )
            cv2.imwrite(
                os.path.join(config.local_data_raster_syn_dir, f'slx_noise-term_{self.x_type}.tif'), noise_star_mat
            )

        elif self.pipeline_option == 'load':
            print('[Notice]:\tloading dataset \'SLX\'')
            data_mat = torch.zeros(size=(5, self.h, self.w))
            for b in range(5):   # 5 bands
                temp = cv2.imread(
                    os.path.join(config.local_data_raster_syn_dir, f'slx_ds_{b}_{self.x_type}.tif'), -1
                )
                data_mat[b, :, :] = torch.FloatTensor(
                    temp
                ) / 5e3

            return data_mat

    def get_gwr(self):
        if self.pipeline_option == 'produce':
            noise_mat = self._load_noise_term_()
            beta_mat = self._load_gwr_beta_mat_()
            x_mat = self._load_x_mat_()

            res_mat = torch.concat((
                x_mat,                                  # x1, x2
                self.coord_mat,                         # x_coord, y_coord
                torch.zeros(size=(1, self.h, self.w))   # y
            ))
            res_mat[-1] += beta_mat[0]                          # add beta0
            res_mat[-1] += torch.mul(beta_mat[1], res_mat[0])   # add beta1 * X1
            res_mat[-1] += torch.mul(beta_mat[2], res_mat[1])   # add beta2 * X2
            res_mat[-1] += noise_mat                            # add noise
            noise_star_mat = noise_mat                          # save real noise for true MAE

            res_mat = (res_mat.numpy() * 5e3).astype(np.int32)
            noise_star_mat = (noise_star_mat.numpy() * 5e3).astype(np.int32)

            for bid, band in enumerate(res_mat):
                cv2.imwrite(os.path.join(config.local_data_raster_syn_dir, f'gwr_ds_{bid}_{self.x_type}.tif'), band)
            cv2.imwrite(os.path.join(config.local_data_raster_syn_dir, f'gwr_noise-term_{self.x_type}.tif'), noise_star_mat)

        elif self.pipeline_option == 'load':
            print('[Notice]:\tloading dataset \'GWR\'')
            data_mat = torch.zeros(size=(5, self.h, self.w))
            for b in range(5):   # 5 bands
                temp = cv2.imread(
                    os.path.join(config.local_data_raster_syn_dir, f'gwr_ds_{b}_{self.x_type}.tif'), -1
                )
                data_mat[b, :, :] = torch.FloatTensor(
                    temp
                ) / 5e3

            return data_mat

    def get_durbin(self):
        if self.pipeline_option == 'produce':
            noise_mat = self._load_noise_term_()
            beta_mat = self._load_linear_beta_mat_()
            x_mat = self._load_x_mat_()

            res_mat = torch.concat((
                x_mat[:2],                              # x1, x2
                self.coord_mat,                         # x_coord, y_coord
                torch.zeros(size=(1, self.h, self.w))   # y
            ))

            res_mat[-1] += beta_mat[0]                          # add beta0
            res_mat[-1] += torch.mul(beta_mat[1], res_mat[0])   # add beta1 * X1
            res_mat[-1] += torch.mul(beta_mat[2], res_mat[1])   # add beta2 * X2
            res_mat[-1] += noise_mat                            # add noise

            W = calc_adjacency_map(order=1, h=self.h, w=self.w)
            res_mat[-1] += self.theta * torch.matmul(
                W,
                res_mat[0].view(1, self.h * self.w).transpose(1, 0)
            ).view(self.h, self.w)
            res_mat[-1] += self.theta * torch.matmul(
                W,
                res_mat[1].view(1, self.h * self.w).transpose(1, 0)
            ).view(self.h, self.w)

            temp = torch.linalg.inv(torch.eye(self.h * self.w) - self.rho * W)
            res_mat[-1] = torch.matmul(
                temp,
                res_mat[-1].view(1, self.h * self.w).transpose(1, 0)
            ).view(self.h, self.w)

            noise_star_mat = torch.matmul(
                temp,
                noise_mat.reshape(1, self.h * self.w).transpose(1, 0)
            ).view(self.h, self.w)

            res_mat = (res_mat.numpy() * 5e3).astype(np.int32)
            noise_star_mat = (noise_star_mat.numpy() * 5e3).astype(np.int32)

            for bid, band in enumerate(res_mat):
                cv2.imwrite(
                    os.path.join(config.local_data_raster_syn_dir, f'durbin_ds_{bid}_{self.x_type}.tif'), band
                )
            cv2.imwrite(
                os.path.join(config.local_data_raster_syn_dir, f'durbin_noise-term_{self.x_type}.tif'), noise_star_mat
            )

        elif self.pipeline_option == 'load':
            print('[Notice]:\tloading dataset \'DURBIN\'')
            data_mat = torch.zeros(size=(5, self.h, self.w))
            for b in range(5):   # 5 bands
                temp = cv2.imread(
                    os.path.join(config.local_data_raster_syn_dir, f'durbin_ds_{b}_{self.x_type}.tif'), -1
                )
                data_mat[b, :, :] = torch.FloatTensor(
                    temp
                ) / 5e3

            return data_mat


def convert_tif_to_csv(syn_ds: str, csv_file: str):
    """
    Convert tif file to csv file.
    :param syn_ds: 'syn - ds - x_type'
    :param csv_file: 'syn-ds-x_type_ds.csv'
    """
    _, ds, x_type = syn_ds.split('-')
    col_repo = TabDataColumns
    gen = SynGenerator(x_type=x_type, pipeline_option='load')

    if ds == 'gwr':
        data_mat = gen.get_gwr()
    elif ds == 'sl':
        data_mat = gen.get_spatial_lag()
    elif ds == 'slx':
        data_mat = gen.get_spatial_lagged_x()
    elif ds == 'durbin':
        data_mat = gen.get_durbin()
    else:
        raise NotImplementedError(f'Illegal dataset \'{ds}\' to convert.')

    data_mat = data_mat.permute((1, 2, 0)).view(gen.h * gen.w, -1)
    df = pd.DataFrame(data=data_mat,
                      columns=col_repo.syn_atr + col_repo.syn_spa + col_repo.syn_y)
    df.to_csv(os.path.join(config.local_data_tabular_dir, csv_file))


if __name__ == '__main__':
    # convert_tif_to_csv(syn_ds='syn-durbin-dem', csv_file='syn-durbin-d-ds.csv')
    SynGenerator()
