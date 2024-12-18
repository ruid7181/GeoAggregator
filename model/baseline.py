import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW

import optuna
from optuna.samplers import TPESampler

from model.baseline_ds import load_tab_as_df
from tools.utils import get_config
from configurations.model_config import RegisteredDS, TabDataColumns


class BaselineXGBoost:
    def __init__(self, ds, train_ratio=0.7):
        """Load df data."""
        self.ds = ds
        if self.ds in RegisteredDS.tab:
            self.df = load_tab_as_df(ds=ds, shuffle=False)
        else:
            raise NotImplementedError(f"Dataset {self.ds} not registered.")

        self.trn_n_sample = int(train_ratio * len(self.df))
        self.val_n_sample = int(0.14 * len(self.df))
        self.tst_n_sample = int(0.3 * len(self.df))

        self.train_df = self.df.sample(frac=1.0, random_state=41)[: self.trn_n_sample]
        self.val_df = self.df.sample(frac=1.0, random_state=41)[self.trn_n_sample: self.trn_n_sample + self.val_n_sample]
        self.test_df = self.df.sample(frac=1.0, random_state=41)[self.trn_n_sample + self.val_n_sample:]

        self.train_ratio = train_ratio
        self.best_params = None
        self.best_value = None
        self.best_trial = None

        self.trained_model = None
        self.config = get_config()

    def __objective__(self, trial, n_split=5):
        param = {
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 12),
            'gamma': trial.suggest_float('gamma', 0.01, 1.0),
            'subsample': trial.suggest_float('subsample', 0.01, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0),
            'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 8, 96),
            'verbosity': 0,
            "objective": 'reg:squarederror'
        }
        loss = np.empty(n_split)
        kf = KFold(n_splits=n_split, shuffle=True)
        train_X, train_y = self.train_df.drop(columns=['y']), self.train_df['y']

        for idx, (trn_idx, val_idx) in enumerate(kf.split(train_X, train_y)):
            trn_X, trn_y = train_X.iloc[trn_idx], train_y.iloc[trn_idx]
            val_X, val_y = train_X.iloc[val_idx], train_y.iloc[val_idx]

            model = XGBRegressor(**param)
            model.fit(X=trn_X, y=trn_y,
                      eval_set=[(trn_X, trn_y),
                                (val_X, val_y)]
                      )

            pred_y = model.predict(val_X)
            # loss[idx] = mean_absolute_error(val_y, pred_y)
            loss[idx] = mean_squared_error(val_y, pred_y)

        return np.mean(loss)

    def tune(self, n_trials=50, study_name='regression', sampler=TPESampler()):
        start_time = time.time()
        study = optuna.create_study(
            direction='minimize',
            study_name=study_name,
            sampler=sampler)
        study.optimize(self.__objective__, n_trials=n_trials)

        self.best_params = study.best_params
        self.best_value = study.best_value
        self.best_trial = study.best_trial

        end_time = time.time()
        print('Elapsed time = {:.4f}s'.format(end_time - start_time))
        print(self.best_params)
        print(self.best_value)
        print(self.best_trial)

    def train_xgboost(self, param_dict=None, mae=False):
        assert (self.best_params or param_dict) is not None
        trn_x, trn_y = self.train_df.drop(columns=['y']), self.train_df['y']
        val_x, val_y = self.val_df.drop(columns=['y']), self.val_df['y']

        try:
            self.trained_model = XGBRegressor(**self.best_params)
        except TypeError:
            self.trained_model = XGBRegressor(**param_dict)

        start_time = time.time()
        self.trained_model.fit(
            X=trn_x,
            y=trn_y,
            eval_set=[(trn_x, trn_y),
                      (val_x, val_y)]
        )
        end_time = time.time()
        if mae:
            mae = np.mean(np.abs(self.trained_model.predict(self.val_df.drop(columns=['y'])) - self.val_df['y']))
            r_square = r2_score(self.trained_model.predict(self.val_df.drop(columns=['y'])), self.val_df['y'])

            print('Elapsed time = {:.4f}s'.format(end_time - start_time))
            print(f'Validation mae = {mae}\nValidation r-square = {r_square}')
        else:
            mse = np.mean(np.abs(self.trained_model.predict(self.val_df.drop(columns=['y'])) - self.val_df['y']) ** 2)
            r_square = r2_score(self.trained_model.predict(self.val_df.drop(columns=['y'])), self.val_df['y'])

            print('Elapsed time = {:.4f}s'.format(end_time - start_time))
            print(f'mse = {mse}\nr-square = {r_square}')

    def draw_visual_validate(self, split_mode='train', draw_residual_map=False, draw_scatter=False):
        if split_mode == 'all':
            pred_y = self.trained_model.predict(self.df.drop(columns=['y']))
            real_y = self.df['y'].to_numpy()
        elif split_mode == 'train':
            pred_y = self.trained_model.predict(self.train_df.drop(columns=['y']))
            real_y = self.train_df['y'].to_numpy()
        elif split_mode == 'test':
            pred_y = self.trained_model.predict(self.test_df.drop(columns=['y']))
            real_y = self.test_df['y'].to_numpy()
        else:
            raise TypeError(f'split mode {split_mode} not implemented.')

        if draw_residual_map:
            assert split_mode == 'all', 'To draw residual map, split mode must be \'all\'.'
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
            real = axes[0].imshow(real_y.reshape((50, 50)), cmap='Spectral_r')
            axes[0].set_title('Ground Truth Map')

            residual = axes[1].imshow((pred_y - real_y).reshape((50, 50)), cmap='coolwarm')
            axes[1].set_title('Residual Map')
            fig.subplots_adjust(wspace=0.4, hspace=0)
            plt.subplots_adjust(left=0.05)

            cax = fig.add_axes([axes[0].get_position().x1 + 0.02,
                                axes[0].get_position().y0, 0.02,
                                axes[0].get_position().height])
            fig.colorbar(real, cax=cax)
            cax = fig.add_axes([axes[1].get_position().x1 + 0.02,
                                axes[1].get_position().y0, 0.02,
                                axes[1].get_position().height])
            fig.colorbar(residual, cax=cax)
            residual.set_clim(-5, 5)
            plt.show()

        if draw_scatter:
            # real_y, pred_y = real_y[~np.isnan(real_y)], pred_y[~np.isnan(pred_y)]
            plt.figure(figsize=(3, 4))
            plt.plot([-1e3, 1e3], [-1e3, 1e3], c='r', linewidth=0.5)
            plt.scatter(pred_y, real_y, edgecolors='0.8', s=18, linewidths=0.4)

            axis_ranger = np.concatenate((pred_y, real_y))
            plt.xlabel(f'Predicted {self.ds}')
            plt.ylabel(f'Real {self.ds}')
            plt.xlim(axis_ranger.min() - 1, axis_ranger.max() + 1)
            plt.ylim(axis_ranger.min() - 1, axis_ranger.max() + 1)
            plt.title(f'\'{self.ds}\' {split_mode} set\n' +
                      '${R^2 = }$' +
                      '{:.4f}'.format(r2_score(y_true=real_y.flatten(), y_pred=pred_y.flatten())) + '\n' +
                      '${MAE = }$' +
                      '{:.4f}'.format(np.mean(np.abs(pred_y - real_y))))
            plt.tight_layout()
            plt.show()


class GWGraphConvolution(nn.Module):
    """
    Geographically weighted graph convolution operation that
    adds locally parameterized weights to all the variables (to be used in the SRGCNNs-GW model)

    ### Implementation of: ###
    Zhu, Di, et al. "Spatial regression graph convolutional neural networks:
    A deep learning paradigm for spatial multivariate distributions." GeoInformatica 26.4 (2022): 645-676.
    """

    def __init__(self, f_in, f_out, N, use_bias=True, activation=nn.Tanh()):
        super().__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.n_sample = N
        self.use_bias = use_bias
        self.activation = activation
        # Geographically local parameters
        self.gwr_weight = nn.Parameter(torch.FloatTensor(N, f_in), requires_grad=True)
        self.weight = nn.Parameter(torch.FloatTensor(f_in, f_out), requires_grad=True)
        self.bias = nn.Parameter(torch.FloatTensor(f_out), requires_grad=True) if use_bias else None
        self.initialize_weights()

        print(f'Total params = {N * f_in + f_in * f_out + f_out}.')

    def initialize_weights(self):
        nn.init.constant_(self.weight, 1)
        nn.init.constant_(self.gwr_weight, 1)
        if self.use_bias: nn.init.constant_(self.bias, 0)

    def forward(self, input, adj):
        # self.method == 'SRGCNN-GW':
        gwr_support = torch.mul(input, self.gwr_weight)  # use torch.mul to enable element-wise product
        support = torch.mm(adj, gwr_support)  # adj here has to be re-normalized
        out = torch.mm(support, self.weight)

        if self.use_bias: out.add_(self.bias)
        if self.activation is not None:
            out = self.activation(out)
        return out


class GWGCN(nn.Module):
    """
    SRGCNN-GW model

    ### Implementation of: ###
    Zhu, Di, et al. "Spatial regression graph convolutional neural networks:
    A deep learning paradigm for spatial multivariate distributions." GeoInformatica 26.4 (2022): 645-676.
    """
    def __init__(self, f_in, N, n_classes, hidden=[16], dropouts=[0.0]):
        if hidden == []:
            super().__init__()
            self.layers = []
            self.dropouts = []
            self.out_layer = GWGraphConvolution(f_in, n_classes, N, activation=None)   # None for regression task.
        else:
            super().__init__()
            layers = []
            for f_in, f_out in zip([f_in] + hidden[:-1], hidden):
                layers += [GWGraphConvolution(f_in, f_out, N)]

            self.layers = nn.Sequential(*layers)
            self.dropouts = dropouts
            self.out_layer = GWGraphConvolution(f_out, n_classes, N, activation=None)   # None for regression task.

            print(f'f_out = {f_out}, f_in = {f_in}, n_classes = {n_classes}, N = {N}.')

    def forward(self, x, adj):
        for layer, d in zip(self.layers, self.dropouts):
            x = layer(x, adj)
            if d > 0: x = F.dropout(x, d, training=self.training, inplace=False)

        return self.out_layer(x, adj)


class BaselineGWR:
    def __init__(self, ds, train_ratio=0.7):
        self.pred_X = None
        self.pred_y = None
        self.cal_coords = None
        self.cal_X = None
        self.cal_y = None
        self.pred_coords = None
        self.ds = ds
        if self.ds in RegisteredDS.tab:
            self.df = load_tab_as_df(ds=ds, shuffle=False)
        else:
            raise NotImplementedError(f"Dataset {self.ds} not registered.")

        self.train_ratio = train_ratio

        self.trn_n_sample = int(train_ratio * len(self.df))
        self.val_n_sample = int(0.1 * len(self.df))
        self.tst_n_sample = int(0.2 * len(self.df))

        self.train_df = self.df.sample(frac=1.0, random_state=41)[: self.trn_n_sample]
        self.val_df = self.df.sample(frac=1.0, random_state=41)[self.trn_n_sample: self.trn_n_sample + self.val_n_sample]
        self.test_df = self.df.sample(frac=1.0, random_state=41)[self.trn_n_sample + self.val_n_sample:]

        self.config = get_config()
        self.col_repository = TabDataColumns

    def predict(self):
        if self.ds == 'shp':
            self.cal_y = self.train_df[self.col_repository.shp_y].values.reshape((-1, 1))
            self.cal_X = self.train_df[self.col_repository.shp_atr].values
            self.cal_coords = self.train_df[self.col_repository.shp_spa].values

            self.pred_y = self.test_df[self.col_repository.shp_y].values.reshape((-1, 1))
            self.pred_X = self.test_df[self.col_repository.shp_atr].values
            self.pred_coords = self.test_df[self.col_repository.shp_spa].values
        elif self.ds == 'pm25':
            self.cal_y = self.train_df[self.col_repository.pm25_y].values.reshape((-1, 1))
            self.cal_X = self.train_df[self.col_repository.pm25_atr].values
            self.cal_coords = self.train_df[self.col_repository.pm25_spa].values

            self.pred_y = self.test_df[self.col_repository.pm25_y].values.reshape((-1, 1))
            self.pred_X = self.test_df[self.col_repository.pm25_atr].values
            self.pred_coords = self.test_df[self.col_repository.pm25_spa].values
        elif self.ds == 'sdoh':
            self.cal_y = self.train_df[self.col_repository.sdoh_y].values.reshape((-1, 1))
            self.cal_X = self.train_df[self.col_repository.sdoh_atr].values
            self.cal_coords = self.train_df[self.col_repository.sdoh_spa].values

            self.pred_y = self.test_df[self.col_repository.sdoh_y].values.reshape((-1, 1))
            self.pred_X = self.test_df[self.col_repository.sdoh_atr].values
            self.pred_coords = self.test_df[self.col_repository.sdoh_spa].values
        elif 'syn' in self.ds:
            self.cal_y = self.train_df[self.col_repository.syn_y].values.reshape((-1, 1))
            self.cal_X = self.train_df[self.col_repository.syn_atr].values
            self.cal_coords = self.train_df[self.col_repository.syn_spa].values

            self.pred_y = self.test_df[self.col_repository.syn_y].values.reshape((-1, 1))
            self.pred_X = self.test_df[self.col_repository.syn_atr].values
            self.pred_coords = self.test_df[self.col_repository.syn_spa].values

        # Calibrate GWR model
        gwr_selector = Sel_BW(self.cal_coords, self.cal_y, self.cal_X)
        gwr_bw = gwr_selector.search(bw_min=2)
        print(f'Learned band width: {gwr_bw}')
        model = GWR(self.cal_coords, self.cal_y, self.cal_X, gwr_bw)
        gwr_results = model.fit()
        print(gwr_results.summary())
        scale = gwr_results.scale
        residuals = gwr_results.resid_response

        pred_results = model.predict(self.pred_coords, self.pred_X, scale, residuals)

        plt.figure(figsize=(3, 4))
        plt.plot([-1e3, 1e3], [-1e3, 1e3], c='r', linewidth=0.5)
        plt.scatter(pred_results.predictions, self.pred_y,
                    edgecolors='0.8', s=18, linewidths=0.4)

        axis_ranger = np.concatenate((pred_results.predictions, self.pred_y))
        plt.xlabel(f'Predicted {self.ds}')
        plt.ylabel(f'Real {self.ds}')
        plt.xlim(axis_ranger.min() - 1, axis_ranger.max() + 1)
        plt.ylim(axis_ranger.min() - 1, axis_ranger.max() + 1)
        plt.title(f'\'{self.ds}\' test set\n' +
                  '${R^2 = }$' +
                  '{:.4f}'.format(r2_score(pred_results.predictions, self.pred_y)) + '\n' +
                  '${MAE = }$' +
                  '{:.4f}'.format(mean_absolute_error(pred_results.predictions, self.pred_y)))
        plt.tight_layout()
        plt.show()

        return mean_absolute_error(pred_results.predictions, self.pred_y)
