import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from sklearn.metrics import r2_score

from configurations.model_config import XGBHyperparameters
from model.baseline_ds import load_tab_as_graph
from model.baseline import GWGCN, BaselineXGBoost, BaselineGWR


def train_srgcnn(epoch=1000):
    """
    Train a SRGCNN model.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x_tensor, y_tensor, adj, idx_trn, idx_val, idx_tst = load_tab_as_graph(
        ds='pm25', shuffle=True
    )
    x_tensor = x_tensor.to(device)
    y_tensor = y_tensor.to(device)
    adj = adj.to(device)
    idx_trn = idx_trn.to(device)
    idx_val = idx_val.to(device)
    idx_tst = idx_tst.to(device)
    N, f_in = x_tensor.shape

    print(f'[Notice]: f_in = {f_in}, N = {N}')

    srgcnn = GWGCN(N=N,
                   f_in=f_in,
                   n_classes=1,
                   hidden=[16 * f_in],
                   dropouts=[0.1]).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, srgcnn.parameters()),
        lr=4e-3
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor=1,
        end_factor=0.1,
        total_iters=400
    )

    srgcnn.to(device)
    print(srgcnn)

    train_loss_arr, val_loss_arr = [], []
    for i in range(epoch):
        with torch.autograd.set_detect_anomaly(True):
            srgcnn.train()
            optimizer.zero_grad()
            output = srgcnn(x_tensor, adj)
            loss = F.mse_loss(output[idx_trn], y_tensor[idx_trn])
            loss.backward()
            optimizer.step()
            train_loss_arr.append(loss.mean().item())

            if i % 100 == 0:
                # Validate
                srgcnn.eval()
                output = srgcnn(x_tensor, adj)
                val_loss = F.l1_loss(output[idx_val], y_tensor[idx_val])
                val_loss_arr.append(val_loss.mean().item())
                print(f'Epoch: {i}/{epoch}\t|\ttrain_loss (mse) = {loss.mean().item():.5f}'
                      f'\t|\tval_loss (mae) = {val_loss.mean().item():.5f}\t|\t'
                      f'loss = {scheduler.get_last_lr()[0]:.5f}')

            scheduler.step()

    srgcnn.eval()
    output = srgcnn(x_tensor, adj)
    tst_loss = F.l1_loss(output[idx_tst], y_tensor[idx_tst])
    tst_r2 = r2_score(output[idx_tst].detach().numpy(), y_tensor[idx_tst].detach().numpy())
    print(f'test loss (mae) = {tst_loss.mean().item()}\ntest r2 = {tst_r2}')

    plt.figure(figsize=(3, 4))
    plt.plot([-1e3, 1e3], [-1e3, 1e3], c='r', linewidth=0.5)
    plt.scatter(output[idx_tst].detach().numpy(),
                y_tensor[idx_tst].detach().numpy(),
                s=18, edgecolors='0.8', linewidths=0.4)

    axis_ranger = np.concatenate((output[idx_tst].detach().numpy(),
                                  y_tensor[idx_tst].detach().numpy()))
    plt.xlabel(f'Predicted')
    plt.ylabel(f'Real')
    plt.xlim(axis_ranger.min() - 1, axis_ranger.max() + 1)
    plt.ylim(axis_ranger.min() - 1, axis_ranger.max() + 1)
    plt.title(f'test set\n' +
              '${R^2 = }$' +
              '{:.4f}'.format(tst_r2) + '\n' +
              '${MAE = }$' +
              '{:.4f}'.format(tst_loss))
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    """
    Baseline experiments on the PM25 dataset.
    For other datasets, change the codes in baseline_ds.py and baseline.py as needed.
    """
    """ XGBOOST """
    hp = XGBHyperparameters.hp_tab_shp_mae
    bx = BaselineXGBoost(ds='shp', train_ratio=0.56)
    # bx.tune(n_trials=200)
    bx.train_xgboost(param_dict=hp, mae=True)
    bx.draw_visual_validate(split_mode='test', draw_scatter=True, draw_residual_map=False)

    """SRGCNN"""
    # train_srgcnn(epoch=5000)

    """GWR"""
    # gwr = BaselineGWR(ds='pm25', train_ratio=0.7)
    # print(gwr.predict())
