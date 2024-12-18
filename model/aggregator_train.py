import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

from tools.utils import get_config
from configurations.model_config import AggregatorHyperParameters as params
from model.aggregator_ds import TabDataLoaderWrapper
from model.aggregator import MaskedVanillaTransformer, GeoAggregator
from model.customs import MaskedMAELoss

config = get_config()


def target_pos_gt_locator(input_tensor, dists):
    """
    Locate the target variable of the target point in the input tensor.
    :param input_tensor: [bs, sl, fd]
    :param dists: [bs, sl]
    :return: target_position: [3, bs], gt: [bs, 1, 1]
    """
    batch_size = input_tensor.shape[0]
    # * -> [2, bs]
    target_position = torch.argwhere(dists == 0).transpose(1, 0)
    # [2, bs] -> [3, bs]
    target_position = torch.cat((target_position, torch.IntTensor([-1] * batch_size).unsqueeze(0)), dim=0)

    gt = input_tensor[[target_position[0], target_position[1], target_position[2]]]
    gt = gt.unsqueeze(-1).unsqueeze(-1)

    return target_position, gt


def train_aggregator(dataset='shp',
                     batch_size=4,
                     epoch=200,
                     not_decreasing_rounds=5,
                     sample_radius=0.014,
                     sequence_len=36,
                     attn_dropout=0.01,
                     n_attn_layer=1,
                     inducing_points=4,
                     attn_bias_factor=50,
                     model_save_fn=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ########################################## #
    # Load Datasets:                             #
    # ########################################## #

    wrapper = TabDataLoaderWrapper(
        ds=dataset, shuffle=True, batch_size=batch_size,
        split_mode='train', get_sample_loc=False,
        sample_radius=sample_radius, sequence_len=sequence_len
    )
    trn_loader = wrapper.get_dataloader()
    if sample_radius is None:
        sample_radius = wrapper.get_dataiter().s_radius
    wrapper = TabDataLoaderWrapper(
        ds=dataset, shuffle=True, batch_size=batch_size,
        split_mode='val', get_sample_loc=False,
        sample_radius=sample_radius, sequence_len=sequence_len
    )
    val_loader = wrapper.get_dataloader()

    # ########################################## #
    # Initiate Model:                            #
    # ########################################## #

    geo_aggregator = GeoAggregator(
        x_dims=wrapper.atr_dims,
        spa_dims=wrapper.spa_dims,
        y_dims=wrapper.target_dims,
        n_attn_layer=n_attn_layer,
        inducing_points=inducing_points,
        seq_len=sequence_len,
        attn_dropout=attn_dropout,
        attn_bias_factor=attn_bias_factor,
        dc_lin_dims=params.decoder_lin_dims
    )
    # geo_aggregator = MaskedVanillaTransformer(
    #     x_dims=wrapper.atr_dims,
    #     spa_dims=wrapper.spa_dims,
    #     y_dims=wrapper.target_dims,
    #     n_attn_layer=n_attn_layer,
    #     inducing_points=inducing_points,
    #     seq_len=sequence_len,
    #     attn_dropout=attn_dropout,
    #     attn_bias_factor=attn_bias_factor,
    #     dc_lin_dims=params.decoder_lin_dims,
    #     hidden_token=hidden_token
    # )
    geo_aggregator.to(device=device)

    criteria = MaskedMAELoss()
    criteria.to(device)
    optimizer = torch.optim.Adam(
        params=geo_aggregator.parameters(),
        betas=(0.9, 0.999),
        lr=5e-3,
        weight_decay=1e-4
    )
    lr_schedular = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=5e-3,
        steps_per_epoch=len(trn_loader),
        epochs=epoch,
        pct_start=0.02
    )

    # ########################################## #
    # Optimize:                                  #
    # ########################################## #

    losses, loss_run_avg, val_losses = [], [], np.ones(3) * 1e9

    for epo in range(1, epoch + 1):
        print(f'Epoch: {epo}/{epoch}')
        geo_aggregator.train()
        for idx, (data, dists) in enumerate(trn_loader):
            input_tensor = data.to(device)
            optimizer.zero_grad()

            target_position, gt = target_pos_gt_locator(input_tensor=input_tensor, dists=dists)
            pred_y = geo_aggregator(input_tensor, target_position, dists)

            loss = criteria(gt, pred_y)
            loss_run_avg.append(loss.item())
            loss.backward()
            optimizer.step()
            lr_schedular.step()

            if idx % 500 == 0:
                step_loss = sum(loss_run_avg) / len(loss_run_avg)
                losses.append(step_loss)
                loss_run_avg = []
                print(f'Step: {idx}/{len(trn_loader)}\t|\t'
                      f'loss_run_avg:{step_loss:.5f}\t|\t'
                      f'lr:{lr_schedular.get_last_lr()[0]:.5f}')

        # Early stopping:
        if not_decreasing_rounds > 0:
            print(f'Validation at epoch: {epo}/{epoch}')
            geo_aggregator.eval()
            val_losses_this_epo = []
            with torch.no_grad():
                for idx, (data, dists) in enumerate(val_loader):
                    input_tensor = data.to(device)

                    target_position, gt = target_pos_gt_locator(input_tensor=input_tensor, dists=dists)
                    pred_y = geo_aggregator(input_tensor, target_position, dists)

                    loss = criteria(gt, pred_y)
                    val_losses_this_epo.append(loss.item())
                val_loss_this_epo = sum(val_losses_this_epo) / len(val_losses_this_epo)
                if not (val_loss_this_epo < val_losses[-not_decreasing_rounds:]).any():
                    print(f'Early stopped at epo {epo}')
                    break
                else:
                    val_losses = np.append(val_losses, val_loss_this_epo)
                    print(f'Validation loss = {val_loss_this_epo:.5f}')
        print(f'learned abf = {geo_aggregator.attn.attn_bias_factor.item()}')

    # ########################################## #
    # Test:                                      #
    # ########################################## #

    criteria = MaskedMAELoss()
    wrapper = TabDataLoaderWrapper(
        ds=dataset, shuffle=True, batch_size=1,
        split_mode='test', get_sample_loc=False,
        sample_radius=sample_radius, sequence_len=sequence_len
    )
    tst_loader = wrapper.get_dataloader()

    losses, predictions, gts = [], [], []
    geo_aggregator.eval()
    with torch.no_grad():
        for idx, (data, dists) in enumerate(tst_loader):
            input_tensor = data.to(device)

            target_position, gt = target_pos_gt_locator(input_tensor=input_tensor, dists=dists)
            pred_y = geo_aggregator(input_tensor, target_position, dists)

            loss = criteria(gt, pred_y)
            losses.append(loss.item())
            predictions.append(pred_y.item())
            gts.append(gt.item())

    print('Average test loss:', sum(losses) / len(losses))
    predictions, gts = np.array(predictions), np.array(gts)

    # ########################################## #
    # Save Model:                                #
    # ########################################## #

    if model_save_fn:
        state = {'net': geo_aggregator.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, str(os.path.join(config.model_save_path, model_save_fn)))

    # ########################################## #
    # Plot regression results:                   #
    # ########################################## #

    plt.figure(figsize=(3, 4))
    plt.plot([-1e3, 1e3], [-1e3, 1e3], c='r', linewidth=0.5)
    plt.scatter(predictions, gts, edgecolors='0.8', s=18, linewidths=0.4)

    axis_ranger = np.concatenate((predictions, gts))
    plt.xlabel(f'Predicted {dataset}')
    plt.ylabel(f'Real {dataset}')
    plt.xlim(axis_ranger.min() - 1, axis_ranger.max() + 1)
    plt.ylim(axis_ranger.min() - 1, axis_ranger.max() + 1)
    plt.title(f'\'{dataset}\' test set\n' +
              '${R^2 = }$' +
              '{:.4f}'.format(r2_score(y_true=gts.flatten(), y_pred=predictions.flatten())) + '\n' +
              '${MAE = }$' +
              '{:.4f}'.format(np.mean(np.abs(predictions - gts))))
    plt.tight_layout()
    plt.show()

    print(f'r2 = {r2_score(gts.flatten(), predictions.flatten())}')
    print(f'mae = {mean_absolute_error(gts.flatten(), predictions.flatten())}')
