import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention import CartesianPerceiver, VanillaPerceiver
from configurations.model_config import AggregatorHyperParameters as params
from model.customs import MaskedMAELoss, NanBatchNorm1dNaive


class FCNEncoder(nn.Module):
    """
    FCN encodes raw features to higher-dimensional features.
    Not encoding spa_feats, where we use 2-D rotary PE instead.
    """
    def __init__(self, x_dims, y_dims, spa_dims):
        self.x_dims, self.y_dims, self.spa_dims = x_dims, y_dims, spa_dims
        super(FCNEncoder, self).__init__()
        self.x_encoder = nn.Sequential(
            nn.Linear(len(self.x_dims), params.d_model // 2, bias=True),
            nn.Tanhshrink(),
            nn.Linear(params.d_model // 2, params.d_model // 2, bias=True)
        )
        self.y_encoder = nn.Sequential(
            nn.Linear(len(self.y_dims), params.d_model // 2, bias=True),
            nn.Tanhshrink(),
            nn.Linear(params.d_model // 2, params.d_model // 2, bias=True)
        )
        self.learnable_y = nn.Parameter(torch.randn(1, 1, params.d_model // 2))

    def forward(self, x_tensor, y_tensor):
        """
        :param x_tensor: [bs, sl, x_dim]
        :param y_tensor: [bs, sl-1, 1]
        """
        batch_size, seq_len = x_tensor.shape[0], x_tensor.shape[1]
        x_embed = self.x_encoder(x_tensor)   # -> [bs, sl, dm//2]
        y_embed = self.y_encoder(y_tensor)   # -> [bs, sl-1, dm//2]

        y_embed = torch.concat((
            y_embed,
            self.learnable_y.repeat(batch_size, 1, 1)
        ), dim=1)   # -> [bs, sl, dm//2]

        return x_embed, y_embed   # [bs, sl, dm//2], [bs, sl, dm//2]


class FCNDecoder(nn.Module):
    def __init__(self, linear_dims):
        super(FCNDecoder, self).__init__()
        self.decoder = nn.ModuleList([
            nn.Linear(linear_dims[lyr], linear_dims[lyr + 1]) for lyr in range(len(linear_dims) - 1)
        ])

    def forward(self, encoding_tensor):
        """
        :param encoding_tensor: encoder output. [bs, 1, dm]
        :return: regression result [bs, 1, 1]
        """
        in_mat = encoding_tensor
        for lyr in self.decoder[:-1]:
            in_mat = lyr(in_mat)
            in_mat = F.tanhshrink(in_mat)

        in_mat = self.decoder[-1](in_mat)

        return in_mat


class RotaryEmbeddingNaive:
    """
    Naive implementation of 2D rotary embedding.
    """
    def __init__(self, d=4):
        self.d = d
        self.n_thetas = self.d // 4
        self.thetas = torch.arange(0, self.n_thetas, 1)
        self.thetas = 10000 ** ((2 - 2 * self.thetas) / self.d)
        self.thetas = self.thetas.unsqueeze(0).unsqueeze(2)     # [1, n_thetas, 1]

    def embed(self, spa_feat):
        """
        :param spa_feat: [bs, sl, 2]
        :return: Rotation matrix R [bs, 1, d, d]
        """
        batch_size, seq_len, _ = spa_feat.shape
        spa_feat = spa_feat.permute(0, 2, 1)               # [bs, 2, sl]
        thetas = self.thetas.repeat(batch_size, 1, 1)      # [bs, nt, 1]

        thetas_x = torch.bmm(thetas, spa_feat[:, :1, :])   # [bs, nt, 1] * [bs, 1, sl] -> [bs, nt, sl]
        thetas_y = torch.bmm(thetas, spa_feat[:, 1:, :])   # [bs, nt, sl]

        R_x = torch.stack([
            torch.cos(thetas_x),
            - torch.sin(thetas_x),
            torch.sin(thetas_x),
            torch.cos(thetas_x)
        ], dim=-1)                                                                        # [bs, nt, sl, 4]
        R_x = R_x.view(batch_size, self.n_thetas, seq_len, 2, 2).permute(0, 2, 1, 3, 4)   # [bs, sl, nt, 2, 2]

        R_y = torch.stack([
            torch.cos(thetas_y),
            - torch.sin(thetas_y),
            torch.sin(thetas_y),
            torch.cos(thetas_y)
        ], dim=-1)                                                                        # [bs, nt, sl, 4]
        R_y = R_y.view(batch_size, self.n_thetas, seq_len, 2, 2).permute(0, 2, 1, 3, 4)   # [bs, sl, nt, 2, 2]

        RR = torch.stack((R_x, R_y), dim=3)   # [bs, sl, nt, 2D, 2, 2]
        RR = RR.view(batch_size, seq_len, self.n_thetas * 2, 2, 2)   # [bs, sl, nt * 2D, 2, 2]

        R = torch.stack([
            torch.stack([
                torch.block_diag(*RR[bs][sl]) for sl in range(seq_len)
            ]) for bs in range(batch_size)
        ])   # [bs, sl, dk, dk]

        return R


class GeoAggregator(nn.Module):
    def __init__(self, x_dims, spa_dims, y_dims, n_attn_layer=2, inducing_points=4, attn_dropout=0.2,
                 attn_bias_factor=1, seq_len=25, dc_lin_dims=None):
        """
        :param spa_dims: spatial feature dims (exclude target)
        :param x_dims: x feature dims
        :param y_dims: y feature dims
        :param attn_bias_factor: Attention score bias
        :param seq_len: Input sequence length
        :param dc_lin_dims: Decoder linear dims.
        """
        super(GeoAggregator, self).__init__()
        # ----------------------------------------------------------------
        self.seq_len = seq_len
        self.x_dims = x_dims
        self.spa_dims = spa_dims
        self.y_dims = y_dims
        # ----------------------------------------------------------------
        # Encoder
        self.n_batch_norm = NanBatchNorm1dNaive(n_feature=len(self.x_dims + self.y_dims))
        self.feat_ec = FCNEncoder(
            spa_dims=spa_dims,
            x_dims=x_dims,
            y_dims=y_dims
        )
        self.rotary_embed = RotaryEmbeddingNaive(d=8)
        self.attn = CartesianPerceiver(
            d_model=params.d_model,
            n_attn_layer=n_attn_layer,
            attn_dropout=attn_dropout,
            attn_bias_factor=attn_bias_factor,
            n_hidden_token=inducing_points
        )
        # ----------------------------------------------------------------
        # Decoder
        self.dc_lin_dims = dc_lin_dims.copy()
        self.dc_lin_dims.insert(0, params.d_model + len(self.x_dims + self.spa_dims))   # considering res connection.

        self.y_dc = FCNDecoder(linear_dims=self.dc_lin_dims)
        self.loss = MaskedMAELoss()

    def encoder(self, input_tensor, mask, dists):
        """
        :param input_tensor: [bs, sl, fd]
        :param mask: [bs, sl-1], masked: 1, not masked: 0.
        :param dists: [bs, sl-1]
        """
        # -> [bs, sl, 2]
        spa_embed = input_tensor[:, :, self.spa_dims]
        # [bs, sl, 2] -> [bs, sl, dk, dk]
        spa_embed = self.rotary_embed.embed(spa_embed)
        # -> [bs, sl, x_dims]
        x_embed = input_tensor[:, :, self.x_dims]
        # -> [bs, sl-1, y_dims]
        y_embed = input_tensor[:, :-1, self.y_dims]
        # -> [bs, sl, dm//2], [bs, sl, dm//2]
        x_embed, y_embed = self.feat_ec(x_tensor=x_embed, y_tensor=y_embed)

        # -> [2, bs, ql(1), dm//2]
        return self.attn(
            a_embed=x_embed[:, :-1, :],
            b_embed=y_embed[:, :-1, :],
            q_a_embed=x_embed[:, -1:, :],
            q_b_embed=y_embed[:, -1:, :],
            ctx_spa_embed=spa_embed[:, :-1, :, :],
            q_spa_embed=spa_embed[:, -1:, :, :],
            mask=mask,
            dists=dists
        )

    def decoder(self, encoding_tensor):
        """
        :param encoding_tensor: [bs, ql, dm]
        """
        pred_y = self.y_dc(encoding_tensor)

        return pred_y

    def forward(self, input_tensor, target_position, dists, get_attn_score=False):
        """
        :param dists: [bs, sl]
        :param input_tensor: [bs, sl, x_dims + y_dims + spa_dims]
        :param target_position: [3, bs]
        :param get_attn_score: True or False
        """
        batch_size = input_tensor.shape[0]
        # Preprocess
        mask = torch.where(input_tensor.isnan(), 1, 0)[:, :-1, 0]   # -> [bs, sl-1]
        input_tensor[:, :, self.x_dims + self.y_dims] = \
            self.n_batch_norm(input_tensor[:, :, self.x_dims + self.y_dims])

        input_tensor = input_tensor.nan_to_num()
        dists = dists.nan_to_num()
        dists[[target_position[0], target_position[1]]] = torch.nan
        dists = dists[~dists.isnan()].view(batch_size, -1)   # -> [bs, sl-1]

        central_x = input_tensor[[target_position[0], target_position[1]]][:, :-1].clone()   # [bs, fd-1]
        central_x = central_x.unsqueeze(1)   # [bs, fd-1] -> [bs, 1, fd-1]

        # Encoder
        encoding_tensor, attn_weights = self.encoder(
            input_tensor=input_tensor,
            mask=mask,
            dists=dists
        )

        _, batch_size, query_len, d_model_half = encoding_tensor.shape
        encoding_tensor = encoding_tensor.permute(1, 2, 0, 3).contiguous().view(batch_size, query_len, -1)
        encoding_tensor = torch.concat((encoding_tensor, central_x), dim=-1)   # res connection of central x.

        # Decoder
        pred = self.decoder(encoding_tensor=encoding_tensor)   # -> [bs, 1, 1]

        if get_attn_score:
            return pred, attn_weights
        else:
            return pred


class MaskedVanillaTransformer(nn.Module):
    def __init__(self, x_dims, spa_dims, y_dims, n_attn_layer=2, hidden_token=4, attn_dropout=0.2,
                 attn_bias_factor=1, seq_len=25, dc_lin_dims=None):
        """
        :param spa_dims: spatial feature dims (exclude target)
        :param x_dims: x feature dims
        :param y_dims: y feature dims
        :param n_attn_layer:
        :param attn_dropout:
        :param attn_bias_factor: Attention score bias
        :param seq_len: Same as TabDataSampler Class in aggregator_ds.
        :param dc_lin_dims: Decoder linear dims.
        """
        super(MaskedVanillaTransformer, self).__init__()
        # ----------------------------------------------------------------
        self.seq_len = seq_len
        self.x_dims = x_dims
        self.spa_dims = spa_dims
        self.y_dims = y_dims
        # ----------------------------------------------------------------
        # Encoder
        self.n_batch_norm = NanBatchNorm1dNaive(n_feature=len(self.x_dims + self.y_dims))
        self.feat_ec = FCNEncoder(
            spa_dims=spa_dims,
            x_dims=x_dims,
            y_dims=y_dims
        )
        self.rotary_embed = RotaryEmbeddingNaive(d=8)
        self.attn = VanillaPerceiver(
            d_model=params.d_model,
            n_head=4,
            n_attn_layer=n_attn_layer,
            n_hidden_token=hidden_token,
            attn_dropout=attn_dropout,
            attn_bias_factor=attn_bias_factor
        )
        # ----------------------------------------------------------------
        # Decoder
        self.dc_lin_dims = dc_lin_dims.copy()
        self.dc_lin_dims.insert(0, params.d_model + len(self.x_dims + self.spa_dims))   # considering res connection.

        self.y_dc = FCNDecoder(linear_dims=self.dc_lin_dims)
        self.loss = MaskedMAELoss()

    def encoder(self, input_tensor, mask, dists):
        """
        :param input_tensor: [bs, sl, fd]
        :param mask: [bs, sl-1], masked: 1, not masked: 0.
        :param dists: [bs, sl-1]
        """
        # -> [bs, sl, 2]
        spa_embed = input_tensor[:, :, self.spa_dims]
        # [bs, sl, 2] -> [bs, sl, dk, dk]
        spa_embed = self.rotary_embed.embed(spa_embed)
        # -> [bs, sl, x_dims]
        x_embed = input_tensor[:, :, self.x_dims]
        # -> [bs, sl-1, y_dims]
        y_embed = input_tensor[:, :-1, self.y_dims]
        # -> [bs, sl, dm//2], [bs, sl, dm//2]
        x_embed, y_embed = self.feat_ec(x_tensor=x_embed, y_tensor=y_embed)
        # -> [bs, sl-1, dm]
        ctx_embed = torch.cat((x_embed[:, :-1, :], y_embed[:, :-1, :]), dim=-1)
        q_embed = torch.cat((x_embed[:, -1:, :], y_embed[:, -1:, :]), dim=-1)

        # -> [bs, ql(1), dm]
        return self.attn(
            ctx_embed=ctx_embed,
            q_embed=q_embed,
            ctx_spa_embed=spa_embed[:, :-1, :, :],
            q_spa_embed=spa_embed[:, -1:, :, :],
            mask=mask,
            dists=dists
        )

    def decoder(self, encoding_tensor):
        """
        :param encoding_tensor: [bs, ql, dm]
        """
        pred_y = self.y_dc(encoding_tensor)

        return pred_y

    def forward(self, input_tensor, target_position, dists, get_attn_score=False):
        """
        :param dists: [bs, sl]
        :param input_tensor: [bs, sl, x_dims + y_dims + spa_dims]
        :param target_position: [3, bs]
        :param get_attn_score: True or False
        """
        batch_size = input_tensor.shape[0]
        # Preprocess
        mask = torch.where(input_tensor.isnan(), 1, 0)[:, :-1, 0]   # -> [bs, sl-1]
        input_tensor[:, :, self.x_dims + self.y_dims] = \
            self.n_batch_norm(input_tensor[:, :, self.x_dims + self.y_dims])

        input_tensor = input_tensor.nan_to_num()
        dists = dists.nan_to_num()
        dists[[target_position[0], target_position[1]]] = torch.nan
        dists = dists[~dists.isnan()].view(batch_size, -1)   # -> [bs, sl-1]

        central_x = input_tensor[[target_position[0], target_position[1]]][:, :-1].clone()   # [bs, fd-1]
        central_x = central_x.unsqueeze(1)   # [bs, fd-1] -> [bs, 1, fd-1]

        # Encoder
        encoding_tensor, attn_weights = self.encoder(
            input_tensor=input_tensor,
            mask=mask,
            dists=dists
        )   # -> [bs, 1, dm]

        # -> [bs, 1, dm+fd-1]
        encoding_tensor = torch.concat((encoding_tensor, central_x), dim=-1)   # res connection of central x.
        pred = self.decoder(encoding_tensor=encoding_tensor)

        if get_attn_score:
            return pred, attn_weights
        else:
            return pred
