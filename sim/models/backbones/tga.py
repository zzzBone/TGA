import math
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer

from ..builder import BACKBONES
from .base_backbone import BaseBackbone
from sim.models.utils.feedforward_networks import FFN

import torch
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv, EGConv, SuperGATConv, GraphSAGE, GATConv
from torch_geometric.data import Data, Batch

import numpy as np

from .transformer_implicit_edges import TransformerEncoder


class FCConvToBinary(nn.Module):
    def __init__(self, in_channels):
        super(FCConvToBinary, self).__init__()
        # 使用1x1卷积将 c 个通道映射为 1 个通道
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        
    def forward(self, x):
        # 输入 x 的形状是 [batch_size, c, w, h]
        # 通过1x1卷积得到输出 [batch_size, 1, w, h]
        x = self.conv(x)
        
        # 使用 Sigmoid 激活函数将输出映射到 [0, 1]
        x = torch.sigmoid(x)
        
        # 使用阈值化操作转换为 [0, 1] 的二进制矩阵
        x = torch.round(x)  # 将小于 0.5 的值置为 0，大于等于 0.5 的值置为 1
        
        return x

class TGAMultiHeadAttion(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, dropout=0.0, percent=0.9):
        super(TGAMultiHeadAttion, self).__init__()
        assert (
            output_dim % num_heads == 0
        ), "Embedding size must be divisible by number of heads"

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads  # 每个头的维度
        self.percent = percent

        # 定义线性变换层，用于计算查询（Q），键（K），值（V）
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)

        self.ln = nn.LayerNorm(output_dim)

        # dropout层
        self.dropout = nn.Dropout(dropout)

        # self.graphConv = FCConvToBinary(num_heads)

    def forward(self, x, mask=None):
        # x: shape (batch_size, seq_len, input_size)

        batch_size, seq_len, input_size = x.shape

        # # Step 1: 线性变换得到 Q, K, V
        Q = self.query(x)  # (batch_size, seq_len, embed_size)
        Q = self.ln(Q)
        K = self.key(x)  # (batch_size, seq_len, embed_size)
        K = self.ln(K)
        V = self.value(x)  # (batch_size, seq_len, embed_size)
        V = self.ln(V)

        # Step 2: 将 Q, K, V 拆分成多个头，每个头有 head_dim 的维度
        Q = Q.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )  # (batch_size, seq_len, num_heads, head_dim)
        K = K.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )  # (batch_size, seq_len, num_heads, head_dim)
        V = V.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )  # (batch_size, seq_len, num_heads, head_dim)

        # 转换为 (batch_size, num_heads, seq_len, head_dim) 形状
        Q = Q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        V = V.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)

        dot_product = torch.matmul(Q, K.transpose(-1, -2))  # [batch, num_head, len, len]
        Q = (Q ** 2).sum(dim=-1, keepdim=True)  # [batch, num_head, len, 1]
        K = (K ** 2).sum(dim=-1, keepdim=True)  # [batch, num_head, len, 1]
        scores = -torch.sqrt(Q + K.transpose(-1, -2) - 2 * dot_product)  # [batch, num_head, len, len]

        # # Step 3: 计算每个头的注意力分数
        # scores = torch.matmul(Q, K.transpose(-2, -1)) / (
        #     self.head_dim**0.5
        # )  # (batch_size, num_heads, seq_len, seq_len)

        # Step 4: 如果有mask，则应用mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Step 5: 通过softmax归一化得到注意力权重
        attention_weights = F.softmax(
            scores, dim=-1
        )  # (batch_size, num_heads, seq_len, seq_len)
        # torch.save(scores,"/home/zbl/sim/TIE_ECCV2022/view_data/scores_fluidfall.pt")
        # Step 6: 应用dropout
        # attention_weights = self.dropout(attention_weights)

        # Step 7: 计算加权值
        output = torch.matmul(
            attention_weights, V
        )  # (batch_size, num_heads, seq_len, head_dim)

        # Step 8: 将所有头的输出拼接在一起
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.output_dim)
        )  # (batch_size, seq_len, embed_size)

        # 选择注意力分数大于threshold的边
        edge_index_list = []

        # attention_scores = 0.2 * attention_weights.mean(dim=1)[0] + 0.8 * attention_weights.max(dim=1)[0]

        # attention_scores = self.graphConv(attention_weights)
        # attention_scores = attention_scores.squeeze(1)

        # flat_score = attention_scores.flatten()
        # # threshold = min(flat_score)
        # threshold = torch.quantile(flat_score, self.percent)

        # for b in range(batch_size):
        #     # 过滤 scores
        #     # edge_mask = attention_scores[b] >= threshold
        #     edge_mask = attention_scores[b] == 1

        #     # 使用 nonzero 提取符合条件的边索引
        #     edge_indices = torch.nonzero(edge_mask, as_tuple=False)  # (num_edges, 2)

        #     # 将边加入到 edge_index_list 中
        #     edge_index_list.append(edge_indices.T)  # 转置成 (2, num_edges)
        attention_scores = torch.sigmoid(scores)
        mask = attention_scores > 0.5
        for b in range(batch_size):
            edge_batch_list = []
            for h in range(self.num_heads):
                current_mask = mask[b, h]
                # 获取满足条件的坐标 (i, j)
                indices = torch.nonzero(current_mask, as_tuple=False).t()  # [2, K]
                
                # 如果没有任何元素满足条件，填充一个空的张量
                if indices.size(1) == 0:
                    indices = torch.zeros(2, 1, dtype=torch.long, device=attention_scores.device)
                
                # 将坐标添加到列表中
                edge_batch_list.append(indices)
            edge_index_list.append(edge_batch_list)

        # # 将列表转换为张量，并调整为 [B, H, 2, K] 的形状
        # max_K = max([o.size(1) for o in edge_index_list])  # 找到最大的 K
        # O = torch.zeros(batch_size, self.num_heads, 2, max_K, dtype=torch.long, device=attention_scores.device)
        
        # idx = 0
        # for b in range(batch_size):
        #     for h in range(self.num_heads):
        #         K = edge_index_list[idx].size(1)
        #         O[b, h, :, :K] = edge_index_list[idx]
        #         idx += 1
        return output, edge_index_list


class TGAGraphAttention(nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, num_heads=1, residual=False
    ):
        super(TGAGraphAttention, self).__init__()
        self.num_heads = num_heads

        self.residual = residual

        self.gat = nn.ModuleList(
            GATv2Conv((int)(in_channels/num_heads), (int)(out_channels/num_heads))
            for i in range(num_heads)
        )

        # 第一层 GAT
        # self.gat1 = GATv2Conv(in_channels, out_channels)
        # self.sage1 = GraphSAGE(in_channels, out_channels, num_layers=1)
        # 第二层 GAT
        # self.gat2 = GATConv(hidden_channels, out_channels)
        # self.sage2 = GraphSAGE(hidden_channels, out_channels, num_layers=1)

    def forward(self, batch):
        """
        x: [batch_size, num_nodes, num_features]
        edge_index: [batch_size, 2, num_edge]
        """
        output = []
        for i, gat in enumerate(self.gat):
            output.append(gat(batch[i].x, batch[i].edge_index))
        # return self.gat1(batch.x, batch.edge_index)
        return torch.cat(output, dim=-1)


class TGATransformer(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=8, dropout=0, residual=False):
        super(TGATransformer, self).__init__()
        self.residual = residual
        self.attn_layer = TGAMultiHeadAttion(
            input_dim=input_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.ffn = FFN([output_dim, output_dim], final_act=True, bias=True)
        self.ln = nn.LayerNorm(output_dim)

    def forward(self, x, mask=None):
        x_out, edge_index = self.attn_layer(x, mask)
        x_out = self.ffn(x_out)
        if self.residual:
            x_out = self.ln(x + x_out)
        else:
            x_out = self.ln(x_out)
        return x_out, edge_index


@BACKBONES.register_module()
class TGA(BaseBackbone):
    """Implements the simulation transformer."""

    def __init__(
        self,
        attr_dim,
        state_dim,
        position_dim,
        embed_dim,
        attn_dims,
        gat_hidden_dims,
        attn_num_heads,
        gat_num_heads,
        num_layers,
        dropouts,
        output_dim,
        order=('selfattn', 'norm', 'ffn', 'norm'),
        act_cfg=dict(type='ReLU', inplace=True),
        norm_cfg=dict(type="LN"),
        norm_eval=False,
        num_abs_token=0,
        **kwargs
    ):
        super(TGA, self).__init__()
        self.attr_dim = attr_dim
        self.state_dim = state_dim
        self.position_dim = position_dim
        self.embed_dim = embed_dim
        self.num_abs_token = num_abs_token
        self.num_encoder_layers = num_layers

        self.input_dim = attr_dim + state_dim

        self.attn_num_heads = attn_num_heads

        self.norm_eval = norm_eval
        self.input_projection = FFN(
            [self.input_dim, self.embed_dim], final_act=True, bias=True
        )

        self.rs_weight = nn.Parameter(torch.Tensor(embed_dim, 2 * attr_dim + 2 * state_dim))
        nn.init.kaiming_uniform_(self.rs_weight, a=math.sqrt(5))

        if norm_cfg is not None:
            self.receiver_norm = build_norm_layer(norm_cfg, embed_dim)[1]
            self.sender_norm = build_norm_layer(norm_cfg, embed_dim)[1]
            self.particle_norm = build_norm_layer(norm_cfg, embed_dim)[1]
        else:
            self.receiver_norm = nn.Identity()
            self.sender_norm = nn.Identity()
            self.particle_norm = nn.Identity()

        if self.num_abs_token > 0:
            assert self.num_abs_token == 2
            self.abs_token = nn.Parameter(torch.Tensor(self.num_abs_token, attr_dim + state_dim))
            nn.init.zeros_(self.abs_token)

        self.tie_encoder = TransformerEncoder(4, embed_dim,
                                          attn_num_heads[0],
                                          dropouts[0], order, act_cfg,
                                          norm_cfg, **kwargs)

        self.encoder = nn.ModuleList(
            TGATransformer(
                input_dim=self.embed_dim if i == 0 else attn_dims[i - 1],
                output_dim=attn_dims[i],
                num_heads=attn_num_heads[i],
                dropout=dropouts[i],
                residual=False,
            )
            for i in range(num_layers)
        )

        self.neck = nn.Linear(attn_dims[-1], self.embed_dim)
        self.ln1 = nn.LayerNorm(self.embed_dim)

        self.decoder = nn.ModuleList(
            TGAGraphAttention(
                in_channels=self.embed_dim,
                hidden_channels=gat_hidden_dims[i],
                out_channels=self.embed_dim,
                num_heads=gat_num_heads[i],
                residual=True,
            )
            for i in range(num_layers)
        )

        self.back = nn.Linear(self.embed_dim, output_dim)
        self.ln2 = nn.LayerNorm(output_dim)

    def forward(
        self, attr, state, fluid_mask, rigid_mask, output_mask, attn_mask, **kwargs
    ):
        x = torch.cat(
            [attr.squeeze(1).transpose(-1, -2), state.squeeze(1).transpose(-1, -2)],
            dim=-1,
        )
        if self.num_abs_token > 0:
            x[:, -self.num_abs_token:] = self.abs_token
            abs_mask = torch.cat([rigid_mask, fluid_mask], dim=1).unsqueeze(1)
            # pad_mask[:, :, -self.num_abs_token:] = ~(abs_mask.sum(dim=-1) > 0)
            abs_mask = ~(abs_mask.bool())
            attn_mask[:, :, -self.num_abs_token:] = abs_mask
            attn_mask[:, :, :, -self.num_abs_token:] = abs_mask.transpose(-1, -2)

        r_r_w = self.rs_weight[:, :self.attr_dim + self.state_dim]
        r_s_w = self.rs_weight[:, self.attr_dim + self.state_dim:]
        receiver_val_res = x.matmul(r_r_w.t())
        sender_val_res = x.matmul(r_s_w.t())
        receiver_val_res = receiver_val_res.permute(1, 0, 2)
        sender_val_res = sender_val_res.permute(1, 0, 2)

        x = self.input_projection(x)
        x = x.permute(1, 0, 2)  # [bs, n_p, c] -> [n_p, bs, c]
        x = self.tie_encoder(
            x,
            attn_mask=attn_mask, key_padding_mask=None, output_mask=output_mask,
            receiver_val_res=receiver_val_res, sender_val_res=sender_val_res)
        x = x.permute(1, 0, 2)
        x_out = x
        edge_index = []
        for i in range(self.num_encoder_layers):
            x_out, edge = self.encoder[i](x_out)
            edge_index.append(edge)

        x_out = self.neck(x_out)
        x_out = self.ln1(x_out)
        x_out = F.relu(x_out)
        x_out = x_out + x

        

        for i in range(self.num_encoder_layers):
            batch_size, num_nodes, num_features = x_out.shape
            batch_list = []
            split_matrices = torch.chunk(x_out, self.attn_num_heads[i], dim=-1)
            for h in range(self.attn_num_heads[i]):
                data_list = []
                for k in range(batch_size):
                    data_list.append(Data(x=split_matrices[h][k], edge_index=edge_index[i][k][h]))
                batch = Batch.from_data_list(data_list)
                batch_list.append(batch)
            x_out = self.decoder[i](batch_list)
            x_out = x_out.view(batch_size, num_nodes, -1)

        x_out = self.back(x_out)
        x_out = self.ln2(x_out)
        x_out = F.relu(x_out)
        return x_out + x

    def train(self, mode=True):
        super(TGA, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):  # type: ignore
                    m.eval()
