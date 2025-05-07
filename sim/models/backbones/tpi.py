import math
import torch
import torch.nn as nn
from mmcv.cnn import  build_norm_layer

import torch.nn.functional as F
from ..builder import BACKBONES
from .base_backbone import BaseBackbone
from sim.models.utils.feedforward_networks import FFN


class TPIMultiheadAttention(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super().__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim)
        
    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        # 修改这里的注意力分数计算逻辑
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        # 获取投影权重和偏置
        in_proj_weight = self.in_proj_weight
        in_proj_bias = self.in_proj_bias
        
        # 分割Q、K、V的投影权重
        head_dim = embed_dim // self.num_heads
        q_proj_weight = in_proj_weight[:embed_dim, :]
        k_proj_weight = in_proj_weight[embed_dim:embed_dim*2, :]
        v_proj_weight = in_proj_weight[embed_dim*2:, :]
        
        # 分割偏置（如果有）
        if in_proj_bias is not None:
            q_proj_bias = in_proj_bias[:embed_dim]
            k_proj_bias = in_proj_bias[embed_dim:embed_dim*2]
            v_proj_bias = in_proj_bias[embed_dim*2:]
        else:
            q_proj_bias = k_proj_bias = v_proj_bias = None
            
        # 计算Q、K、V
        q = F.linear(query, q_proj_weight, q_proj_bias)
        k = F.linear(key, k_proj_weight, k_proj_bias)
        v = F.linear(value, v_proj_weight, v_proj_bias)
    
        
        # 调整形状为(num_heads * batch_size) x seq_len x head_dim
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        
        # 计算注意力分数 - 这是你可以自定义的部分
        # 原始实现是: attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        # 你可以修改为自定义的计算方式，例如:
        attn_output_weights = self.custom_attention_score(q, k)
        
        # 缩放
        attn_output_weights = attn_output_weights / (self.head_dim ** 0.5)
        
        # 应用注意力掩码
        if attn_mask is not None:
            attn_output_weights += attn_mask
            
        # 应用key_padding_mask
        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, -1)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_output_weights = attn_output_weights.view(bsz * self.num_heads, tgt_len, -1)
        
        # 计算softmax
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)
        
        # 计算输出
        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        
        if need_weights:
            return attn_output, attn_output_weights
        else:
            return attn_output, None
    
    def custom_attention_score(self, Q, K):
        # 在这里实现你的自定义注意力分数计算
        # 例如，你可以添加一个可学习的温度参数
        # temperature = nn.Parameter(torch.ones(1)  # 可学习的温度参数
        dot_product = torch.matmul(Q, K.transpose(-1, -2))  # [batch, num_head, len, len]
        Q = (Q ** 2).sum(dim=-1, keepdim=True)  # [batch, num_head, len, 1]
        K = (K ** 2).sum(dim=-1, keepdim=True)  # [batch, num_head, len, 1]
        scores = -torch.sqrt(Q + K.transpose(-1, -2) - 2 * dot_product)  # [batch, num_head, len, len]
        return scores

class TPIEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)
        
        # 替换原有的自注意力机制
        self.self_attn = TPIMultiheadAttention(d_model, nhead, dropout=dropout)
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # 其余部分保持不变
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TPIEncoder(nn.Module):
    def __init__(self, num_layers, embed_dims,
                 num_heads,):
        super(TPIEncoder, self).__init__()
        self.embed_dim = embed_dims

        # A stack of transformer encoder layers
        self.tpi_layers = nn.ModuleList([
            TPIEncoderLayer(d_model = embed_dims, nhead=num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        # x shape: [n, b, c] --> batch_size, channels, num_features

        # Pass through transformer encoder layers
        for layer in self.tpi_layers:
            x = layer(x)  # Apply each transformer encoder layer

        return x

@BACKBONES.register_module()
class TPI(BaseBackbone):
    """Implements the simulation transformer.
    """

    def __init__(self,
                 attr_dim,
                 state_dim,
                 position_dim,
                 embed_dims,
                 num_heads=8,
                 num_encoder_layers=4,
                 dropout=0.0,
                 order=('selfattn', 'norm', 'ffn', 'norm'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 norm_eval=False, num_abs_token=0, **kwargs):
        super(TPI, self).__init__()
        self.attr_dim = attr_dim
        self.state_dim = state_dim
        self.position_dim = position_dim
        self.embed_dims = embed_dims
        self.num_abs_token = num_abs_token

        self.norm_eval = norm_eval
        self.input_projection = FFN(
            [attr_dim + state_dim, embed_dims],
            final_act=True, bias=True)

        self.rs_weight = nn.Parameter(torch.Tensor(embed_dims, attr_dim + state_dim))
        nn.init.kaiming_uniform_(self.rs_weight, a=math.sqrt(5))

        if norm_cfg is not None:
            self.particle_norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.particle_norm = nn.Identity()

        if self.num_abs_token > 0:
            assert self.num_abs_token == 2
            self.abs_token = nn.Parameter(torch.Tensor(self.num_abs_token, attr_dim + state_dim))
            nn.init.zeros_(self.abs_token)

        self.encoder = TPIEncoder(num_encoder_layers, embed_dims,
                                          num_heads)

    def forward(self, attr, state, fluid_mask, rigid_mask, output_mask, attn_mask, **kwargs):
        """Forward function for `TPI`.
        """
        x = torch.cat([attr.squeeze(1).transpose(-1, -2), state.squeeze(1).transpose(-1, -2)], dim=-1)
        if self.num_abs_token > 0:
            x[:, -self.num_abs_token:] = self.abs_token
            abs_mask = torch.cat([rigid_mask, fluid_mask], dim=1).unsqueeze(1)
            # pad_mask[:, :, -self.num_abs_token:] = ~(abs_mask.sum(dim=-1) > 0)
            abs_mask = ~(abs_mask.bool())
            attn_mask[:, :, -self.num_abs_token:] = abs_mask
            attn_mask[:, :, :, -self.num_abs_token:] = abs_mask.transpose(-1, -2)

        x = self.input_projection(x)

        x = x.permute(1, 0, 2)  # [bs, n_p, c] -> [n_p, bs, c]

        x_enc = self.encoder(x)
        x_enc = x_enc.permute(1, 0, 2)
        return x_enc

    def train(self, mode=True):
        super(TPI, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
