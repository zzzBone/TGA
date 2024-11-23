import math
import torch
import torch.nn as nn
from mmcv.cnn import  build_norm_layer

from ..builder import BACKBONES
from .base_backbone import BaseBackbone
from sim.models.utils.feedforward_networks import FFN


class TransformerEncoder(nn.Module):
    def __init__(self, c, num_layers, embed_dims,
                 num_heads,
                 dropout, ):
        super(TransformerEncoder, self).__init__()
        self.embed_dim = embed_dims

        # Linear layer to project input channels 'c' to 'd_model'
        self.input_projection = nn.Linear(c, embed_dims)

        # MultiheadAttention layer
        self.attn_layer = nn.MultiheadAttention(embed_dim=embed_dims, num_heads=num_heads)

        # Feed forward layer (optional for transformer block)
        self.ff_layer = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims)
        )

        # A stack of transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dims, nhead=num_heads)
            for _ in range(num_layers)
        ])

        # Final projection to output size 'm'
        self.output_projection = nn.Linear(embed_dims, embed_dims)

    def forward(self, x):
        # x shape: [b, c, n] --> batch_size, channels, num_features
        b, c, n = x.shape

        # First, project input channels 'c' to 'embed_dim'
        x = self.input_projection(x.transpose(1, 2))  # Shape: [b, n, embed_dim]

        # Pass through transformer encoder layers
        for layer in self.transformer_layers:
            x = layer(x)  # Apply each transformer encoder layer

        # Final projection to output shape [b, m, n]
        x = self.output_projection(x.transpose(1, 2))  # Shape: [b, m, n]
        return x


@BACKBONES.register_module()
class TRANS(BaseBackbone):
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
        super(TRANS, self).__init__()
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

        self.encoder = TransformerEncoder(attr_dim + state_dim, num_encoder_layers, embed_dims,
                                          num_heads,
                                          dropout)

    def forward(self, attr, state, fluid_mask, rigid_mask, output_mask, attn_mask, **kwargs):
        """Forward function for `TRANS`.
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

        x_enc = self.encoder(x, mask=attn_mask)
        x_enc = x_enc.permute(1, 0, 2)
        return x_enc

    def train(self, mode=True):
        super(TRANS, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
