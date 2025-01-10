model = dict(
    type="ParticleSimulator",
    backbone=dict(
        type="TGA",
        attr_dim=6,
        state_dim=6,
        position_dim=3,
        embed_dim=128,
        attn_dims=(128, 128),
        attn_num_heads=(8, 8),
        gat_hidden_dims=(128, 128),
        gat_num_heads=(8, 8),
        num_layers=1,
        dropouts=(0.2, 0.2),
        output_dim=128,
    ),
    head=dict(
        type="ParticleHead",
        # in_channels is nf_effect
        in_channels=128,
        # out_channels is position_dim
        out_channels=3,
        seperate=False,
        rotation_dim=0,
        weighted=False,
        loss=dict(type="MSELoss", loss_weight=1.0),
    ),
)
