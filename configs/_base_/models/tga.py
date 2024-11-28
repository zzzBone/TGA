model = dict(
    type="ParticleSimulator",
    backbone=dict(
        type="TGA",
        attr_dim=6,
        state_dim=6,
        position_dim=3,
        embed_dims=(64, 64, 128),
        hidden_dims=(24, 36, 48),
        num_heads=(8, 8, 8),
        num_encoder_layers=3,
        dropout=(0, 0.1, 0.1, 0.2),
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
