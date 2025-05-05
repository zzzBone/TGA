_base_ = [
    './tga_ricegrip.py'
]

# Custom model
num_abs_token=2
model = dict(
    backbone=dict(
        num_abs_token=num_abs_token,),)

# Custom dataset
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        env_cfg=dict(num_abs_token=num_abs_token,)),
    val=dict(
        env_cfg=dict(num_abs_token=num_abs_token,)),
    test=dict(
        env_cfg=dict(num_abs_token=num_abs_token,)),)