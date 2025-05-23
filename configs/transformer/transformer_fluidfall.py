_base_ = [
    "../_base_/models/transformer.py",
    "../_base_/datasets/fluid_fall.py",
    "../_base_/schedules/adam_plateau_bs16.py",
    "../_base_/default_runtime.py",
]

find_unused_parameters = True

# Custom model
model = dict(
    backbone=dict(
        attr_dim=3,
    )
)

# Custom dataset
data = dict(
    # Batch size = num_gpu * samples_per_gpu
    samples_per_gpu=8,
    workers_per_gpu=8,
)

# Custom scheduler
runner = dict(max_epochs=13)
dist_params = dict(port="29513")

checkpoint_config = dict(interval=1)
