python tools/train.py configs/tie/tie_boxbath.py --work_dir work_dirs/tie_boxbath;
python tools/predict_rollout.py configs/tie/tie_boxbath.py predict/tie_boxbath --checkpoint work_dirs/tie_boxbath/epoch_5.pth;
python tools/train.py configs/tie/tie_fluidshake.py --work_dir work_dirs/tie_fluidshake;
python tools/predict_rollout.py configs/tie/tie_fluidshake.py predict/tie_fluidshake --checkpoint work_dirs/tie_fluidshake/epoch_5.pth;
python tools/train.py configs/tie/tie_ricegrip.py --work_dir work_dirs/tie_ricegrip;
python tools/predict_rollout.py configs/tie/tie_ricegrip.py predict/tie_ricegrip --checkpoint work_dirs/tie_ricegrip/epoch_20.pth;
python tools/train.py configs/tie/tie_fluidfall.py --work_dir work_dirs/tie_fluidfall;
python tools/predict_rollout.py configs/tie/tie_fluidfall.py predict/tie_fluidfall --checkpoint work_dirs/tie_fluidfall/epoch_13.pth;