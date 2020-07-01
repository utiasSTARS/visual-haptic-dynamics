import os, sys
os.sys.path.insert(0, "..")

from datasets import VisualHaptic

dataset = VisualHaptic(
        "/Users/oliver/visual-haptic-dynamics/experiments/data/datasets/visual_haptic_1D_B20B033AC27B4EC6AA690C653AC0AE70.pkl",
        img_shape=(1,64,64)
    )

