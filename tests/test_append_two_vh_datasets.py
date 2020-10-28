import os, sys, time
os.sys.path.insert(0, "..")
from datasets import VisualHaptic

def test_append_two_cache():
    ds1 = "/Users/oliver/visual-haptic-dynamics/experiments/data/datasets/visual_haptic_2D_len16_osc_withGT_8C12919B740845539C0E75B5CBAF7965.pkl"
    ds2 = "/Users/oliver/visual-haptic-dynamics/experiments/data/datasets/visual_haptic_2D_len16_withGT_3D9E4376CF4746EEA20DCD520218038D.pkl"
    
    dataset = VisualHaptic(
        ds1,
        rgb=True
    )
    dataset.append_cache(ds2)
    print(len(dataset))
    print(dataset.data["img"].shape)

if __name__ == "__main__":
    test_append_two_cache()