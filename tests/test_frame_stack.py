import os, sys, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from utils import frame_stack
import torch
import numpy as np

def test_shape(frames=1):
    img = torch.zeros((32, 15, 3, 4, 4))
    img_shape = img.shape 
    print(f"Original image shape: {img_shape}")
    stacked = frame_stack(img, frames=frames)
    print(f"Stacking {frames} frames image shape: {stacked.shape}")
    assert stacked.shape == (img_shape[0], 
                                img_shape[1] - frames,
                                img_shape[2] * (frames + 1),
                                img_shape[3],
                                img_shape[4])

def test_order():
    img = torch.zeros((32, 4, 1, 3, 3))
    for ii in range(4):
        img[:, ii, 0] = (ii + 1) * torch.ones((3,3))

    print(f"Original trajectory: {img[0, :, 0, :, :]}")

    stacked = frame_stack(img, frames=1)

    # Convert to numpy for negative stride... :(
    np_img = np.array(img)
    np_stacked = np.array(stacked)
    print(f"Stacked images at index 0: {stacked[0, 0, :, :, :]}")
    assert (np_img[0, 0:2, 0, :, :][::-1] == np_stacked[0, 0, :, :, :]).all()
    print(f"Stacked images at index 1: {stacked[0, 1, :, :, :]}")
    assert (np_img[0, 1:3, 0, :, :][::-1] == np_stacked[0, 1, :, :, :]).all()
    print(f"Stacked images at index 2: {stacked[0, 2, :, :, :]}")
    assert (np_img[0, 2:4, 0, :, :][::-1] == np_stacked[0, 2, :, :, :]).all()

if __name__=="__main__":
    # test_shape(1)
    # test_shape(2)
    # test_shape(3)
    test_order()