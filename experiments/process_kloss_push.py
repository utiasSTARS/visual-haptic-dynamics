import pickle as pkl
import os
import argparse
import gzip
import numpy as np
from PIL import Image
import warnings
import re

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pkl.load(f, fix_imports=True, encoding="latin1")
        return loaded_object

def save_zipped_pickle(obj, filename):
    with gzip.open(filename, 'wb') as f:
        pkl.dump(obj, f)

def sanitize_data(
    data_dir, 
    min_tr=3.0, 
    min_rot=0.30, 
    min_length=50,
    total_trajs=1600,
    surfaces=['delrin', 'plywood', 'pu', 'abs'], 
    objects=['ellip1', 'ellip2', 'ellip3']):

    data_sources = 0
    datapaths = []
    for s in surfaces:
        for o in objects:
            path = os.path.join(data_dir, s, o)
            if not os.path.exists(path):
                continue
            datapaths.append(path)
            data_sources += 1

    # Split dataset uniformly 
    traj_per_category = total_trajs / data_sources
    dataset = []

    ii = 0
    jj = 0
    for datapath in datapaths:
        filenames = [os.path.join(datapath, f) for f in os.listdir(datapath)]
        added_traj = 0
        for filename in filenames:
            trajectory = load_zipped_pickle(filename)

            if trajectory["image"].shape[0] < min_length:
                continue

            #XXX: Assume small angle change so just take difference of angle
            pose = trajectory["object"][:(min_length * 10)]
            pose_offset = np.roll(pose, shift=1, axis=0)
            pose_offset[0] = pose[0]
            tr_diff = np.abs(pose[:, :2] - pose_offset[:, :2])
            rot_diff = np.abs(pose[:, 2] - pose_offset[:, 2])
            total_tr_diff = np.sum(np.sqrt(np.sum((pose[:, :2] - pose_offset[:, :2])**2, axis=-1)) * 100)
            total_rot_diff = np.sum(rot_diff)

            # If a jump more than ~5 degrees happens at 180hz -> orientation error
            if (rot_diff > 0.0875).any():
                continue

            # If a translation jump of more than 0.5 cm happens...
            if ((tr_diff * 100) > 0.5).any():
                continue
            
            if total_tr_diff > min_tr or total_rot_diff > min_rot:
                dataset.append(trajectory)
                added_traj += 1

            if added_traj >= traj_per_category:
                break
    return dataset

def trim_data(dataset, length):
    # assert 
    for data in dataset:
        data["object"] = data["object"][:(length * 10)]
        data["force"] = data["force"][:(length * 10)]
        data["tip"] = data["tip"][:(length * 10)]
        data["image"] = data["image"][1:(length + 1)]
    return dataset

def package_data(dataset):
    n = len(dataset)
    l = dataset[0]["image"].shape[0]

    packaged_dataset = {
        "img": 9999 * np.ones((n, l, 64, 64, 3)),
        "ft": 9999 * np.ones((n, l, 10, 3)),
        "arm": 9999 * np.ones((n, l, 10, 2)),
        "action": 9999 * np.ones((n, l, 2)),
        "gt_plate_pos": 9999 * np.ones((n, l, 10, 3))
    }

    for ii, data in enumerate(dataset):
        packaged_dataset["img"][ii] = data["image"]
        for jj in range(l):
            idx_i = jj * 10
            idx_f = (jj + 1) * 10
            packaged_dataset["ft"][ii][jj] = data["force"][idx_i:idx_f]
            packaged_dataset["arm"][ii][jj] = data["tip"][idx_i:idx_f]
            packaged_dataset["action"][ii][jj] = data["tip"][idx_f-1] - data["tip"][idx_i]
            packaged_dataset["gt_plate_pos"][ii][jj] = data["object"][idx_i:idx_f]
    
    for k, v in packaged_dataset.items():
        if (v == 9999).any():
            warnings.warn(f"Unfilled value detected in processed dataset for {k}")

    return packaged_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='', help='Base directory to load processed data')
    parser.add_argument('--save_dir', default="./data/datasets/mit_push/", help='Base directory to save processed data')
    args = parser.parse_args()

    dataset = sanitize_data(
        args.data_dir, 
        min_tr=2.5, 
        min_rot=0.5,
        min_length=49,
        total_trajs=4096,
        surfaces=['delrin', 'plywood', 'pu', 'abs'], 
        objects=['ellip1', 'ellip2', 'ellip3']
    )
    dataset = trim_data(dataset, length=48)
    processed_dataset = package_data(dataset)

    for k, v in processed_dataset.items():
        print(k, v.shape)

    save_zipped_pickle(
        processed_dataset, 
        os.path.join(args.save_dir, "rng-initial_min-tr2.5_min-rot0.5_len48.pkl")
    )