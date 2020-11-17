# Script taken from: https://github.com/edenton/svg

import os
import io

import numpy as np
from PIL import Image
import tensorflow as tf
import pickle

from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile

from skimage.io import imsave

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='', help='base directory to save processed data')
args = parser.parse_args()

def get_seq(dname):
    data_dir = f'{args.data_dir}/softmotion30_44k/{dname}'

    filenames = gfile.Glob(os.path.join(data_dir, '*'))
    if not filenames:
        raise RuntimeError('No data files found.')

    for f in filenames:
        k=0
        for serialized_example in tf.compat.v1.python_io.tf_record_iterator(f):
            example = tf.train.Example()
            example.ParseFromString(serialized_example)
            image_seq = []
            action_seq = []
            ee_seq = []

            for i in range(30):
                image_name = str(i) + '/image_aux1/encoded'
                action_name = str(i) + '/action'
                ee_pos_name = str(i) + '/endeffector_pos'

                # extract image
                byte_str = example.features.feature[image_name].bytes_list.value[0]
                img = Image.frombytes('RGB', (64, 64), byte_str)
                arr = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)
                image_seq.append(arr.reshape(1, 64, 64, 3)/255.)

                # extract action
                action_values = example.features.feature[action_name].float_list.value
                action = np.array([action_values[i] for i in range(len(action_values))])[None, :]
                action_seq.append(action)

                # extract ee pose
                ee_pos_values = example.features.feature[ee_pos_name].float_list.value
                ee_pos = np.array([ee_pos_values[i] for i in range(len(ee_pos_values))])[None, :]
                ee_seq.append(ee_pos)

            image_seq = np.concatenate(image_seq, axis=0)
            action_seq = np.concatenate(action_seq, axis=0)
            ee_seq = np.concatenate(ee_seq, axis=0)
            k=k+1
            yield f, k, image_seq, action_seq, ee_seq

def convert_data(dname):    
    seq_generator = get_seq(dname)
    n = 0
    while True:
        n+=1
        try:
            f, k, image_seq, action_seq, ee_seq = next(seq_generator)
        except StopIteration:
            break
        f = f.split('/')[-1]

        save_dir = f'{args.data_dir}/processed_data/{dname}/{f[:-10]}/{k}/'
        os.makedirs(save_dir, exist_ok=True)
        for i in range(len(image_seq)):
            imsave(save_dir + f'{i}.png', image_seq[i])
        with open(save_dir + 'actions.pkl', 'wb') as handle:
            pickle.dump(action_seq, handle)
        with open(save_dir + 'ee_pos.pkl', 'wb') as handle:
            pickle.dump(ee_seq, handle)
        print(f'{dname} data: {f} ({k})  ({n})')

convert_data('test')
convert_data('train')