#!/bin/bash
python /home/olimoyo/latent-metric-control/train.py \
--comment basez2_softplus \
--dataset /media/m2-drive/datasets/pendulum-srl-sim/pendulum64_total_2048_traj_16_repeat_2_with_angle_train.pkl \
--storage_base_path /home/olimoyo/latent-metric-control/saved_models \
--dim_z 2 \
--dim_x "1, 64, 64" \
--non_linearity "softplus" \