device="cuda:0"

dataset="/home/olimoyo/visual-haptic-dynamics/experiments/data/datasets/visual_haptic_2D_osc_len16_withGT_11A15E63D82B4CFE85B1B56DF8F07893.pkl"
# dataset="/home/olimoyo/visual-haptic-dynamics/experiments/data/datasets/visual_haptic_2D_9985E1798153438E880A8AD60B9146FE.pkl"
storage_base_path="/home/olimoyo/visual-haptic-dynamics/saved_models/"

n_batches=(32)
learning_rates=(3e-4)
batch_norms=('True')
bi_directionals=('False')
weight_inits=('custom')
Ks=(15)
rnn_nets=('gru')
dyn_nets=('linearmix')
n_epochs=(4096)
opt=('adam')
opt_vae_base_epochs=(1024)
debug=('False')
nl=('relu')
frame_stack=(1)
val_split=(0)
lam_rec=(0.95)
lam_kl=(0.80)
n_checkpoint_epoch=(64)
task="push64vh"
use_img_enc="True"
use_haptic_enc="False"
use_arm_enc="False"
use_joint_enc="False"
use_haptic_dec="False"
use_arm_dec="False"
tcn_channels="128,64,32"
n_step_pred=1

for n in {1..1}; do
    for dyn_net in ${dyn_nets[@]}; do
        for opt_vae_base_epoch in ${opt_vae_base_epochs[@]}; do
            for batch_norm in ${batch_norms[@]}; do
                for K in ${Ks[@]}; do
                    for weight_init in ${weight_inits[@]}; do
                        for n_batch in ${n_batches[@]}; do
                            for rnn_net in ${rnn_nets[@]}; do
                                for lr in ${learning_rates[@]}; do
                                    for bi_directional in ${bi_directionals[@]}; do
                                        for n_epoch in ${n_epochs[@]}; do
                                            python ../train.py \
                                                                --use_arm_dec $use_arm_dec \
                                                                --use_haptic_dec $use_haptic_dec \
                                                                --use_img_enc $use_img_enc \
                                                                --use_haptic_enc $use_haptic_enc \
                                                                --use_arm_enc $use_arm_enc \
                                                                --use_joint_enc $use_joint_enc \
                                                                --K $K \
                                                                --n_step_pred $n_step_pred \
                                                                --dim_u 2 \
                                                                --dim_z 16 \
                                                                --dim_x "1,64,64" \
                                                                --tcn_channels $tcn_channels \
                                                                --n_worker 8 \
                                                                --use_binary_ce "False" \
                                                                --n_epoch $n_epoch \
                                                                --n_batch $n_batch \
                                                                --debug $debug \
                                                                --comment "${task}_lm_vha" \
                                                                --device $device \
                                                                --lr $lr \
                                                                --weight_init $weight_init \
                                                                --dataset $dataset \
                                                                --lam_rec $lam_rec \
                                                                --lam_kl $lam_kl \
                                                                --storage_base_path $storage_base_path \
                                                                --fc_hidden_size 256 \
                                                                --rnn_hidden_size 256 \
                                                                --use_bidirectional $bi_directional \
                                                                --use_batch_norm $batch_norm \
                                                                --opt_vae_epochs 0 \
                                                                --opt_vae_base_epochs $opt_vae_base_epoch \
                                                                --opt $opt \
                                                                --rnn_net $rnn_net \
                                                                --dyn_net $dyn_net \
                                                                --task $task \
                                                                --val_split $val_split \
                                                                --non_linearity $nl \
                                                                --frame_stacks $frame_stack \
                                                                --n_checkpoint_epoch $n_checkpoint_epoch
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done