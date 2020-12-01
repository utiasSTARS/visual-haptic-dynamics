device="cuda:1"

# dataset="/home/olimoyo/visual-haptic-dynamics/experiments/data/datasets/visual_haptic_2D_len16_withGT_3D9E4376CF4746EEA20DCD520218038D.pkl"
dataset="/home/olimoyo/visual-haptic-dynamics/experiments/data/datasets/visual_haptic_2D_len16_osc_withGT_8C12919B740845539C0E75B5CBAF7965.pkl"
storage_base_path="/home/olimoyo/visual-haptic-dynamics/saved_models/monolith/"

n_batches=(32)
learning_rates=(3e-4)
batch_norms=('True')
bi_directionals=('False')
weight_inits=('default')
Ks=(15)
rnn_nets=('gru')
dyn_nets=('nonlinear')
n_epochs=(1024)
opt=('rmsprop')
opt_vae_base_epochs=(0)
opt_n_step_pred_epochs=(4096)
debug=('True')
nl=('relu')
frame_stack=(1)
val_split=(0.1)
lam_rec=(0.95)
lam_kl=(0.80)
n_checkpoint_epoch=(3)
task="push64vh"
comment="${task}_checkpoint_test"
context_modality="joint"
context="none"
use_context_frame_stack=('False')
train_initial_hidden=('True')
fc_hidden_size=(256)
rnn_hidden_size=(256)

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
                                                --train_initial_hidden $train_initial_hidden \
                                                --context_modality $context_modality \
                                                --use_context_frame_stack $use_context_frame_stack \
                                                --context $context \
                                                --K $K \
                                                --dim_u 2 \
                                                --dim_z 16 \
                                                --dim_x "1,64,64" \
                                                --n_worker 0 \
                                                --use_binary_ce "False" \
                                                --n_epoch $n_epoch \
                                                --n_batch $n_batch \
                                                --debug $debug \
                                                --comment $comment \
                                                --device $device \
                                                --lr $lr \
                                                --weight_init $weight_init \
                                                --dataset $dataset \
                                                --lam_rec $lam_rec \
                                                --lam_kl $lam_kl \
                                                --storage_base_path $storage_base_path \
                                                --fc_hidden_size $fc_hidden_size \
                                                --rnn_hidden_size $rnn_hidden_size \
                                                --use_bidirectional $bi_directional \
                                                --use_batch_norm $batch_norm \
                                                --opt_vae_epochs 0 \
                                                --opt_vae_base_epochs $opt_vae_base_epoch \
                                                --opt_n_step_pred_epochs $opt_n_step_pred_epochs \
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