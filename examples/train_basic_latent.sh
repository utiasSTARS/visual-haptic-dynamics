device="cuda:0"
dataset="/media/m2-drive/datasets/pendulum-srl-sim/pendulum16_total_2048_traj_16_repeat_2_with_angle_train.pkl"
storage_base_path="/home/olimoyo/latent-metric-control/saved_models/"

# dataset="/Users/oliver/Datasets/pendulum-srl-sim/pendulum64_total_2048_traj_16_repeat_2_with_angle_train.pkl"
# storage_base_path="/Users/oliver/latent-metric-control/saved_models"

n_batches=(32)
learning_rates=(3e-4)
batch_norms=('True')
bi_directionals=('False')
weight_inits=('custom')
Ks=(15)
rnn_nets=('lstm')
dyn_nets=('linearmix')
n_epochs=(4096)
opt=('adam')
enc_dec_nets=('fcn')
opt_vae_base_epochs=(1024)
debug=('True')
nl=('relu')
traj_len=(31)
frame_stack=(1)
val_split=(0)
lam_rec=(0.95)
lam_kl=(0.80)
n_checkpoint_epoch=(1024)

for n in {1..1}; do
    for dyn_net in ${dyn_nets[@]}; do
        for opt_vae_base_epoch in ${opt_vae_base_epochs[@]}; do
            for batch_norm in ${batch_norms[@]}; do
                for K in ${Ks[@]}; do
                    for weight_init in ${weight_inits[@]}; do
                        for n_batch in ${n_batches[@]}; do
                            for rnn_net in ${rnn_nets[@]}; do
                                for enc_dec_net in ${enc_dec_nets[@]}; do
                                    for lr in ${learning_rates[@]}; do
                                        for bi_directional in ${bi_directionals[@]}; do
                                            for n_epoch in ${n_epochs[@]}; do
                                                python ../train.py \
                                                                    --K $K \
                                                                    --dim_u 1 \
                                                                    --dim_z 3 \
                                                                    --dim_x "1,16,16" \
                                                                    --n_worker 8 \
                                                                    --use_binary_ce "False" \
                                                                    --n_epoch $n_epoch \
                                                                    --n_batch $n_batch \
                                                                    --debug $debug \
                                                                    --comment "encdecnet-${enc_dec_net}_lr-${lr}_n_batch-${n_batch}_weightinit-${weight_init}_traj-${traj_len}_bn-${batch_norm}_dyntype-${dyn_net}_dynnet-${rnn_net}_framestacks-${frame_stack}_optvaebaseepochs-${opt_vae_base_epoch}_lamrec-${lam_rec}_lamkl-${lam_kl}" \
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
                                                                    --enc_dec_net $enc_dec_net \
                                                                    --opt_vae_epochs 0 \
                                                                    --opt_vae_base_epochs $opt_vae_base_epoch \
                                                                    --opt $opt \
                                                                    --rnn_net $rnn_net \
                                                                    --dyn_net $dyn_net \
                                                                    --task "pendulum16" \
                                                                    --val_split $val_split \
                                                                    --non_linearity $nl \
                                                                    --traj_len $traj_len \
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
done