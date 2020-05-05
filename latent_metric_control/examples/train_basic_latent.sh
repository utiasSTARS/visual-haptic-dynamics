device="cpu"
dataset="/Users/oliver/Datasets/pendulum-srl-sim/pendulum64_total_2048_traj_16_repeat_2_with_angle_train.pkl"
comment=('traj32_base_latent')
storage_base_path="/Users/oliver/latent-metric-control/latent_metric_control/saved_models/${comment}/"

n_batches=(32)
learning_rates=(3e-4)
batch_norms=('False')
bi_directionals=('False')
weight_inits=('custom')
Ks=(15)
rnn_nets=('lstm')
n_epochs=(4096)
opt=('adam')
enc_dec_nets=('cnn')
debug=('True')
nl=('relu')
traj_len=(16)

for n in {1..1}; do
    lam_rec=1.00
    lam_kl=1.00
    opt_vae_epoch=0
    opt_vae_base_epoch=2048
    for batch_norm in ${batch_norms[@]}; do
        for K in ${Ks[@]}; do
            for weight_init in ${weight_inits[@]}; do
                for n_batch in ${n_batches[@]}; do
                    for rnn_net in ${rnn_nets[@]}; do
                        for enc_dec_net in ${enc_dec_nets[@]}; do
                            for lr in ${learning_rates[@]}; do
                                for bi_directional in ${bi_directionals[@]}; do
                                    for n_epoch in ${n_epochs[@]}; do
                                        python3 ../train.py \
                                                            --K $K \
                                                            --dim_u 1 \
                                                            --dim_z 3 \
                                                            --dim_x "1,64,64" \
                                                            --n_worker 4 \
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
                                                            --fc_hidden_size 128 \
                                                            --rnn_hidden_size 128 \
                                                            --use_bidirectional $bi_directional \
                                                            --use_batch_norm $batch_norm \
                                                            --enc_dec_net $enc_dec_net \
                                                            --opt_vae_epochs $opt_vae_epoch \
                                                            --opt_vae_base_epochs $opt_vae_base_epoch \
                                                            --opt $opt \
                                                            --rnn_net $rnn_net \
                                                            --task "pendulum64" \
                                                            --val_split 0 \
                                                            --non_linearity $nl \
                                                            --traj_len $traj_len 
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