device="cpu"

dataset="/Users/oliver/visual-haptic-dynamics/experiments/data/datasets/vha/vha1_2D_len16_oscxy_withGT_0B7AB071F98942578ABDA66879290F2F.pkl"
# dataset1="/home/olimoyo/visual-haptic-dynamics/experiments/data/datasets/vha2_2D_len16_oscxy_withGT_3502DE81F7C343FB8B57FA92FDECF4DA.pkl"
# dataset2="/home/olimoyo/visual-haptic-dynamics/experiments/data/datasets/vha3_2D_len16_oscxy_withGT_5DB32B21A6AA4E5892D2F6B8F40EF9E6.pkl"
# dataset_mit="/home/olimoyo/visual-haptic-dynamics/experiments/data/datasets/mit_push/rng-initial_min-tr2.5_min-rot0.5_len48.pkl"

storage_base_path="/Users/oliver/visual-haptic-dynamics/saved_models/obelisk/"

n_batches=(32)
learning_rates=(3e-4)
batch_norms=('False')
weight_norm=('True')
bi_directionals=('False')
weight_inits=('default')
Ks=(15)
dyn_nets=('linearmix')
n_epochs=(3072)
opt=('adam')
opt_vae_base_epochs=(0)
opt_n_step_pred_epochs=(0)
n_annealing_epoch=(0)
debug=('True')
nl=('elu')
frame_stack=(1)
val_split=(0.1)
lam_rec=(1.00)
lam_kl=(1.00)
n_checkpoint_epoch=(64)
task="push64vh"
comment="${task}_prior_expert"
context_modality="joint"
reconstruct_context=('True')
inference_network="none"
use_context_frame_stack=('False')
train_initial_hidden=('False')
fc_hidden_size=(256)
rnn_hidden_size=(256)
use_scheduler=('False')
learn_uncertainty=('True')
ft_normalization=(100.0)
dim_arm=(6)
dim_ft=(6)
context_seq_len=(32)
# ft_normalization=(1.0)
# dim_arm=(2)
# dim_ft=(3)
# context_seq_len=(10)
use_prior_expert=("True")

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
                                                --use_prior_expert $use_prior_expert \
                                                --context_seq_len $context_seq_len \
                                                --dim_arm $dim_arm \
                                                --dim_ft $dim_ft \
                                                --ft_normalization $ft_normalization \
                                                --n_annealing_epoch $n_annealing_epoch \
                                                --learn_uncertainty $learn_uncertainty \
                                                --use_scheduler $use_scheduler \
                                                --reconstruct_context $reconstruct_context \
                                                --train_initial_hidden $train_initial_hidden \
                                                --context_modality $context_modality \
                                                --use_context_frame_stack $use_context_frame_stack \
                                                --inference_network $inference_network \
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
                                                --use_weight_norm $weight_norm \
                                                --opt_vae_epochs 0 \
                                                --opt_vae_base_epochs $opt_vae_base_epoch \
                                                --opt_n_step_pred_epochs $opt_n_step_pred_epochs 3072 \
                                                --opt $opt \
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