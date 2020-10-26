python ../experiments/control_experiment.py \
--model_path /Users/oliver/visual-haptic-dynamics/saved_models/nstep2_z16_net512_l0_lm_osc_v_r0.95_kl0.80_lr3e4 \
--H 6 \
--mpc_opt cvxopt \
--device cpu \
--n_episodes 1024 \
--n_train_episodes 16 \
--n_test_episodes 1 \
--n_epochs 128 \
--n_checkpoint_episodes 4 \
--opt adam \
--render False \
--exploration_noise_var 0.2 \
--debug False