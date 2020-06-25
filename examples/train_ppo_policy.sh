# python ../experiments/train_ppo_policy.py \
# --env_name ThingReacher2D-v0 \
# --is_render False  \
# --solved_reward -5 \
# --update_timestep 3900

python ../experiments/train_ppo_policy.py \
--env_name ThingVisualReacher2D-v0 \
--is_render False \
--solved_reward -5 \
--update_timestep 300 \
--architecture cnn