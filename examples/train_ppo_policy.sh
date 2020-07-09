python ../experiments/train_ppo_policy.py \
--env_name ThingReacher2D-v0 \
--is_render False  \
--solved_reward -5 \
--device cpu

# python ../experiments/train_ppo_policy.py \
# --env_name ThingVisualReacher2D-v0 \
# --is_render False \
# --solved_reward -5 \
# --architecture cnn \
# --device cuda:0 \
# --max_episodes 1000000 \
# --random_seed 333344444