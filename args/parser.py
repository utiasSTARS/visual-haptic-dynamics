import argparse

from args.utils import str2bool, str2inttuple, str2tuple, str2floattuple

def parse_common_training_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    # Debug Settings
    parser.add_argument('--debug', type=str2bool, default=False, help='Debug and do not save models or log anything')

    # Experiment Settings
    parser.add_argument('--storage_base_path', type=str, required=True, help='Base path to store all training data')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for PyTorch')
    parser.add_argument('--cudnn_deterministic', type=str2bool, default=True, help='Use cudnn deterministic')
    parser.add_argument('--cudnn_benchmark', type=str2bool, default=False, help='Use cudnn benchmark')
    parser.add_argument('--task', type=str, default="push64vh", help='The task that is being trained on')
    parser.add_argument('--comment', type=str, default="None", help='Comment to describe model')
    parser.add_argument('--random_seed', type=int, default=333, help='Random seed')

    # Dataset Settings
    parser.add_argument('--dataset', type=str, required=True,  help='Name of dataset to train on')
    parser.add_argument('--val_split', type=float, default=0, help='Amount of dataset to use for validation')

    # Training Settings
    parser.add_argument('--n_epoch', type=int, default=4096, help='Number of epochs')
    parser.add_argument('--n_checkpoint_epoch', type=int, default=64, help='Save model every n epochs')
    parser.add_argument('--n_batch', type=int, default=32,  help='Batch size')
    parser.add_argument('--n_example', type=int, default=10000000,  help='Maximum samples to train from the dataset')
    parser.add_argument('--n_worker', type=int, default=16,  help='Amount of workers for dataloading.')

    # Network Settings                        
    parser.add_argument('--use_batch_norm', type=str2bool, default=True, help='Use batch normalization')
    parser.add_argument('--use_dropout', type=str2bool, default=False, help='Use dropout')
    parser.add_argument('--weight_init', choices=['custom', 'default'], default='custom',  help='Weight initialization')

    # Optimizer Settings                        
    parser.add_argument('--opt', choices=['adam', 'sgd'], default='adam',  help='Optimizer used')
    args = parser.parse_args()
    return args

def parse_vh_training_args():
    parser = argparse.ArgumentParser()

    # Network args
    parser.add_argument('--dim_u', type=int, default=2, help='Action dimension')
    parser.add_argument('--dim_z', type=int, default=16, help='Final latent state dimension')
    parser.add_argument('--use_img_enc', type=str2bool, default=True, help='Encode image data')
    parser.add_argument('--use_img_dec', type=str2bool, default=True, help='Decode image data')
    parser.add_argument('--dim_z_img', type=int, default=16, help='Latent state dimension of images')
    parser.add_argument('--use_haptic_enc', type=str2bool, default=True, help='Encode haptic data')
    parser.add_argument('--use_haptic_dec', type=str2bool, default=False, help='Decode haptic data')
    parser.add_argument('--dim_z_haptic', type=int, default=8, help='Latent state dimension of haptic')
    parser.add_argument('--use_arm_enc', type=str2bool, default=True, help='Encode arm data')
    parser.add_argument('--use_arm_dec', type=str2bool, default=False, help='Decode arm data')
    parser.add_argument('--use_joint_enc', type=str2bool, default=False, help='Encode haptic and arm data jointly')
    parser.add_argument('--dim_z_arm', type=int, default=8, help='Latent state dimension of arm')
    parser.add_argument('--dim_x', type=str2inttuple, default=(1, 64, 64), help='3-tuple image dimension (C, H, W)')
    parser.add_argument('--K', type=int, default=15, help='Number of mixture component for dynamic models')
    parser.add_argument('--fc_hidden_size', type=int, default=256, help='The number of hidden units for each linear layer')
    parser.add_argument('--rnn_hidden_size', type=int, default=256, help='The number of hidden units for each GRU or LSTM layer')
    parser.add_argument('--use_bidirectional', type=str2bool, default=False, help='Use bidirectional RNN')
    parser.add_argument('--rnn_net', choices=['gru', 'lstm'], default='gru',  help='RNN network type')
    parser.add_argument('--dyn_net', choices=['linearmix', 'linearrank1', 'nonlinear'], default='linearmix', help='Network architecture for dynamics')     
    parser.add_argument('--non_linearity', choices=['relu', 'elu', 'softplus'], default='relu', help='Activation used for decoder neural network')
    parser.add_argument('--frame_stacks', type=int, default=1, help="Number of frames to stack")
    parser.add_argument('--tcn_channels', type=str2inttuple, default=(128, 64, 32), help='3-tuple image dimension (C, H, W)')
    parser.add_argument('--n_step_pred', type=int, default=1, help="Number of steps to predict during training")
    
    # Training Settings
    parser.add_argument('--lr', type=float, default= 3e-4, help='Learning rate')
    parser.add_argument('--opt_vae_epochs', type=int, default=0, help='Number of epochs to train VAE only')
    parser.add_argument('--opt_vae_base_epochs', type=int, default=1024, help='Number of epochs to train VAE and base mixture matrices (must be >= opt_vae_epochs)')
    parser.add_argument('--traj_len', type=int, default= 7, help='Size of trajectory to train on')
    parser.add_argument('--lam_rec', type=float, default=0.95, help='Weight of image reconstruction loss')
    parser.add_argument('--lam_kl', type=float, default=0.80, help='Weight of kl loss')
    parser.add_argument('--use_binary_ce', type=str2bool, default=False, help='Use Binary Cross Entropy loss insted of default Mean Squared Error loss')

    args = parse_common_training_args(parser=parser)
    return args

def parse_ppo_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--action_std', type=float, default=0.5, help='Std of action distribution')
    parser.add_argument('--eps_clip', type=float, default=0.2, help='PPO clip parameter')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--epochs', type=int, default=80, help='Amount of update policy epochs')
    parser.add_argument('--logging_interval', type=int, default=10, help='Print avg reward every interval')
    parser.add_argument('--update_timesteps', type=int, default=3750, help='Update policy every timestep amount')
    parser.add_argument('--opt_timesteps', type=int, default=3750, help='Size of minibatch update')
    parser.add_argument('--max_timesteps', type=int, default=1500, help='Maximum timesteps per episode')
    parser.add_argument('--max_episodes', type=int, default=10000, help='Maximum episodes for training')
    parser.add_argument('--solved_reward', type=int, default=300, help='Reward threshold for solved environment')
    parser.add_argument('--architecture', choices=['cnn', 'mlp'], default='mlp',  help='Network architecture used')
    parser.add_argument('--lr', type=float, default=0.0003, help='Weight of kl loss')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for PyTorch')
    parser.add_argument('--env_name', type=str, default='ThingReacher2D-v0', help='Name of environment from gym')

    parser.add_argument('--render', type=str2bool, default=False, help='Render environment')
    parser.add_argument('--is_render', type=str2bool, default=None, help='Rendering specific to pybullet environment')
    parser.add_argument('--dim_u', type=int, default=2, help='Action dimension')
    parser.add_argument('--frame_stack', type=int, default=1, help='Number of frames to stack')
    parser.add_argument('--dim_x', type=str2inttuple, default=(64, 64, 1), help='3-tuple image dimension (H, W, C)')
    args = parser.parse_args()

    return args
