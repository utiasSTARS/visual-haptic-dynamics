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
    parser.add_argument('--dim_z_img', type=int, default=16, help='Latent state dimension of images')
    parser.add_argument('--dim_z_context', type=int, default=16, help='Latent state dimension of contextual data')
    parser.add_argument('--dim_x', type=str2inttuple, default=(1, 64, 64), help='3-tuple image dimension (C, H, W)')
    parser.add_argument('--K', type=int, default=15, help='Number of mixture component for dynamic models')
    parser.add_argument('--fc_hidden_size', type=int, default=256, help='The number of hidden units for each linear layer')
    parser.add_argument('--rnn_hidden_size', type=int, default=256, help='The number of hidden units for each GRU or LSTM layer')
    parser.add_argument('--use_bidirectional', type=str2bool, default=False, help='Use bidirectional RNN')
    parser.add_argument('--rnn_net', choices=['gru', 'lstm'], default='gru',  help='RNN network type')
    parser.add_argument('--dyn_net', choices=['linearmix', 'linearrank1', 'nonlinear'], default='linearmix', help='Network architecture for dynamics')     
    parser.add_argument('--non_linearity', choices=['relu', 'elu', 'softplus'], default='relu', help='Activation used for decoder neural network')
    parser.add_argument('--frame_stacks', type=int, default=1, help="Number of frames to stack")
    parser.add_argument('--context_modality', choices=['none', 'arm', 'ft', 'joint'], default='none', help='Context modality')
    parser.add_argument('--use_context_frame_stack', type=str2bool, default=False, help='Stack context modality')
    parser.add_argument('--context', choices=['initial_latent_state', 'goal_latent_state', 'initial_image', 'goal_image', 'all_past_states', 'none'], default='none', help='Extra information for recognition network')
    parser.add_argument('--ft_normalization', type=int, default=100.0, help='Normalize ft data by this')
    parser.add_argument('--dim_arm', type=int, default=6, help='Feature length of arm sensors')
    parser.add_argument('--dim_ft', type=int, default=6, help='Feature length of ft data')
    parser.add_argument('--context_seq_len', type=int, default=32, help='Sequence length of high frequency data per image')

    # Training Settings
    parser.add_argument('--lr', type=float, default= 3e-4, help='Learning rate')
    parser.add_argument('--opt_vae_epochs', type=int, default=0, help='Number of epochs to train VAE only')
    parser.add_argument('--opt_vae_base_epochs', type=int, default=1024, help='Number of epochs to train VAE and base mixture matrices (must be >= opt_vae_epochs)')
    parser.add_argument('--opt_n_step_pred_epochs', type=str2inttuple, default=(4096,), help='Number of epochs before training with n step reconstruction')
    parser.add_argument('--lam_rec', type=float, default=0.95, help='Weight of image reconstruction loss')
    parser.add_argument('--lam_kl', type=float, default=0.80, help='Weight of kl loss')
    parser.add_argument('--use_binary_ce', type=str2bool, default=False, help='Use Binary Cross Entropy loss insted of default Mean Squared Error loss')

    args = parse_common_training_args(parser=parser)
    return args

def parse_control_experiment_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to directory of model')
    parser.add_argument('--dataset_path', type=str, default="/Users/oliver/visual-haptic-dynamics/experiments/data/datasets/visual_haptic_2D_len16_osc_withGT_8C12919B740845539C0E75B5CBAF7965.pkl", help='Path to directory of offline dataset')
    parser.add_argument('--H', type=int, default=6, help='Horizon length for MPC controller')
    parser.add_argument('--mpc_opt', choices=['grad', 'cem', 'cvxopt'], default='cvxopt', help='Optimization used for MPC')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for PyTorch')
    parser.add_argument('--n_episodes', type=int, default=1024, help='Amount of episodes to collect')
    parser.add_argument('--n_train_episodes', type=int, default=64, help='Train model every n episodes')
    parser.add_argument('--n_test_episodes', type=int, default=8, help='Test model every n episodes (remove exploration noise)')
    parser.add_argument('--n_epochs', type=int, default=128, help='Train model for n epochs')
    parser.add_argument('--n_checkpoint_episodes', type=int, default=64, help='Save model every n epochs')
    parser.add_argument('--n_worker', type=int, default=0,  help='Amount of workers for dataloading.')
    parser.add_argument('--opt', choices=['adam', 'sgd'], default='adam', help='Optimizer used')
    parser.add_argument('--lr', type=float, default= 3e-4, help='Learning rate')
    parser.add_argument('--render', type=str2bool, default=True, help='Debug and do not save models or log anything')
    parser.add_argument('--random_seed', type=int, default=333, help='Random seed')
    parser.add_argument('--exploration_noise_var', type=float, default=0.3, help='Exploration noise used')
    parser.add_argument('--debug', type=str2bool, default=False, help='Debug and do not save models or log anything')
    parser.add_argument('--comment', type=str, default="None", help='Comment to describe save directory')

    args = parser.parse_args()
    return args

