import argparse

from args.utils import str2bool, str2inttuple, str2tuple, str2floattuple

def parse_common_training_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    # Debug Settings
    parser.add_argument('--debug', type=str2bool, default=False,
                        help='Debug and do not save models or log anything')

    # Experiment Settings
    parser.add_argument('--storage_base_path', type=str, required=True,
                        help='Base path to store all training data')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use for PyTorch')
    parser.add_argument('--cudnn_deterministic', type=str2bool, default=True,
                        help='Use cudnn deterministic')
    parser.add_argument('--cudnn_benchmark', type=str2bool, default=False,
                        help='Use cudnn benchmark')
    parser.add_argument('--task', type=str, default="pendulum64",
                        help='The task that is being trained on')
    parser.add_argument('--comment', type=str, default="None",
                        help='Comment to describe model')

    # Dataset Settings
    parser.add_argument('--dataset', type=str, required=True, 
                        help='Name of dataset to train on')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Amount of dataset to use for validation')

    # Training Settings
    parser.add_argument('--n_epoch', type=int, default=600,
                        help='Number of epochs',)
    parser.add_argument('--n_batch', type=int, default=128, 
                        help='Batch size')
    parser.add_argument('--n_example', type=int, default=10000000, 
                        help='Maximum samples to train from the dataset')
    parser.add_argument('--n_worker', type=int, default=4, 
                        help='Amount of workers for dataloading.')

    # Network Settings                        
    parser.add_argument('--use_batch_norm', type=str2bool, default=False,
                        help='Use batch normalization')
    parser.add_argument('--use_dropout', type=str2bool, default=False,
                        help='Use dropout')
    parser.add_argument('--weight_init', choices=['custom', 'none'], default='none', 
                        help='Weight initialization')

    # Optimizer Settings                        
    parser.add_argument('--beta1', type=float, default=0.9, 
                        help='Adam optimizer beta 1')
    parser.add_argument('--beta2', type=float, default=0.999, 
                        help='Adam optimizer beta 2')
    parser.add_argument('--opt', choices=['adam', 'sgd', 'adadelta', 'rmsprop'], default='adam', 
                        help='Optimizer used')
    args = parser.parse_args()
    return args