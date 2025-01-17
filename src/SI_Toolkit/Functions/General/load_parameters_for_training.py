# -*- coding: utf-8 -*-
"""
This function loads parameters for training (TF/Pytorch/GPs) from config,
adds however an option to overwrite config values with command line
"""
import argparse
import glob
import yaml, os

config = yaml.load(open(os.path.join('SI_Toolkit_ASF', 'config_training.yml'), 'r'), Loader=yaml.FullLoader)

library = config['library']

net_name = config['modeling']['NET_NAME']

# Path to trained models and their logs
PATH_TO_MODELS = config["paths"]["PATH_TO_EXPERIMENT_FOLDERS"] + config['paths']['path_to_experiment'] + "Models/"

PATH_TO_NORMALIZATION_INFO = config["paths"]["PATH_TO_EXPERIMENT_FOLDERS"] + config['paths']['path_to_experiment'] + "NormalizationInfo/"

# Get path to normalisation info as to a newest csv file in indicated folder
paths = sorted([os.path.join(PATH_TO_NORMALIZATION_INFO, d) for d in os.listdir(PATH_TO_NORMALIZATION_INFO)], key=os.path.getctime)
for path in paths:
    if path[-4:] == '.csv':
        PATH_TO_NORMALIZATION_INFO = path

# The following paths to dictionaries may be replaced by the list of paths to data files.
TRAINING_FILES = config["paths"]["PATH_TO_EXPERIMENT_FOLDERS"] + config['paths']['path_to_experiment'] + "/Recordings/Train/"
VALIDATION_FILES = config["paths"]["PATH_TO_EXPERIMENT_FOLDERS"] + config['paths']['path_to_experiment'] + "/Recordings/Validate/"
TEST_FILES = config["paths"]["PATH_TO_EXPERIMENT_FOLDERS"] + config['paths']['path_to_experiment'] + "/Recordings/Test/"


# region Set inputs and outputs

control_inputs = config['training_default']['control_inputs']
state_inputs = config['training_default']['state_inputs']
setpoint_inputs = config['training_default']['setpoint_inputs']
outputs = config['training_default']['outputs']
translation_invariant_variables = config['training_default']['translation_invariant_variables']

EPOCHS = config['training_default']['EPOCHS']
BATCH_SIZE = config['training_default']['BATCH_SIZE']
SEED = config['training_default']['SEED']
LR = config['training_default']['LR']
SHIFT_LABELS = config['training_default']['SHIFT_LABELS']

WASH_OUT_LEN = config['training_default']['WASH_OUT_LEN']
POST_WASH_OUT_LEN = config['training_default']['POST_WASH_OUT_LEN']
ON_FLY_DATA_GENERATION = config['training_default']['ON_FLY_DATA_GENERATION']
NORMALIZE = config['training_default']['NORMALIZE']
USE_NNI = config['training_default']['USE_NNI']
CONSTRUCT_NETWORK = config['training_default']['CONSTRUCT_NETWORK']

# For l2race
# control_inputs = ['u1', 'u2']
# state_inputs = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']
# outputs = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']

# endregion

def args():
    parser = argparse.ArgumentParser(description='Train a GRU network.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--library', default=library, type=str,
                        help='Decide if you want to use TF or Pytorch for training.')

    # Defining the model
    parser.add_argument('--net_name', default=net_name, type=str,
                        help='Name defining the network.'
                             'It has to have the form:'
                             '(RNN type [GRU/LSTM]/Dense)-(size first hidden layer)H1-(size second hidden layer)H2-...'
                             'e.g. GRU-64H1-64H2-32H3')

    parser.add_argument('--training_files', default=TRAINING_FILES, type=str,
                        help='File name of the recording to be used for training the RNN'
                             'e.g. oval_easy.csv ')
    parser.add_argument('--validation_files', default=VALIDATION_FILES, type=str,
                        help='File name of the recording to be used for validating the RNN'
                             'e.g. oval_easy_test.csv ')
    parser.add_argument('--test_files', default=TEST_FILES, type=str,
                        help='File name of the recording to be used for validating the RNN'
                             'e.g. oval_easy_test.csv ')

    parser.add_argument('--control_inputs', default=control_inputs,
                        help='List of control inputs to neural network')
    parser.add_argument('--state_inputs', default=state_inputs,
                        help='List of state inputs to neural network')
    parser.add_argument('--setpoint_inputs', default=setpoint_inputs,
                        help='List of setpoint inputs to neural network')
    parser.add_argument('--outputs', default=outputs,
                        help='List of outputs from neural network')
    parser.add_argument('--translation_invariant_variables', default=translation_invariant_variables,
                        help='List of translation_invariant_variables to neural network - shift of the whole series does not change the result')

    # Training only:
    parser.add_argument('--wash_out_len', default=WASH_OUT_LEN, type=int, help='Number of timesteps for a wash-out sequence, min is 0')
    parser.add_argument('--post_wash_out_len', default=POST_WASH_OUT_LEN, type=int,
                        help='Number of timesteps after wash-out sequence (this is used to calculate loss), min is 1')

    # Training parameters
    parser.add_argument('--num_epochs', default=EPOCHS, type=int, help='Number of epochs of training')
    parser.add_argument('--batch_size', default=BATCH_SIZE, type=int, help='Size of a batch')
    parser.add_argument('--seed', default=SEED, type=int, help='Set seed for reproducibility')
    parser.add_argument('--lr', default=LR, type=float, help='Learning rate')
    parser.add_argument('--shift_labels', default=SHIFT_LABELS, type=int, help='How much to shift labels/targets with respect to features while reading data file for training')

    parser.add_argument('--path_to_models', default=PATH_TO_MODELS, type=str,
                        help='Path where to save/ from where to load models')

    parser.add_argument('--path_to_normalization_info', default=PATH_TO_NORMALIZATION_INFO, type=str,
                        help='Path where the cartpole data is saved')

    parser.add_argument('--on_fly_data_generation', default=ON_FLY_DATA_GENERATION, type=bool,
                        help='Generate data for training during training, instead of loading previously saved data')
    parser.add_argument('--normalize', default=NORMALIZE, type=bool, help='Make all data between 0 and 1')
    parser.add_argument('--use_nni', default=USE_NNI, type=bool, help='Use NNI package to search hyperparameter space')
    parser.add_argument('--construct_network', default=CONSTRUCT_NETWORK, type=str,
                        help='For Pytorch you can decide if you want to construct network with modules or cells.'
                             'First is needed for DeltaRNN, second gives more flexibility in specifying layers sizes.')


    args = parser.parse_args()

    # Make sure that the provided lists of inputs and outputs are in alphabetical order

    if args.post_wash_out_len < 1:
        raise ValueError('post_wash_out_len, the part relevant for loss calculation must be at least 1, also for dense network')

    if args.control_inputs is not None:
        args.control_inputs = sorted(args.control_inputs)

    if args.state_inputs is not None:
        args.state_inputs = sorted(args.state_inputs)

    if args.setpoint_inputs is not None:
        args.setpoint_inputs = sorted(args.setpoint_inputs)

    args.inputs = args.control_inputs + args.state_inputs + args.setpoint_inputs

    if args.outputs is not None:
        args.outputs = sorted(args.outputs)

    return args

