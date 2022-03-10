"""
This is a CLASS of predictor.
The idea is to decouple the estimation of system future state from the controller design.
While designing the controller you just chose the predictor you want,
 initialize it while initializing the controller and while stepping the controller you just give it current state
    and it returns the future states

"""

"""
This is a predictor for autoregressive neural network constructed in tensorflow
Control inputs should be first (regarding vector indices) inputs of the vector.
all other net inputs in the same order as net outputs
"""

"""
Using predictor:
1. Initialize while initializing controller
    This step load the RNN - it may take quite a bit of time
    During initialization you need to provide RNN which should be loaded
2. Call iterativelly three functions
    a) setup(initial_state, horizon, etc.)
    b) predict(Q)
    c) update_net
    
    ad a) at this stage you can change the parameters for prediction like e.g. horizon, dt
            It also prepares 0 state of the prediction, and tensors for saving the results,
            to make b) max performance. This function should be called BEFORE starting solving an optimisation problem
    ad b) predict is optimized to get the prediction of future states of the system as fast as possible.
        It accepts control input (vector) as its only input and is intended to be used at every evaluation of the cost functiomn
    ad c) this method updates the internal state of RNN. It accepts control input for current time step (scalar) as its only input
            it should be called only after the optimization problem is solved with the control input used in simulation
            
"""

# TODO: Make horizon updatable in runtime

# "Command line" parameters
from SI_Toolkit.TF.TF_Functions.Initialization import get_net, get_norm_info_for_net
from SI_Toolkit.TF.TF_Functions.Network import get_internal_states, load_internal_states
from SI_Toolkit.load_and_normalize import denormalize_numpy_array, normalize_numpy_array

try:
    from SI_Toolkit_ApplicationSpecificFiles.predictors_customization import STATE_VARIABLES, STATE_INDICES, \
        CONTROL_INPUTS, augment_predictor_output
except ModuleNotFoundError:
    print('SI_Toolkit_ApplicationSpecificFiles not yet created')

import numpy as np

from types import SimpleNamespace
import os
import yaml

import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Restrict printing messages from TF

config = yaml.load(open(os.path.join('SI_Toolkit_ApplicationSpecificFiles', 'config_testing.yml'), 'r'),
                   Loader=yaml.FullLoader)

PATH_TO_NN = config['testing']['PATH_TO_NN']


def check_dimensions(s, Q):

    # Make sure the input is at least 2d
    if s.ndim == 1:
        s = s[np.newaxis, :]

    if Q.ndim == 3:  # Q.shape = [batch_size, timesteps, features]
        pass
    elif Q.ndim == 2:  # Q.shape = [timesteps, features]
        Q = Q[np.newaxis, :, :]
    else:  # Q.shape = [features;  tf.rank(Q) == 1
        Q = Q[np.newaxis, np.newaxis, :]

    return s, Q


def convert_to_tensors(s, Q):
    return tf.convert_to_tensor(s, dtype=tf.float32), tf.convert_to_tensor(Q, dtype=tf.float32)


class predictor_autoregressive_tf:
    def __init__(self, horizon=None, batch_size=None, net_name=None):

        self.batch_size = batch_size
        self.horizon = horizon

        a = SimpleNamespace()

        a.path_to_models = PATH_TO_NN

        a.net_name = net_name

        # Create a copy of the network suitable for inference (stateful and with sequence length one)
        self.net, self.net_info = \
            get_net(a, time_series_length=1,
                    batch_size=self.batch_size, stateful=True)

        self.normalization_info = get_norm_info_for_net(self.net_info)

        self.rnn_internal_states = get_internal_states(self.net)

        self.net_input_reg_initial_normed = None

        self.output = np.zeros([self.batch_size, self.horizon + 1, len(STATE_VARIABLES)],
                                     dtype=np.float32)

        print('Init done')

    def predict(self, initial_state, Q) -> np.array:

        initial_state, Q = check_dimensions(initial_state, Q)
        self.output[:, 0, :] = initial_state

        net_input_reg_initial = initial_state[:, [STATE_INDICES.get(key) for key in
                                                  self.net_info.inputs[len(CONTROL_INPUTS):]]]  # [batch_size, features]

        self.net_input_reg_initial_normed = normalize_numpy_array(net_input_reg_initial,
                                                                  self.net_info.inputs[len(CONTROL_INPUTS):],
                                                                  self.normalization_info)

        self.net_input_reg_initial_normed, Q = convert_to_tensors(self.net_input_reg_initial_normed, Q)

        # load internal RNN state if applies
        load_internal_states(self.net, self.rnn_internal_states)

        net_outputs = self.iterate_net(Q, self.net_input_reg_initial_normed)

        # Denormalize
        self.output[..., 1:, [STATE_INDICES.get(key) for key in self.net_info.outputs]] = \
            denormalize_numpy_array(net_outputs.numpy(), self.net_info.outputs, self.normalization_info)

        # Augment
        augment_predictor_output(self.output, self.net_info)

        return self.output

    @tf.function(experimental_compile=True)
    def iterate_net(self, Q, net_input_reg_initial_normed):

        net_outputs = tf.TensorArray(tf.float32, size=self.horizon)
        net_output = tf.zeros(shape=(self.batch_size, len(self.net_info.outputs)), dtype=tf.float32)

        for i in tf.range(self.horizon):

            Q_current = Q[:, i, :]

            if i == 0:
                net_input = tf.reshape(
                    tf.concat([Q_current, net_input_reg_initial_normed], axis=1),
                    shape=[-1, 1, len(self.net_info.inputs)])
            else:
                net_input = tf.reshape(
                    tf.concat([Q_current, net_output], axis=1),
                    shape=[-1, 1, len(self.net_info.inputs)])

            net_output = self.net(net_input)

            net_output = tf.reshape(net_output, [-1, len(self.net_info.outputs)])

            net_outputs = net_outputs.write(i, net_output)

        net_outputs = tf.transpose(net_outputs.stack(), perm=[1, 0, 2])

        return net_outputs

    # FIXME: STATE IS NOT USED CORRECTLY
    def update_internal_state(self, s, Q0):
        # Run current input through network
        Q0 = tf.convert_to_tensor(Q0, dtype=tf.float32)
        self.update_internal_state_tf(Q0)


    # @tf.function(experimental_compile=True)
    def update_internal_state_tf(self, Q0):
        # load internal RNN state

        load_internal_states(self.net, self.rnn_internal_states)

        if self.net_info.net_type == 'Dense':
            net_input = tf.concat([Q0[:, 0, :], self.net_input_reg_initial_normed], axis=1)
        else:
            net_input = (tf.reshape(tf.concat([Q0[:, 0, :], self.net_input_reg_initial_normed], axis=1),
                                    [-1, 1, len(self.net_info.inputs)]))

        self.net(net_input)  # Using net directly

        self.rnn_internal_states = get_internal_states(self.net)


if __name__ == '__main__':
    import timeit

    initialisation = '''
from SI_Toolkit.Predictors.predictor_autoregressive_tf import predictor_autoregressive_tf
from SI_Toolkit_ApplicationSpecificFiles.predictors_customization import CONTROL_INPUTS
import numpy as np
batch_size = 2000
horizon = 50
predictor = predictor_autoregressive_tf(horizon, batch_size=batch_size, net_name='GRU-6IN-32H1-32H2-5OUT-0')
initial_state = np.random.random(size=(batch_size, 6))
# initial_state = np.random.random(size=(1, 6))
Q = np.float32(np.random.random(size=(batch_size, horizon, len(CONTROL_INPUTS))))
predictor.predict(initial_state, Q)
'''

    code = '''\
predictor.predict(initial_state, Q)
predictor.update_internal_state(initial_state, Q)'''

    print(timeit.timeit(code, number=10, setup=initialisation) / 10.0)