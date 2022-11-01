import numpy as np
import tensorflow as tf
from tensorflow import keras

from SI_Toolkit.Predictors.autoregression import autoregression_loop
from SI_Toolkit.computation_library import TensorFlowLibrary

loss_fn = keras.losses.MeanSquaredError()

"""
Only pole lenght at output.
External input:state, initial input: random pole length
"""


def fit_autoregressive_Marcin(net, net_info, training_dataset, validation_dataset, test_dataset, a):

    batch_size_training = training_dataset.batch_size
    epochs = a.num_epochs
    optimizer = keras.optimizers.Adam(a.lr)
    loss = []
    training_loss = []
    validation_loss = []

    AL: autoregression_loop = autoregression_loop(
        model_inputs_len=len(net_info.inputs),
        model_outputs_len=len(net_info.outputs),
        batch_size=batch_size_training,
        lib=TensorFlowLibrary,
        differential_model_autoregression_helper_instance=None,
    )

    for epoch in range(epochs):

        print("\nStart of epoch %d" % (epoch,))
        for batch in tf.range(len(training_dataset)):   # Iterate over the batches of the dataset.

            x_batch, y_batch = training_dataset[batch]

            pole_length_initial_guess = tf.random.uniform(shape=(batch_size_training, 1))
            states = x_batch[:, :, 1:]
            initial_input = pole_length_initial_guess
            external_input_left = None
            external_input_right = states

            # initial_input = x_batch[:, 0, 1:]
            # external_input_left = x_batch[:, :, :1]
            # external_input_right = None

            with tf.GradientTape() as tape:

                outputs = AL.run(
                    model=net,
                    horizon=a.wash_out_len+a.post_wash_out_len,
                    initial_input=initial_input,
                    external_input_left=external_input_left,
                    external_input_right=external_input_right,
                )

                # Compute the loss value for this minibatch.
                loss_value = loss_fn(y_batch, outputs)
                loss.append(loss_value)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, net.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, net.trainable_weights))

            # Log every 200 batches.
            if batch % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (batch, float(loss_value))
                )
                print("Seen so far: %s samples" % ((batch + 1) * a.batch_size))

        # validation
        training_loss.append(np.mean(loss))
        validation_loss.append(validation_step(net, net_info, validation_dataset, a))

    return np.array(loss), validation_loss


def validation_step(net, net_info, validation_dataset, a):

    batch_size_validation = validation_dataset.batch_size

    loss = []

    AL: autoregression_loop = autoregression_loop(
        model_inputs_len=len(net_info.inputs),
        model_outputs_len=len(net_info.outputs),
        batch_size=batch_size_validation,
        lib=TensorFlowLibrary,
        differential_model_autoregression_helper_instance=None,
    )

    for batch in tf.range(len(validation_dataset)):  # Iterate over the batches of the dataset.

        x_batch, y_batch = validation_dataset[batch]

        pole_length_initial_guess = tf.random.uniform(shape=(batch_size_validation, 1))
        states = x_batch[:, :, 1:]
        initial_input = pole_length_initial_guess
        external_input_left = None
        external_input_right = states

        # initial_input = x_batch[:, 0, 1:]  # state
        # external_input_left = x_batch[:, :, :1]  # Control input
        # external_input_right = None


        outputs = AL.run(
            model=net,
            horizon=a.wash_out_len + a.post_wash_out_len,
            initial_input=initial_input,
            external_input_left=external_input_left,
            external_input_right=external_input_right,
        )

        # Compute the loss value for this minibatch.
        loss_value = loss_fn(y_batch, outputs)
        loss.append(loss_value)

    return np.mean(loss)