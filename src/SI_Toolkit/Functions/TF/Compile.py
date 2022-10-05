import logging
import tensorflow as tf
import platform

try:
    from SI_Toolkit_ASF import GLOBALLY_DISABLE_COMPILATION, USE_JIT_COMPILATION
except ImportError:
    logging.warn("No compilation option set in SI_Toolkit_ASF. Setting GLOBALLY_DISABLE_COMPILATION to True.")
    GLOBALLY_DISABLE_COMPILATION = True

def tf_function_jit(func):
    return tf.function(func=func, jit_compile=True)


def tf_function_experimental(func):
    return tf.function(func=func, experimental_compile=True)


def identity(func):
    return func


if GLOBALLY_DISABLE_COMPILATION:
    Compile = identity
else:
    if platform.machine() == 'arm64' and platform.system() == 'Darwin':  # For M1 Apple processor
        Compile = tf.function
    elif not USE_JIT_COMPILATION:
        Compile = tf.function
    else:
        Compile = tf_function_jit
        # Compile = tf_function_experimental # Should be same as tf_function_jit, not appropriate for newer version of TF