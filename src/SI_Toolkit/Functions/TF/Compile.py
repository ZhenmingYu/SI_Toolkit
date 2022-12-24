import logging
import platform

import tensorflow as tf
import torch

from Control_Toolkit.others.globals_and_utils import get_logger
log=get_logger(__name__)

from SI_Toolkit.computation_library import ComputationLibrary



try:
    from SI_Toolkit_ASF import USE_TENSORFLOW_EAGER_MODE, USE_TENSORFLOW_XLA
except ImportError:
    log.warn("No compilation option set in SI_Toolkit_ASF/__init.py__. Setting USE_TENSORFLOW_EAGER_MODE to True.")
    USE_TENSORFLOW_EAGER_MODE = True

def tf_function_jit(func):
    log.info(f'compiling tf.function from {func}')
    return tf.function(func=func, jit_compile=True,)


def tf_function_experimental(func):
    return tf.function(func=func, experimental_compile=True)


def identity(func):
    return func


if USE_TENSORFLOW_EAGER_MODE:
    log.info('TensorFlow compilation is disabled by USE_TENSORFLOW_EAGER_MODE=True')
    CompileTF = identity
else:
    if platform.machine() == 'arm64' and platform.system() == 'Darwin':  # For M1 Apple processor
        log.info('TensorFlow compilation  (but not JIT) is enabled by tf.function by USE_TENSORFLOW_EAGER_MODE=False  and USE_TENSORFLOW_XLA = False')
        CompileTF = tf.function
    elif not USE_TENSORFLOW_XLA:
        log.info('TensorFlow compilation (but not JIT) is enabled by tf.function by USE_TENSORFLOW_EAGER_MODE=False and USE_TENSORFLOW_XLA = False')
        CompileTF = tf.function
    else:
        log.info('TensorFlow compilation and JIT are both enabled by tf.function_jit by USE_TENSORFLOW_EAGER_MODE=False and USE_TENSORFLOW_XLA = True')
        CompileTF = tf_function_jit
    log.info(f'using {CompileTF} compilation')
        # CompileTF = tf_function_experimental # Should be same as tf_function_jit, not appropriate for newer version of TF

def CompileAdaptive(fun):
    """
    Compiles the function using options for TensorFlow and XLA JIT, according to global flags USE_TENSORFLOW_EAGER_MODE.

    See SI_Toolkit_ASF\__init__.py

    """
    instance = fun.__self__
    assert hasattr(instance, "lib"), "Instance with this method has no computation library defined"
    computation_library: "type[ComputationLibrary]" = instance.lib
    lib_name = computation_library.lib

    if USE_TENSORFLOW_EAGER_MODE:
        return identity(fun)
    elif lib_name == 'TF':
        log.info(f'compiling tensorflow {fun}')
        return CompileTF(fun)
    else:
        log.warning(f'JIT compilation for {lib_name} not yet implemented.')
        return identity(fun)
