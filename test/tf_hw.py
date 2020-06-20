import tensorflow as tf
from tensorflow.python.client import device_lib
import keras

print("*** Tensorflow version   : ", tf.__version__)
print("*** Tensorflow Keras     : ", tf.keras.__version__)

print("*** TF Builf with cuda   : ", tf.test.is_built_with_cuda())
print("*** TF compile flags     : ", tf.sysconfig.get_compile_flags())
print("*** TF include           : ", tf.sysconfig.get_include())
print("*** TF lib               : ", tf.sysconfig.get_lib())
print("*** TF link flags        : ", tf.sysconfig.get_link_flags())

print("*** keras version: ", keras.__version__)


print("--- All seen hardware: ", device_lib.list_local_devices())
print("--- GPU Available: ", tf.config.experimental.list_physical_devices('GPU'))
