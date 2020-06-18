import tensorflow as tf
from tensorflow.python.client import device_lib
import keras

print("*** Tensorflow version: ", tf.__version__)
print("*** keras version: ", keras.__version__)


print("--- All seen hardware: ", device_lib.list_local_devices())
print("--- GPU Available: ", tf.config.experimental.list_physical_devices('GPU'))
