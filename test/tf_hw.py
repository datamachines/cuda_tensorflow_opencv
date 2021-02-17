import tensorflow as tf
from tensorflow.python.client import device_lib

print("*** Tensorflow version   : ", tf.__version__)
print("*** Tensorflow Keras     : ", tf.keras.__version__)

print("*** TF Builf with cuda   : ", tf.test.is_built_with_cuda())
print("*** TF compile flags     : ", tf.sysconfig.get_compile_flags())
print("*** TF include           : ", tf.sysconfig.get_include())
print("*** TF lib               : ", tf.sysconfig.get_lib())
print("*** TF link flags        : ", tf.sysconfig.get_link_flags())

import keras
print("*** Keras version        : ", keras.__version__)

import torch
print("*** PyTorch version      : ", torch.__version__)

import pandas
print("*** pandas version       : ", pandas.__version__)

import sklearn
print("*** scikit-learn version : ", sklearn.__version__)

print("")
print("(!! the following is build device specific, and here only to confirm hardware availability, ignore !!)")
print("--- All seen hardware    :\n", device_lib.list_local_devices())
print("--- TF GPU Available     :\n", tf.config.experimental.list_physical_devices('GPU'))

