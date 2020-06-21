#!/bin/bash
set -e

# Preliminary list of variables found at https://gist.github.com/PatWie/0c915d5be59a518f934392219ca65c3d

cd /usr/local/src/tensorflow
# Args: 
# 1 = CuDNN:yes/no
# 2 = Tensorflow config build extra (ex:v1)

# Default config is to have it "CPU"
config_add="--config=opt"

if [ "A$1" == "A" ]; then 
  echo "Usage: $0 CuDNN TF_config"
  echo "  CuDNN: yes no"
  echo "  TF_config: 1x extra argument to pass to --config= (ex: v1)"
  exit 1
fi

if [ "A$2" != "A" ]; then
  config_add="$config_add --config=$2"
fi

echo "--- Tensorflow Build --- " > /tmp/tf_env.dump
export TF_NEED_CUDA=0
cuda=0
cudnn_inc="/usr/include/cudnn.h"
if [ "A$1" == "Ayes" ]; then
  cuda=1
  echo "** CUDNN requested" | tee -a /tmp/tf_env.dump
  if [ ! -f $cuddn_inc ]; then
    echo "** Unable to find $cudnn_inc, will not be able to compile GPU build" | tee -a /tmp/tf_env.dump
    cuda=0
  fi
fi
if [ "A$cuda" == "A1" ]; then
  export TF_CUDNN_VERSION="$(sed -n 's/^#define CUDNN_MAJOR\s*\(.*\).*/\1/p' $cudnn_inc)"
  if [ "A$TF_CUDNN_VERSION" == "A" ]; then
    cuda=0
    echo "** Problem finding DNN major version, unsetting GPU optimizations for TF" | tee -a /tmp/tf_env.dump
    export TF_NEED_CUDA=0
    unset TF_CUDNN_VERSION
  else
    export TF_NEED_CUDA=1
    export TF_NEED_TENSORRT=0
    export TF_CUDA_VERSION="$(nvcc --version | sed -n 's/^.*release \(.*\),.*/\1/p')"
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64
    # v1.15.3 specific build fix -- see https://github.com/tensorflow/tensorflow/issues/34429
    if grep -q 1.15.3 /usr/local/src/tensorflow/RELEASE.md; then
      echo "-- Patching third_party/nccl/build_defs.bzl.tpl"
      perl -pi.bak -e 's/("--bin2c-path=%s")/## 1.15.3 compilation ## $1/' third_party/nccl/build_defs.bzl.tpl
    fi
  fi
fi
if [ "A$cuda" == "A1" ]; then
  config_add="$config_add --config=cuda"
fi

export GCC_HOST_COMPILER_PATH=$(which gcc)
export CC_OPT_FLAGS="-march=native"
export PYTHON_BIN_PATH=$(which python)
export PYTHON_LIB_PATH="$(python -c 'import site; print(site.getsitepackages()[0])')"
#export PYTHONPATH=${TF_ROOT}/lib
#export PYTHON_ARG=${TF_ROOT}/lib

export TF_NEED_GCP=0
export TF_NEED_HDFS=0
export TF_NEED_OPENCL=0
export TF_NEED_JEMALLOC=1
export TF_ENABLE_XLA=0
export TF_NEED_VERBS=0
export TF_CUDA_CLANG=0
export TF_NEED_MKL=0
export TF_DOWNLOAD_MKL=0
export TF_NEED_AWS=0
export TF_NEED_MPI=0
export TF_NEED_GDR=0
export TF_NEED_S3=0
export TF_NEED_OPENCL_SYCL=0
export TF_SET_ANDROID_WORKSPACE=0
export TF_NEED_COMPUTECPP=0
export TF_NEED_KAFKA=0
export TF_NEED_ROCM=0
export TF_DOWNLOAD_CLANG=0

echo "-- Environment variables set:"  | tee -a /tmp/tf_env.dump
env | grep TF_ | grep -v CTO_ | sort | tee -a /tmp/tf_env.dump
for i in GCC_HOST_COMPILER_PATH CC_OPT_FLAGS PYTHON_BIN_PATH PYTHON_LIB_PATH; do
  echo "$i="`printenv $i` | tee -a /tmp/tf_env.dump
done

echo "-- ./configure output:" | tee -a /tmp/tf_env.dump
./configure | tee -a /tmp/tf_env.dump 

start_time=$SECONDS
echo "-- bazel command to run:" | tee -a /tmp/tf_env.dump
echo bazel build --verbose_failures $config_add //tensorflow/tools/pip_package:build_pip_package | tee -a /tmp/tf_env.dump 
bazel build --verbose_failures $config_add //tensorflow/tools/pip_package:build_pip_package
end_time=$SECONDS
elapsed=$(( end_time - start_time ))
echo "-- TensorFlow building time (in seconds): $elapsed" | tee -a /tmp/tf_env.dump

