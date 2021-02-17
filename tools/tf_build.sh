#!/bin/bash
set -e

# Preliminary list of variables found at https://gist.github.com/PatWie/0c915d5be59a518f934392219ca65c3d
# Addtional (from 2.2):
# 'build' options: --apple_platform_type=macos --define framework_shared_object=true --define open_source_build=true --java_toolchain=//third_party/toolchains/java:tf_java_toolchain --host_java_toolchain=//third_party/toolchains/java:tf_java_toolchain --define=use_fast_cpp_protos=true --define=allow_oversize_protos=true --spawn_strategy=standalone -c opt --announce_rc --define=grpc_no_ares=true --noincompatible_remove_legacy_whole_archive --noincompatible_prohibit_aapt1 --enable_platform_specific_config --config=v2
#INFO: Reading rc options for 'build' from /usr/local/src/tensorflow/.tf_configure.bazelrc:
#  'build' options: --action_env PYTHON_BIN_PATH=/usr/local/bin/python --action_env PYTHON_LIB_PATH=/usr/local/lib/python3.6/dist-packages --python_path=/usr/local/bin/python --action_env TF_CUDA_VERSION=10.1 --action_env TF_CUDNN_VERSION=7 --action_env CUDA_TOOLKIT_PATH=/usr/local/cuda-10.1 --action_env TF_CUDA_COMPUTE_CAPABILITIES=3.5,7.0 --action_env LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/extras/CUPTI/lib64 --action_env GCC_HOST_COMPILER_PATH=/usr/bin/x86_64-linux-gnu-gcc-7 --config=cuda --action_env TF_CONFIGURE_IOS=0

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
cudnn=0
if [ "A$1" == "Ayes" ]; then
  cudnn=1
  echo "** CUDNN requested" | tee -a /tmp/tf_env.dump
  # CuDNN8
  cudnn_inc="/usr/include/cudnn.h"
  cudnn8_inc="/usr/include/x86_64-linux-gnu/cudnn_version_v8.h"
  if [ -f $cudnn8_inc ]; then
    cudnn_inc="${cudnn8_inc}"
  fi
  if [ ! -f $cudnn_inc ]; then
    echo "** Unable to find $cudnn_inc, will not be able to compile requested GPU build, aborting" | tee -a /tmp/tf_env.dump
    exit 1
  fi
fi
if [ "A$cudnn" == "A1" ]; then
  export TF_CUDNN_VERSION="$(sed -n 's/^#define CUDNN_MAJOR\s*\(.*\).*/\1/p' $cudnn_inc)"
  if [ "A$TF_CUDNN_VERSION" == "A" ]; then
    cudnn=0
    echo "** Problem finding DNN major version, aborting" | tee -a /tmp/tf_env.dump
    exit 1
  else
    export TF_NEED_CUDA=1
    export TF_NEED_TENSORRT=0
    export TF_CUDA_VERSION="$(nvcc --version | sed -n 's/^.*release \(.*\),.*/\1/p')"
    export TF_CUDA_COMPUTE_CAPABILITIES="${CTO_DNN_ARCH}"

    nccl_inc="/usr/local/cuda/include/nccl.h"
    if [ -f $nccl_inc ]; then
      export TF_NCCL_VERSION="$(sed -n 's/^#define NCCL_MAJOR\s*\(.*\).*/\1/p' $nccl_inc)"
    fi

    # cudnn build: TF 1.15.[345] with CUDA 10.2 fix -- see https://github.com/tensorflow/tensorflow/issues/34429
    if [ "A${TF_CUDA_VERSION=}" == "A10.2" ]; then
      if grep VERSION /usr/local/src/tensorflow/tensorflow/tensorflow.bzl | grep -q '1.15.[345]' ; then
        echo "-- Patching third_party/nccl/build_defs.bzl.tpl"
        perl -pi.bak -e 's/("--bin2c-path=%s")/## 1.15.x compilation ## $1/' third_party/nccl/build_defs.bzl.tpl
      fi
    fi
    
  fi
fi
if [ "A$cudnn" == "A1" ]; then
  config_add="$config_add --config=cuda"
fi

export GCC_HOST_COMPILER_PATH=$(which gcc)
#export CC_OPT_FLAGS="-march=native"
export CC_OPT_FLAGS=""
export PYTHON_BIN_PATH=$(which python)
export PYTHON_LIB_PATH="$(python -c 'import site; print(site.getsitepackages()[0])')"

export TF_CUDA_CLANG=0

export TF_DOWNLOAD_CLANG=0
export TF_DOWNLOAD_MKL=0

export TF_ENABLE_XLA=0

export TF_NEED_AWS=0
export TF_NEED_COMPUTECPP=0
export TF_NEED_GCP=0
export TF_NEED_GDR=0
export TF_NEED_HDFS=0
export TF_NEED_JEMALLOC=1
export TF_NEED_KAFKA=0
export TF_NEED_MKL=0
export TF_NEED_MPI=0
export TF_NEED_OPENCL=0
export TF_NEED_OPENCL_SYCL=0
export TF_NEED_ROCM=0
export TF_NEED_S3=0
export TF_NEED_VERBS=0

export TF_SET_ANDROID_WORKSPACE=0

##

echo "-- Environment variables set:"  | tee -a /tmp/tf_env.dump
env | grep TF_ | grep -v CTO_ | sort | tee -a /tmp/tf_env.dump
for i in GCC_HOST_COMPILER_PATH CC_OPT_FLAGS PYTHON_BIN_PATH PYTHON_LIB_PATH; do
  echo "$i="`printenv $i` | tee -a /tmp/tf_env.dump
done

echo "-- ./configure output:" | tee -a /tmp/tf_env.dump
./configure | tee -a /tmp/tf_env.dump 

start_time=$SECONDS
echo "-- bazel command to run:" | tee -a /tmp/tf_env.dump
build_cmd="bazel build --verbose_failures $config_add //tensorflow/tools/pip_package:build_pip_package"
echo  $build_cmd| tee -a /tmp/tf_env.dump 
$build_cmd
end_time=$SECONDS
elapsed=$(( end_time - start_time ))
echo "-- TensorFlow building time (in seconds): $elapsed" | tee -a /tmp/tf_env.dump

