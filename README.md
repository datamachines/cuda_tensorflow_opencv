# DockerFile with GPU support for TensorFlow and OpenCV

Install GPU optimized version of TensorFlow and OpenCV. Also install protobuf, Jupyter, Keras, numpy, pandas and X11 support.

Requires a Linux system with nvidia-docker (v2) and the Nvidia drivers installed to run. See https://github.com/NVIDIA/nvidia-docker for setup details

## Directories and tag naming

The directories follow the `cuda_tensorflow_opencv` naming order.
As such `9.0_1.12.0_4.0.1` refers to *Cuda 9.0*, *TensorFlow 1.12.0* and *OpenCV 4.0.1*, and `10.0_1.13.1_4.0.1` refers to *Cuda 10.0*, *TensorFlow 1.13.1* and *OpenCV 4.0.1*.

Docker images are tagged with a version information for the Dockerfile they were built from at the end of the tag string (following a dash character), such that `cuda_tensorflow_opencv:9.0_1.12.0_4.0.1-0.1` is for *version 0.1*.

## Version specific information

### 10.0_1.13.1_4.0.1

`Dockerfile`  using `FROM tensorflow/tensorflow:1.13.1-gpu-py3-jupyter` 
For more information, see https://github.com/tensorflow/tensorflow/releases/tag/v1.13.1 and https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/docker, knowing that `TensorFlow GPU binaries are now built against CUDA 10 and TensorRT 5.0`.
Recommended reading on Tensorflow Docker GPU at https://www.tensorflow.org/install/docker#gpu_support

The TensorFlow Docker image is from a Ubuntu 16.04 (`lsb_release -a` shows `Ubuntu 16.04.5 LTS`) and has CUDA 10.0 libraries available (`dpkg -l | grep cuda`). For more details, see https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/10.0/runtime/cudnn7/Dockerfile

A `runDocker.sh` script is present in the directory to test the built image; it will set up the X11 passthrough and give the use a prompt, as well as mount the calling directory as `/dmc`.The user can test X11 is functional by using a simple X command such as `xlogo` from the command line.

### 9.0_1.12.0_4.0.1

`Dockerfile` using `FROM tensorflow/tensorflow:1.12.0-gpu-py3` using Ubuntu 16.04 and CUDA 9.0

## Using TensorFlow in your code

Code written for Tensorflow should follow principles described in https://www.tensorflow.org/guide/using_gpu

In particular, the following section https://www.tensorflow.org/guide/using_gpu#allowing_gpu_memory_growth might be needed to allow proper use of the GPU's memory. In particular:
   
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config, ...)

Note that this often allocates all the GPU memory to one Tensorflow client. If you intend to run multiple Tensorflow containers, limiting the available memory available to the container's Tensorflow can be achieved as described in https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory by instead specifying the percentage of the GPU memory to be used:

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction=0.125
    session = tf.Session(config=config, ...)

The built Docker images do NOT install any models, add/build/download your own in your `Dockerfile` that is `FROM cuda_tensorflow_opencv:9.0_1.12.0_4.0.1-0.1`

For example:

    FROM cuda_tensorflow_opencv:9.0_1.12.0_4.0.1-0.1
    
    # Download tensorflow object detection models
    RUN GIT_SSL_NO_VERIFY=true git clone -q https://github.com/tensorflow/models /usr/local/lib/python3.5/dist-packages/tensorflow/models

    # Install downloaded models
    ENV PYTHONPATH "$PYTHONPATH:/usr/local/lib/python3.5/dist-packages/tensorflow/models/research:/usr/local/lib/python3.5/dist-packages/tensorflow/models/research/slim"
    RUN cd /usr/local/lib/python3.5/dist-packages/tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=.

## X11 display

To run an interactive `/bin/bash` with X11 set for Docker and the current directory loaded in `/dmc`

    xhost +local:docker
    XSOCK=/tmp/.X11-unix
    XAUTH=/tmp/.docker.xauth
    xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
    nvidia-docker run -it --rm -e DISPLAY=$DISPLAY -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH -v ${PWD}:/dmc --ipc host cuda_tensorflow_opencv:9.0_1.12.0_4.0.1-0.1 /bin/bash
    xhost -local:docker

Note that the base container runs as root, if you want to run it as a non root user, add `-u $(id -u):$(id -g)` to the `nvidia-docker` command line but ensure that you have access to the directories you will work in.