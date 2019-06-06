# DockerFile with Nvidia GPU support for TensorFlow and OpenCV

Install an Nvidia GPU optimized version of TensorFlow and OpenCV. Also install, Jupyter, Keras, numpy, pandas and X11 support.

Requires a Linux system with nvidia-docker (v2) and the Nvidia drivers installed to run. See https://github.com/NVIDIA/nvidia-docker for setup details

**Docker Images built from this repository are publicly available at https://hub.docker.com/r/datamachines/cuda_tensorflow_opencv 
It is possible to use those as `FROM` for your `Dockerfile`.
For example: `FROM datamachines/cuda_tensorflow_opencv:9.0_1.12.0_4.1.0-20190605` **

## Docker images tag naming

The image tags follow the `cuda_tensorflow_opencv` naming order.
As such `9.0_1.12.0_4.1.0` refers to *Cuda 9.0*, *TensorFlow 1.12.0* and *OpenCV 4.1.0*, and `10.0_1.13.1_3.4.6` refers to *Cuda 10.0*, *TensorFlow 1.13.1* and *OpenCV 3.4.6*.

Docker images also tagged with a version information for the date (YYYYMMDD) of the Dockerfile against which they were built from, added at the end of the tag string (following a dash character), such that `cuda_tensorflow_opencv:9.0_1.12.0_4.1.0-20190605` is for *Dockerfile dating June 5th, 2019*.

## Building the images

Use the provided `Makefile` to get a list of tags.
Use `make build_all` to build all containers.
Note that the tag for those images does not contain the `datamachines/` organization addition that is found in the publicly released pre-built container images.

## Using the container images

The use of the provided `runDocker.sh` script  present in the directory to test the built image; it will set up the X11 passthrough and give the use a prompt, as well as mount the calling directory as `/dmc`.The user can test X11 is functional by using a simple X command such as `xlogo` from the command line.

Note that the base container runs as root, if you want to run it as a non root user, add `-u $(id -u):$(id -g)` to the `nvidia-docker` command line but ensure that you have access to the directories you will work in.

To use it, the full tag of the container image should be passed as the `CTO_TAG` environment variable. For example, to use `9.0_1.12.0_4.1.0-20190605`, run `CTO_TAG=9.0_1.12.0_4.1.0-20190605 ./runDocker.sh`. Note that `./` can be replaced by the location of `runDocker.sh` so that the user can mount its current working directory as `/dmc` in order to access local files.

For example, if the user place a picture (named `pic.jpg`) in the directory to be mount as `/dmc` and the following example script (naming it `display_pic.py3`)

    import numpy as np
    import cv2

    img = cv2.imread('pic.jpg')
    print(img.shape, " ", img.size)
    cv2.imshow('image', img)
    cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()

, adapting "PATH_TO_RUNDOCKER" in `CTO_TAG=9.0_1.12.0_4.1.0-20190605 PATH_TO_RUNDOCKER/runDocker.sh`.
From the bash interactive shell, type `cd /dmc; python3 display_pic.py3` will display the picture on the user's X11 display.

## Version specific information

(not all versions available are listed below, only the base builds are detailed)

### 10.0_1.13.1_4.1.0

`Dockerfile`  using `FROM tensorflow/tensorflow:1.13.1-gpu-py3` 
For more information, see https://github.com/tensorflow/tensorflow/releases/tag/v1.13.1 and https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/docker, knowing that `TensorFlow GPU binaries are now built against CUDA 10 and TensorRT 5.0`.
Recommended reading on Tensorflow Docker GPU at https://www.tensorflow.org/install/docker#gpu_support

The TensorFlow Docker image is from a Ubuntu 16.04 (use `lsb_release -a`) and has CUDA 10.0 libraries available (`dpkg -l | grep cuda`). For more details, see https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/10.0/runtime/cudnn7/Dockerfile

### 9.0_1.12.0_3.4.6

`Dockerfile` using `FROM tensorflow/tensorflow:1.12.0-gpu-py3` using Ubuntu 16.04 and CUDA 9.0

## Using the tools

### Using TensorFlow in your code

Code written for Tensorflow should follow principles described in https://www.tensorflow.org/guide/using_gpu

In particular, the following section https://www.tensorflow.org/guide/using_gpu#allowing_gpu_memory_growth might be needed to allow proper use of the GPU's memory. In particular:
   
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config, ...)

Note that this often allocates all the GPU memory to one Tensorflow client. If you intend to run multiple Tensorflow containers, limiting the available memory available to the container's Tensorflow can be achieved as described in https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory by instead specifying the percentage of the GPU memory to be used:

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction=0.125
    session = tf.Session(config=config, ...)

The built Docker images do NOT install any models, add/build/download your own in your `Dockerfile` that is `FROM cuda_tensorflow_opencv:9.0_1.12.0_4.1.0-20190605`

For example:

    FROM cuda_tensorflow_opencv:9.0_1.12.0_4.1.0-20190605
    
    # Download tensorflow object detection models
    RUN GIT_SSL_NO_VERIFY=true git clone -q https://github.com/tensorflow/models /usr/local/lib/python3.5/dist-packages/tensorflow/models

    # Install downloaded models
    ENV PYTHONPATH "$PYTHONPATH:/usr/local/lib/python3.5/dist-packages/tensorflow/models/research:/usr/local/lib/python3.5/dist-packages/tensorflow/models/research/slim"
    RUN cd /usr/local/lib/python3.5/dist-packages/tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=.

### CuDNN 

Because CuDNN needs to be downloaded from Nvidia with a developer account, it is not included in any builds. If you want to add it to your own builds, the simplest way is to use the `cuda_tensorflow_opencv` as a `FROM` and install the needed `.deb` files

For example, basing it on the CUDA 9.0 public image: create a new build directory, and download in this directory the main/dev/doc `deb` files for cudnn (for Ubuntu 16.04 and CUDA 9.0) that can be retrieved from https://developer.nvidia.com/rdp/cudnn-download (registration required)

Use/Adapt the following `Dockerfile` for your need:

	FROM cuda_tensorflow_opencv:9.0_1.12.0_4.1.0-20190605
	
	# Tagged build: docker build --tag="cudnn_tensorflow_opencv:9.0_1.12.0_4.1.0-20190605" .
	# Tag version kept in sync with the cuda_tensorflow_opencv one it is "FROM"
	
	RUN mkdir /tmp/cudnn
	COPY *.deb /tmp/cudnn/
	RUN dpkg -i /tmp/cudnn/*.deb && rm -rf /tmp/cudnn 
	
Warning: This build will copy any `.deb`present in the directory where the `Dockerfile` is found

### OpenCV and GPU

OpenCV is compiled with CUDA support, but note that not all of OpenCV's functions are CUDA optimized. This is true in particular for some of the `contrib` code.