# DockerFile with Nvidia GPU support for TensorFlow and OpenCV
Revision: 20200423

The base OS for those container images is Ubuntu 18.04 or DockerHub's `nvidia/cuda` based on Ubuntu 18.04. 
More details on the Nvidia base images are available at https://hub.docker.com/r/nvidia/cuda/ . 
In particular, please note that "By downloading these images, you agree to the terms of the license agreements for NVIDIA software included in the images"; with further details on DockerHub version from https://docs.nvidia.com/cuda/eula/index.html#attachment-a


As of the `20191107` Dockerfile version, it also builds a non-CUDA version: `tensorflow_opencv`.

As of the `20191210` Dockerfile version, it also builds a CuDNN version: `cudnn_tensorflow_opencv`

As of the `20200211` Dockerfile version, we are making use of Docker 19.03's GPU support and are adding information about the OpenCV builds in the `OpenCV_BuildConf` directory.

As of the `20200327` Dockerfile version, we have added Protobuf, WebP, GStreamer and Eigen to the OpenCV build. 

As of the `20200423` Dockerfile version, we have added support for OpenCV 3.4.10 and 4.3.0, and added GStreamer plugins to the build.
We have also added Nvidia Jetson Nano build steps in the `JetsonNano` directory.

`cuda_tensorflow_opencv`:
- Builds an Nvidia GPU optimized version of TensorFlow and OpenCV. Also install, Jupyter, Keras, numpy, pandas and X11 support.
- Requires a Linux system with nvidia-docker (v2) and the Nvidia drivers installed to run. See https://github.com/NVIDIA/nvidia-docker for setup details

`cudnn_tensorflow_opencv`:
- Similar to `cuda_tensorflow_opencv` but with CuDNN installed and used for OpenCV compilation (this was more deeply integrated within OpenCV after October 2019, see [CUDA backend for the DNN module](https://github.com/opencv/opencv/pull/14827) for additional details).
- For CUDNN, the CUDA backend for DNN module requires CC 5.3 or higher; please see https://en.wikipedia.org/wiki/CUDA#GPUs_supported to confirm your architecture is supported

`tensorflow_opencv`:
- Builds a similar container with a version of TensorFlow and OpenCV. Also install, Jupyter, Keras, numpy, pandas and X11 support.
- Can be used on systems without a Nvidia GPU, and the `runDocker.sh` script will setup proper X11 passthrough
- for MacOS X11 passthrough, install the latest XQuartz server and activate the `Security -> Allow connections from network clients` (must logout for it to take effect)

`jetsonnano-cuda_tensorflow_opencv` (see the `JetsonNano` directory):
- Builds a Nvidia Jetson Nano `cuda_tensorflow_opencv` container image based on Nvidia's provided `l4t-base` container and adapted from the `Makefile` and `Dockerfile` used for the other builds.

**Docker Images built from this repository are publicly available at https://hub.docker.com/r/datamachines/tensorflow_opencv / https://hub.docker.com/r/datamachines/cuda_tensorflow_opencv / https://hub.docker.com/r/datamachines/cudnn_tensorflow_opencv / https://hub.docker.com/r/datamachines/jetsonnano-cuda_tensorflow_opencv .  The [Builds-DockerHub.md](https://github.com/datamachines/cuda_tensorflow_opencv/blob/master/Builds-DockerHub.md) file is a quick way of seeing the list of pre-built container images**

It is possible to use those as `FROM` for your `Dockerfile`; for example: `FROM datamachines/cuda_tensorflow_opencv:10.2_1.15_3.4.8-20191210`

## Docker images tag naming

The image tags follow the `cuda_tensorflow_opencv` naming order.
As such `10.2_1.15_3.4.8-20191210` refers to *Cuda 10.2*, *TensorFlow 1.15* and *OpenCV 3.4.8*.

Docker images are also tagged with a version information for the date (YYYYMMDD) of the Dockerfile against which they were built from, added at the end of the tag string (following a dash character), such that `cuda_tensorflow_opencv:10.2_1.15_3.4.8-20191210` is for the *Dockerfile dating December 10th, 2019*.

Similarly, the `tensorflow_opencv` and `cudnn_tensorflow_opencv` tags follow the same naming convention.

## Building the images

The tag for any image built will contain the `datamachines/` organization addition that is found in the publicly released pre-built container images.

Use the provided `Makefile` by running `make` to get a list of targets to build:
- `make build_all` will build all container images
- `make tensorflow_opencv` to build all the `tensorflow_opencv` container images
- `make cuda_tensorflow_opencv` will build all the `cuda_tensorflow_opencv` container images
- `make cudnn_tensorflow_opencv` will build all the `cudnn_tensorflow_opencv` container images
- use a direct tag to build a specific version; for example `make cudnn_tensorflow_opencv-10.2_2.0_4.1.2`, will build the `datamachines/cudnn_tensorflow_opencv:10.2_2.0_4.1.2-20191210` container image  (if such a built is available, see the `Docker Image tag ending` and the list of `Available Docker images to be built` for accurate values).

If you have a system available to run the `build_all`, sometimes OpenCV will fail, the following `bash` is useful to keep building: `while true; do make -i build_all ; sleep 200; done`. Just be ready to `Ctrl+c` it when done/ready (completion can be seen by the matching log files).

## Using the container images

The use of the provided `runDocker.sh` script present in the source directory allows users to utilize the built image. Dy default, it will set up the X11 passthrough (for Linux and MacOS) and give the user a `/bin/bash` prompt within the running container, as well as mount the calling directory as `/dmc`. A user can test that X11 is functional by using a simple X command such as `xlogo` from the command line.

To use it, the full name of the container image should be passed as the `CONTAINER_ID` environment variable. For example, to use `datamachines/cudnn_tensorflow_opencv-10.2_2.0_4.1.2-20191210`, run `CONTAINER_ID=datamachines/cudnn_tensorflow_opencv-10.2_2.0_4.1.2-20191210 ./runDocker.sh`. Note that `runDocker.sh` can be called from any location using its full path, so that a user can mount its current working directory as `/dmc` in the running container in order to access local files.

`runDocker.sh` can take multiple arguments; running it without any argument will provide a list of those arguments.

As of Docker 19.03, GPU support is native to the container runtime, as such, we have shifted from the use of `nvidia-docker` to the native `docker [...] --gpus all`. We understand not every user want to use all the GPUs installed on his system, as such, to change this option, change the `D_GPUS` line in the first few lines of `runDocker.sh` to reflect the paramaters that best reflect your system or needs. GPU support is only enabled for the `cuda_` and `cudnn_` images.

Note that the base container runs as root, if you want to run it as a non root user, add `-u $(id -u):$(id -g)` to the `docker` command line but ensure that you have access to the directories you will work in. This can be done using the `-e` command line option of `runDocker.sh`.

### Making use of the container

If a user place a picture (named `pic.jpg`) in the directory to be mounted as `/dmc` and the following example script (naming it `display_pic.py3`)

    import numpy as np
    import cv2

    img = cv2.imread('pic.jpg')
    print(img.shape, " ", img.size)
    cv2.imshow('image', img)
    cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()

, adapting `PATH_TO_RUNDOCKER` in `CONTAINER_ID=datamachines/cudnn_tensorflow_opencv-10.2_2.0_4.1.2-20191210 PATH_TO_RUNDOCKER/runDocker.sh`, from the provided bash interactive shell, when the user runs `cd /dmc; python3 display_pic.py3`, this will display the picture from the mounted directory on the user's X11 display.

## Additional usage options

### Using GPU TensorFlow in your code

Code written for Tensorflow should follow principles described in https://www.tensorflow.org/guide/using_gpu

In particular, the following section https://www.tensorflow.org/guide/using_gpu#allowing_gpu_memory_growth might be needed to allow proper use of the GPU's memory. In particular:
   
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config, ...)

Note that this often allocates all the GPU memory to one Tensorflow client. If you intend to run multiple Tensorflow containers, limiting the available memory available to the container's Tensorflow can be achieved as described in https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory by instead specifying the percentage of the GPU memory to be used:

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction=0.125
    session = tf.Session(config=config, ...)

The built Docker images do NOT install any models, add/build/download your own in your `Dockerfile` that is `FROM datamachines/cudnn_tensorflow_opencv-10.2_2.0_4.1.2-20191210`

For example:

    FROM datamachines/cudnn_tensorflow_opencv-10.2_2.0_4.1.2-20191210
    
    # Download tensorflow object detection models
    RUN GIT_SSL_NO_VERIFY=true git clone -q https://github.com/tensorflow/models /usr/local/lib/python3.6/dist-packages/tensorflow/models

    # Install downloaded models
    ENV PYTHONPATH "$PYTHONPATH:/usr/local/lib/python3.6/dist-packages/tensorflow/models/research:/usr/local/lib/python3.6/dist-packages/tensorflow/models/research/slim"
    RUN cd /usr/local/lib/python3.6/dist-packages/tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=.

### A note about OpenCV and GPU

In `cuda_tensorflow_opencv` (resp. `cudnn_tensorflow_opencv`), OpenCV is compiled with CUDA (resp. CUDA+CuDNN support), but note that not all of OpenCV's functions are optimized. This is true in particular for some of the `contrib` code.

### A note on exposing ports

By choice, the containers built do not expose any ports, or start any services. This is left to the end-user. To start any, the simpler solution is to base a new container `FROM` one of those containers, expose a port and start said service to be able to access it.

For example, the start and expose Jupyter Notebook (on port `8888`) from the `tensorflow_opencv` container, one could write the following `Dockerfile` and tag it as `jupnb:local`:
<pre>
FROM datamachines/tensorflow_opencv:2.1.0_4.3.0-20200423
EXPOSE 8888
CMD jupyter-notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
</pre>
, using `docker build --tag jupnb:local .`


When starting it using `docker run -p 8888:8888 jupnb:local` to publish the container's port `8888` to the local system's port `8888`, an `http://127.0.0.1:8888/` based URL will shown with the access token.
Using this url in a web browser will grant access to the running instance of Jupyter Notebook.
