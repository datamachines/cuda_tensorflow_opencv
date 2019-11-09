# DockerFile with Nvidia GPU support for TensorFlow and OpenCV
Revision: 20191108

As of `20191107` Dockerfile version, also builds a non-CUDA version: `tensorflow_opencv`

The base OS for those container images is Ubuntu 18.04 (as the base `tensorflow/tensorflow` container image moved to Ubuntu 18.04 as of images built after May 20 2019).

`cuda_tensorflow_opencv`:
- Builds an Nvidia GPU optimized version of TensorFlow and OpenCV. Also install, Jupyter, Keras, numpy, pandas and X11 support.
- Requires a Linux system with nvidia-docker (v2) and the Nvidia drivers installed to run. See https://github.com/NVIDIA/nvidia-docker for setup details

`tensorflow_opencv`:
- Builds a similar container with a version of TensorFlow and OpenCV. Also install, Jupyter, Keras, numpy, pandas and X11 support.
- Can be used on systems without a Nvidia GPU, and the `runDocker.sh` script will setup proper X11 passthrough
- for MacOS X11 passthrough, install the latest XQuartz server and activate the `Security -> Allow connections from network clients` (must logout for it to take effect)

**Docker Images built from this repository are publicly available at https://hub.docker.com/r/datamachines/cuda_tensorflow_opencv and https://hub.docker.com/r/datamachines/tensorflow_opencv**

It is possible to use those as `FROM` for your `Dockerfile`; for example: `FROM datamachines/cuda_tensorflow_opencv:10.0_1.13.2_4.1.2-20191108`

## Docker images tag naming

The image tags follow the `cuda_tensorflow_opencv` naming order.
As such `10.0_1.13.2_4.1.2` refers to *Cuda 10.0*, *TensorFlow 1.13.2* and *OpenCV 4.1.2*.

Docker images are also tagged with a version information for the date (YYYYMMDD) of the Dockerfile against which they were built from, added at the end of the tag string (following a dash character), such that `cuda_tensorflow_opencv:10.0_1.13.2_4.1.2-20191108` is for *Dockerfile dating November 8th, 2019*.

Similarly, the `tensorflow_opencv` tag follows the same naming convention, and `1.13.2_4.1.2-20191108` refers to *Tensorflow 1.13.2*, *OpenCV 4.1.2*. with a *Dockerfile dating November 8th, 2019*.

## Building the images

The tag for any image built will contain the `datamachines/` organization addition that is found in the publicly released pre-built container images.

Use the provided `Makefile` by running `make` to get a list of targets to build:
- `make build_all` will build all container images
- `make cuda_tensorflow_opencv` will build all the `cuda_tensorflow_opencv` container images
- `make tensorflow_opencv` to build all the `tensorflow_opencv` container images
- use a direct tag to build a specific version; for example `make 10.0_1.13.2_4.1.2`, will build the `datamachines/cuda_tensorflow_opencv:10.0_1.13.2_4.1.2-20191108` container image.

## Using the container images

The use of the provided `runDocker.sh` script present in the source directory allows users to utilize the built image. Dy default, it will set up the X11 passthrough (for Linux and MacOS) and give the user a `/bin/bash` prompt within the running container, as well as mount the calling directory as `/dmc`. A user can test that X11 is functional by using a simple X command such as `xlogo` from the command line.

To use it, the full name of the container image should be passed as the `CONTAINER_ID` environment variable. For example, to use `datamachines/cuda_tensorflow_opencv:10.0_1.13.2_4.1.2-20191108`, run `CTO=datamachines/cuda_tensorflow_opencv:10.0_1.13.2_4.1.2-20191108 ./runDocker.sh`. Note that `runDocker.sh` can be called from any location using its full path, so that a user can mount its current working directory as `/dmc` in the running container in order to access local files.

`runDocker.sh` can take multiple arguments; running it without any argument will provide a list of those arguments.

Note that the base container runs as root, if you want to run it as a non root user, add `-u $(id -u):$(id -g)` to the `nvidia-docker`/`docker` command line but ensure that you have access to the directories you will work in. This can be done using the `-e` command line option of `runDocker.sh`.

### Example

If a user place a picture (named `pic.jpg`) in the directory to be mounted as `/dmc` and the following example script (naming it `display_pic.py3`)

    import numpy as np
    import cv2

    img = cv2.imread('pic.jpg')
    print(img.shape, " ", img.size)
    cv2.imshow('image', img)
    cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()

, adapting `PATH_TO_RUNDOCKER` in `CTO=datamachines/cuda_tensorflow_opencv:10.0_1.13.2_4.1.2-20191108 PATH_TO_RUNDOCKER/runDocker.sh`, from the provided bash interactive shell, when the user runs `cd /dmc; python3 display_pic.py3`, this will display the picture from the mounted directory on the user's X11 display.

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

The built Docker images do NOT install any models, add/build/download your own in your `Dockerfile` that is `FROM datamachines/cuda_tensorflow_opencv:10.0_1.13.2_4.1.2-20191108`

For example:

    FROM datamachines/cuda_tensorflow_opencv:10.0_1.13.2_4.1.2-20191108
    
    # Download tensorflow object detection models
    RUN GIT_SSL_NO_VERIFY=true git clone -q https://github.com/tensorflow/models /usr/local/lib/python3.5/dist-packages/tensorflow/models

    # Install downloaded models
    ENV PYTHONPATH "$PYTHONPATH:/usr/local/lib/python3.5/dist-packages/tensorflow/models/research:/usr/local/lib/python3.5/dist-packages/tensorflow/models/research/slim"
    RUN cd /usr/local/lib/python3.5/dist-packages/tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=.

### CuDNN 

Because CuDNN needs to be downloaded from Nvidia with a developer account, it is not included in any builds. If you want to add it to your own builds, the simplest way is to use the `cuda_tensorflow_opencv` as a `FROM` and install the needed `.deb` files

For example, basing it on the CUDA 10.0 public image: create a new build directory, and download in this directory the main/dev/doc `deb` files for cudnn (for Ubuntu 18.04 and CUDA 10.0) that can be retrieved from https://developer.nvidia.com/rdp/cudnn-download (registration required)

Use/Adapt the following `Dockerfile` for your need:

	FROM datamachines/cuda_tensorflow_opencv:10.0_1.13.2_4.1.2-20191108
	
	# Tagged build: docker build --tag="cudnn_tensorflow_opencv:10.0_1.13.2_4.1.2-20191108" .
	# Tag version kept in sync with the datamachines/cuda_tensorflow_opencv one it is "FROM"
	
	RUN mkdir /tmp/cudnn
	COPY *.deb /tmp/cudnn/
	RUN dpkg -i /tmp/cudnn/*.deb && rm -rf /tmp/cudnn 
	
Warning: This build will copy any `.deb`present in the directory where the `Dockerfile` is found

### OpenCV and GPU

In `cuda_tensorflow_opencv`, OpenCV is compiled with CUDA support, but note that not all of OpenCV's functions are CUDA optimized. This is true in particular for some of the `contrib` code.
