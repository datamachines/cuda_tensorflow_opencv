ARG CTO_FROM
FROM ${CTO_FROM}

##### Splitting installation into controlled parts:
## - base container setup (all apt-get + system + ldd extension)
## - Python pip3 preparation + First batch of python tools installs
## - Tensorflow build and install
## - FFmpeg build and install
## - OpenCV build and install (using FFmpeg)
## - Additional python tools installs
## - Magma build and install (GPU only)
## - PyTorch (+ vison + audio) build and install (using FFmpeg + OpenCV)
## - Final steps: /dmc setup, ...

##### Base

# Install system packages
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y --fix-missing\
  && apt-get install -y \
    apt-utils \
    locales \
    wget \
    ca-certificates \
  && apt-get clean

# UTF-8
RUN localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8
ENV LANG en_US.utf8

# Install system packages
RUN apt-get install -y \
  # Base (min for TensorFlow)
    build-essential \
    checkinstall \
    cmake \
    curl \
    g++ \
    gcc \
    git \
    perl \
    pkg-config \
    protobuf-compiler \
    python3-dev \
    rsync \
    unzip \
    wget \
    zip \
    zlib1g-dev \
  # OpenCV
    doxygen \
    file \
    gfortran \
    gnupg \
    gstreamer1.0-plugins-good \
    imagemagick \
    libatk-adaptor \
    libatlas-base-dev \
    libboost-all-dev \
    libcanberra-gtk-module \
    libdc1394-22-dev \
    libeigen3-dev \
    libfaac-dev \
    libfreetype6-dev \
    libgflags-dev \
    libglew-dev \
    libglu1-mesa \
    libglu1-mesa-dev \
    libgoogle-glog-dev \
    libgphoto2-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-bad1.0-0 \
    libgstreamer-plugins-base1.0-dev \
    libgtk2.0-dev \
    libgtk-3-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libjpeg-dev \
    liblapack-dev \
    libmp3lame-dev \
    libopenblas-dev \
    libopencore-amrnb-dev \
    libopencore-amrwb-dev \
    libopenjp2-7 \
    libopenjp2-7-dev \
    libopenjp2-tools \
    libopenjpip-server \
    libpng-dev \
    libpostproc-dev \
    libprotobuf-dev \
    libtbb2 \
    libtbb-dev \
    libtheora-dev \
    libtiff5-dev \
    libv4l-dev \
    libvorbis-dev \
    libwebp-dev \
    libx264-dev \
    libx265-dev \
    libxi-dev \
    libxine2-dev \
    libxmu-dev \
    libxvidcore-dev \
    libzmq3-dev \
    v4l-utils \
    x11-apps \
    x264 \
    yasm \
  # Torch
    libomp-dev \
    libsox-dev \
    libsox-fmt-all \
    libsphinxbase-dev \
    sphinxbase-utils \
    zlib1g \
  # FFMpeg (source install, do not install packages: libavcodec-dev libavformat-dev libavresample-dev libavutil-dev libswscale-dev)
    libass-dev \
    libc6 \
    libc6-dev \
    libnuma1 \
    libnuma-dev \
    libopus-dev \
    libtool \
    libvpx-dev \
  && apt-get clean

# Additional specialized apt installs
ARG CTO_CUDA_APT
ARG CTO_CUDA11_APT_XTRA
RUN apt-get install -y --no-install-recommends ${CTO_CUDA11_APT_XTRA} \
      time ${CTO_CUDA_APT} \
  && apt-get clean

# CUPTI library needed by TensorFlow1 but sometimes not in default path, adding if at unconventional location
# Also set the /tmp/.{CPU,GPU}_build file
ARG CTO_BUILD
COPY tools/cupti_helper.sh /tmp/
RUN chmod +x /tmp/cupti_helper.sh \
  && /tmp/cupti_helper.sh ${CTO_BUILD} \
  && rm /tmp/cupti_helper.sh

# Prepare ldconfig
RUN mkdir -p /usr/local/lib \
  && sh -c 'echo "/usr/local/lib" >> /etc/ld.so.conf.d/usrlocallib.conf' \
  && ldconfig

# Setup pip
RUN wget -q -O /tmp/get-pip.py --no-check-certificate https://bootstrap.pypa.io/get-pip.py \
  && python3 /tmp/get-pip.py \
  && pip3 install -U pip \
  && rm /tmp/get-pip.py

# Some TF tools expect a "python" binary
RUN ln -s $(which python3) /usr/local/bin/python

# Install Python tools (for buiding) 
ARG CTO_TF_NUMPY=numpy
RUN pip3 install -U \
# TensorFlow
    mock \
    ${CTO_TF_NUMPY} \
    setuptools \
    six \
    wheel \
    future \
    packaging \
  # OpenCV
    Pillow \
    lxml \
  # Torch
    pyyaml \
    mkl \
    mkl-include \
    cmake \
    cffi \
    typing \
    ninja \
  # Extra
    scikit-image \
    scikit-learn \
  && pip3 install -U keras_applications --no-deps \
  && pip3 install -U keras_preprocessing --no-deps \
  && rm -rf /root/.cache/pip


##### TensorFlow

## Download & Building TensorFlow from source in same RUN
ARG LATEST_BAZELISK=1.5.0
ARG LATEST_BAZEL=3.4.1
ARG CTO_TENSORFLOW_VERSION
ARG CTO_TF_OPT=""
ARG CTO_DNN_ARCH=""
COPY tools/tf_build.sh /tmp/
RUN curl -s -Lo /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/download/v${LATEST_BAZELISK}/bazelisk-linux-amd64 \
  && chmod +x /usr/local/bin/bazel \
  && mkdir -p /usr/local/src/tensorflow \
  && cd /usr/local/src \
  && wget -q --no-check-certificate -c https://github.com/tensorflow/tensorflow/archive/v${CTO_TENSORFLOW_VERSION}.tar.gz -O - | tar --strip-components=1 -xz -C /usr/local/src/tensorflow \
  && cd /usr/local/src/tensorflow \
  && fgrep _TF_MAX_BAZEL configure.py | grep '=' | perl -ne '$lb="'${LATEST_BAZEL}'";$brv=$1 if (m%\=\s+.([\d\.]+).$+%); sub numit{@g=split(m%\.%,$_[0]);return(1000000*$g[0]+1000*$g[1]+$g[2]);}; if (&numit($brv) > &numit($lb)) { print "$lb" } else {print "$brv"};' > .bazelversion \
  && bazel clean \
  && chmod +x /tmp/tf_build.sh \
  && time /tmp/tf_build.sh ${CTO_TF_OPT} \
  && time ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg \
  && time pip3 install /tmp/tensorflow_pkg/tensorflow-*.whl \
  && rm -rf /usr/local/src/tensorflow /tmp/tensorflow_pkg /tmp/bazel_check.pl /tmp/tf_build.sh /tmp/hsperfdata_root /root/.cache/bazel /root/.cache/pip /root/.cache/bazelisk
RUN python3 -c "import tensorflow"


##### FFMPEG

ARG CTO_FFMPEG_VERSION="4.4.2"
ARG CTO_FFMPEG_NVCODEC="11.1.5.1"
ARG CTO_FFMPEG_NONFREE=""
RUN if [ -f /tmp/.GPU_build ]; \
  then \
    mkdir -p /usr/local/src/builder; \
    cd /usr/local/src; \
    wget -q --no-check-certificate https://github.com/FFmpeg/nv-codec-headers/archive/refs/tags/n${CTO_FFMPEG_NVCODEC}.tar.gz -O - | tar --strip-components=1 -xz -C /usr/local/src/builder; \
    cd /usr/local/src/builder; \
    make install; \
    rm -rf /usr/local/src/builder; \
  fi
RUN mkdir -p /usr/local/src/builder \
  && cd /usr/local/src \
  && wget -q --no-check-certificate https://ffmpeg.org/releases/ffmpeg-${CTO_FFMPEG_VERSION}.tar.gz -O - | tar --strip-components=1 -xz -C /usr/local/src/builder \
  && cd /usr/local/src/builder \
  && if [ -f /tmp/.GPU_build ]; \
    then \
      time ./configure --enable-cuda --enable-cuvid --enable-nvdec --enable-nvenc ${CTO_FFMPEG_NONFREE} --extra-cflags="-I/usr/local/cuda/include/ -fPIC" --extra-ldflags="-L/usr/local/cuda/lib64/ -Wl,-Bsymbolic" --enable-shared --disable-static --enable-gpl --enable-libv4l2 --enable-libvorbis --enable-libvpx --enable-libwebp --enable-libxvid --enable-libopus --enable-pic --enable-libass --enable-libx264 --enable-libx265 | tee /tmp/ffmpeg_config.txt; \
    else \
      time ./configure --enable-shared --disable-static --enable-gpl --enable-libv4l2 --enable-libvorbis --enable-libvpx --enable-libwebp --enable-libxvid --enable-libopus --enable-pic --enable-libass --enable-libx264 --enable-libx265 | tee /tmp/ffmpeg_config.txt; \
    fi \
  && make -j${CTO_NUMPROC} install \
  && ldconfig \
  && rm -rf /usr/local/src/builder
# From https://docs.nvidia.com/video-technologies/video-codec-sdk/ffmpeg-with-nvidia-gpu/#basic-testing
# GPU Testing: ffmpeg -y -vsync 0 -hwaccel cuda -hwaccel_output_format cuda -i input.mp4 -c:a copy -c:v h264_nvenc -b:v 5M output.mp4


##### OpenCV

# Download & Build OpenCV in same RUN
## FYI: We are removing the OpenCV source and build directory in /usr/local/src to attempt to save additional disk space
# Comment the last line if you want to rerun cmake with additional/modified options. For example:
# cd /usr/local/src/opencv/build
# cmake -DOPENCV_ENABLE_NONFREE=ON -DBUILD_EXAMPLES=ON -DBUILD_DOCS=ON -DBUILD_TESTS=ON -DBUILD_PERF_TESTS=ON .. && make install
ARG CTO_OPENCV_VERSION
ARG CTO_NUMPROC=1
ARG CTO_CUDA_BUILD
ARG CTO_OPENCV_NONFREE=""
RUN mkdir -p /usr/local/src/opencv/build /usr/local/src/opencv_contrib \
  && cd /usr/local/src \
  && wget -q --no-check-certificate https://github.com/opencv/opencv/archive/${CTO_OPENCV_VERSION}.tar.gz -O - | tar --strip-components=1 -xz -C /usr/local/src/opencv \
  && wget -q --no-check-certificate https://github.com/opencv/opencv_contrib/archive/${CTO_OPENCV_VERSION}.tar.gz -O - | tar --strip-components=1 -xz -C /usr/local/src/opencv_contrib \
  && cd /usr/local/src/opencv/build \
  && time cmake \
    -DBUILD_DOCS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_opencv_python2=OFF \
    -DBUILD_opencv_python3=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local/ \
    -DCMAKE_INSTALL_TYPE=Release \
    -DENABLE_FAST_MATH=1 \
    -DFORCE_VTK=ON \
    -DINSTALL_C_EXAMPLES=OFF \
    -DINSTALL_PYTHON_EXAMPLES=OFF \
    -DOPENCV_EXTRA_MODULES_PATH=/usr/local/src/opencv_contrib/modules \
    -DOPENCV_GENERATE_PKGCONFIG=ON \
    -DOPENCV_PC_FILE_NAME=opencv.pc \
    -DOPENCV_PYTHON3_INSTALL_PATH=/usr/lib/python3/dist-packages \
    -DWITH_CSTRIPES=ON \
    -DWITH_EIGEN=ON \
    -DWITH_GDAL=ON \
    -DWITH_GSTREAMER=ON \
    -DWITH_GSTREAMER_0_10=OFF \
    -DWITH_GTK=ON \
    -DWITH_IPP=ON \
    -DWITH_OPENCL=ON \
    -DWITH_OPENMP=ON \
    -DWITH_TBB=ON \
    -DWITH_V4L=ON \
    -DWITH_WEBP=ON \
    -DWITH_XINE=ON \
    ${CTO_CUDA_BUILD} \
    ${CTO_OPENCV_NONFREE} \
    .. \
  && time make -j${CTO_NUMPROC} install \
  && ldconfig \
  && rm -rf /usr/local/src/opencv /usr/local/src/opencv_contrib
RUN python3 -c "import cv2"


##### Additional Python tools

# Install additional Python tools 
RUN pip3 install -U \
  autovizwidget \
  ipython \
  jupyter \
  matplotlib \
  moviepy \
  notebook \
  pandas \
  && rm -rf /root/.cache/pip
# Removed scipy: the version required by TF2 is already installed during the TF2 step (and it will be installed by Keras for TF1)

# Installing a built-TF compatible keras (for TF1) and PyTorch
ARG CTO_TF_KERAS
RUN pip3 install ${CTO_TF_KERAS}  \
  && rm -rf /root/.cache/pip


##### Magma (for PyTorch GPU only)

ARG CTO_MAGMA=2.6.2
RUN if [ -f /tmp/.GPU_build ]; \
  then \
    mkdir -p /usr/local/src/builder/build; \
    cd /usr/local/src; \
    wget -q --no-check-certificate http://icl.utk.edu/projectsfiles/magma/downloads/magma-${CTO_MAGMA}.tar.gz -O - | tar --strip-components=1 -xz -C /usr/local/src/builder; \
    cd /usr/local/src/builder/build; \
    time cmake -DGPU_TARGET="Pascal Volta Ampere" .. | tee /tmp/magma_build.txt; \
    time make -j${CTO_NUMPROC} install; \
    rm -rf /usr/local/src/magma; \
  fi


##### Torch (using FFMpeg + OpenCV + Magma)

ARG CTO_TORCH=1.11
RUN mkdir -p /usr/local/src \
  && cd /usr/local/src \
  && git clone --depth 1 --branch release/${CTO_TORCH}  https://github.com/pytorch/pytorch.git  \
  && cd /usr/local/src/pytorch \
  && if [ -f /tmp/.GPU_build ]; \
    then \
      time env USE_CUDA=ON USE_CUDNN=ON TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" USE_MKLDNN=ON USE_FFMPEG=ON USE_OPENCV=ON python3 ./setup.py bdist_wheel | tee /tmp/torch_config.txt; \
    else \
      time env USE_CUDA=OFF USE_CUDNN=OFF USE_MKLDNN=ON USE_FFMPEG=ON USE_OPENCV=ON python3 ./setup.py bdist_wheel | tee /tmp/torch_config.txt; \
    fi \
  && time pip3 install /usr/local/src/pytorch/dist/*.whl \
  && cd /tmp \
  && perl -i.bak -pe 'exit if m%^-- Configuring done%' torch_config.txt \
  && sh -c "cmp --silent torch_config.txt torch_config.txt.bak && exit 1 || rm torch_config.txt.bak" \
  && ldconfig \
  && rm -rf /root/.cache/pip /usr/local/src/pytorch
RUN python3 -c "import torch"

ARG CTO_TORCHVISION=0.12
RUN mkdir -p /usr/local/src \
  && cd /usr/local/src \
  && git clone --depth 1 --branch release/${CTO_TORCHVISION}  https://github.com/pytorch/vision.git  \
  && cd /usr/local/src/vision \
  && time sh -c "python3 ./setup.py bdist_wheel | tee /tmp/torchvision_config.txt" \
  && time pip3 install /usr/local/src/vision/dist/*.whl \
  && cd /tmp \
  && perl -i.bak -pe 'exit if m%^running bdist_wheel%' torchvision_config.txt \
  && sh -c "cmp --silent torchvision_config.txt torchvision_config.txt.bak && exit 1 || rm torchvision_config.txt.bak" \
  && ldconfig \
  && rm -rf /root/.cache/pip /usr/local/src/vision
RUN python3 -c "import torchvision"

ARG CTO_TORCHAUDIO=0.11
RUN mkdir -p /usr/local/src/builder \
  && cd /usr/local/src \
  && git clone --depth 1 --branch release/${CTO_TORCHAUDIO}  https://github.com/pytorch/audio.git \
  && cd /usr/local/src/audio \
  && time sh -c "python3 ./setup.py bdist_wheel | tee /tmp/torchaudio_config.txt" \
  && time pip3 install /usr/local/src/audio/dist/*.whl \
  && cd /tmp \
  && perl -i.bak -pe 'exit if m%^-- Configuring done%' torchaudio_config.txt \
  && sh -c "cmp --silent torchaudio_config.txt torchaudio_config.txt.bak && exit 1 || rm torchaudio_config.txt.bak" \
  && ldconfig \
  && rm -rf /root/.cache/pip /usr/local/src/builder
RUN python3 -c "import torchaudio"


##### Final steps

# Add dataframe display widget
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension

# Tool to dump some installation details
COPY tools/tf_info.sh /tmp/
RUN chmod +x /tmp/tf_info.sh \
  && touch /.within_container

#ENV NVIDIA_VISIBLE_DEVICES all
#ENV NVIDIA_DRIVER_CAPABILITIES all

ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH

# Setting up working directory 
RUN mkdir /dmc
WORKDIR /dmc

CMD bash

LABEL "Author"="Data Machines Corp <help@datamachines.io>"
