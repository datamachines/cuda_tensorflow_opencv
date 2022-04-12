ARG CTO_FROM
FROM ${CTO_FROM}

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV LLVM_CONFIG="/usr/bin/llvm-config-9"
ARG MAKEFLAGS=-j$(nproc) 

# Confirm TensorFlow python is working
RUN python3 -c "import tensorflow"

# Install system packages
RUN apt-get update -y \
  && apt-get install -y --no-install-recommends apt-utils \
  && apt-get install -y \ 
    autoconf \
    automake \
    checkinstall \
    cmake \
    curl \
    doxygen \
    file \
    gfortran \
    git \
    hdf5-tools \
    imagemagick \
    libatk-adaptor \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libblas-dev \
    libblosc-dev \
    libboost-all-dev \
    libbrotli-dev \
    libbz2-dev \
    libcanberra-gtk-module \
    libcharls-dev \
    libdc1394-22-dev \
    libeigen3-dev \
    libfaac-dev \
    libffi-dev \
    libfreetype6-dev \
    libgflags-dev \
    libgif-dev \
    libglew-dev \
    libgoogle-glog-dev \
    libgphoto2-dev \
    libgstreamer-plugins-bad1.0-0 \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev \
    libgtk-3-dev \
    libgtk2.0-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libjpeg8-dev \
    libjpeg-turbo8-dev \
    libjxr-dev \
    liblapack-dev \
    liblcms2-dev \
    liblz4-dev \
    liblzma-dev \
    libmp3lame-dev \
    libomp-dev \
    libopenblas-base \
    libopenblas-dev \
    libopencore-amrnb-dev \
    libopencore-amrwb-dev \
    libopenjp2-7-dev \
    libopenjp2-tools \
    libopenmpi2 \
    libopenmpi-dev \
    libpng-dev \
    libpostproc-dev \
    libprotobuf-dev \
    libprotoc-dev \
    libpython3-dev \
    libsnappy-dev \
    libsndfile1 \
    libsox-dev \
    libsox-fmt-all \ 
    libswscale-dev \
    libtbb-dev \
    libtbb2 \
    libtheora-dev \
    libtool \
    libv4l-dev \
    libvorbis-dev \
    libwebp-dev \
    libx264-dev \
    libxi-dev \
    libxine2-dev \
    libxmu-dev \
    libxvidcore-dev \
    libzmq3-dev \
    libzopfli-dev \
    libzstd-dev \
    llvm-9 \
		llvm-9-dev \
    openmpi-bin \
    openmpi-common \
    pkg-config \
    protobuf-compiler \
    python3-dev \
    python-imaging-tk \
    python3-lxml \
    python3-matplotlib \
    python3-pil \
    python3-tk \
    software-properties-common \
    time \
    unzip \
    v4l-utils \
    x11-apps \
    x264 \
    yasm \
    zip \
    zlib1g-dev \
  && apt-get clean

# More recent cmake
RUN apt-get update -y  \
  && apt-get install -y build-essential libssl-dev gpg wget \
  && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null \
  && echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ bionic main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null\
  && apt-get update \
  && apt-get install -y --only-upgrade cmake

# Set locale to UTF-8; seems to be ANSI_X3.4-1968 by default (to allow scikit-image to compile)
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
    && dpkg-reconfigure locales \
    && update-locale LANG=en_US.UTF-8
ENV LANG en_US.UTF-8

# CFLAGS (for imagecodecs)
ENV CFLAGS="-I/usr/include/openjpeg-2.3 -I/usr/include/jxrlib"

# Building libtiff (to allow imagecodecs to compile)
RUN mkdir -p /usr/local/src/libtiff \
  && wget -q --no-check-certificate -c https://gitlab.com/libtiff/libtiff/-/archive/v4.3.0/libtiff-v4.3.0.tar.bz2 -O - | tar --strip-components=1 -xj -C /usr/local/src/libtiff \
  && cd /usr/local/src/libtiff \
  && ./autogen.sh \
  && ./configure \
  && make install \
  && rm -rf /usr/local/src/libtiff

# brunsli (for imagecodecs)
RUN cd /tmp \
  && git clone --depth=1 https://github.com/google/brunsli.git \
  && cd brunsli \
  && git submodule update --init --recursive \
  && cmake -DCMAKE_BUILD_TYPE=Release \
  && make install

# Setup pip
RUN pip3 install --no-cache-dir -U pip
RUN pip3 install --no-cache-dir -U testresources setuptools

# Install Python tools (split install for Jetson)
RUN pip3 install --no-cache-dir -U Cython
RUN pip3 install --no-cache-dir --ignore-installed pybind11 
RUN pip3 install --no-cache-dir onnx
RUN pip3 install --no-cache-dir scipy
RUN pip3 install --no-cache-dir pycuda six
RUN pip3 install --no-cache-dir pillow
RUN pip3 install --no-cache-dir numba==0.47
RUN pip3 install --no-cache-dir matplotlib
RUN pip3 install --no-cache-dir PyWavelets
RUN pip3 install --no-cache-dir kiwisolver
RUN pip3 install --no-cache-dir imagecodecs
RUN pip3 install --no-cache-dir scikit-image
RUN pip3 install --no-cache-dir scikit-learn
RUN pip3 install --no-cache-dir ipython
RUN pip3 install --no-cache-dir notebook
RUN pip3 install --no-cache-dir pandas
RUN pip3 install --no-cache-dir moviepy
RUN pip3 install --no-cache-dir autovizwidget
RUN pip3 install --no-cache-dir jupyter
RUN pip3 install --no-cache-dir scikit-build
RUN pip3 install --no-cache-dir ninja
RUN pip3 install --no-cache-dir pysoundfile

# CuPy
ARG CUPY_VERSION=v9.2.0
ARG CUPY_NVCC_GENERATE_CODE="arch=compute_53,code=sm_53;arch=compute_62,code=sm_62;arch=compute_72,code=sm_72"
RUN git clone -b ${CUPY_VERSION} --recursive https://github.com/cupy/cupy cupy && \
    cd cupy && \
    pip3 install --no-cache-dir fastrlock && \
    python3 setup.py install --verbose && \
    cd ../ && \
    rm -rf cupy

# Installing PyTorch
# https://github.com/dusty-nv/jetson-containers/blob/master/scripts/docker_build_ml.sh
#RUN cd /tmp \
#  && wget -q --no-check-certificate https://developer.download.nvidia.com/compute/redist/jp/v461/pytorch/torch-1.11.0a0+17540c5-cp36-cp36m-linux_aarch64.whl -O torch-1.11.0a0+17540c5-cp36-cp36m-linux_aarch64.whl \
#  && pip3 install torch-1.11.0a0+17540c5-cp36-cp36m-linux_aarch64.whl \
#  && rm -rf /root/.cache/pip torch-1.11.0a0+17540c5-cp36-cp36m-linux_aarch64.whl
# https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048
RUN cd /tmp \
  && wget -q --no-check-certificate https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl \
  && pip3 install --no-cache-dir torch-1.10.0-cp36-cp36m-linux_aarch64.whl
RUN python3 -c "import torch"

# Download OpenCV
ARG CTO_OPENCV_VERSION
ARG CTO_NUMPROC=1
ARG CTO_CUDA_BUILD
RUN if [ -f /usr/include/aarch64-linux-gnu/cudnn_v8.h ]; then ln -s /usr/include/aarch64-linux-gnu/cudnn_v8.h /usr/include/aarch64-linux-gnu/cudnn.h; fi
RUN mkdir -p /usr/local/src/opencv /usr/local/src/opencv_contrib \
  && cd /usr/local/src \
  && wget -q --no-check-certificate https://github.com/opencv/opencv/archive/${CTO_OPENCV_VERSION}.tar.gz -O - | tar --strip-components=1 -xz -C /usr/local/src/opencv \
  && wget -q --no-check-certificate https://github.com/opencv/opencv_contrib/archive/${CTO_OPENCV_VERSION}.tar.gz -O - | tar --strip-components=1 -xz -C /usr/local/src/opencv_contrib \
  && mkdir -p /usr/local/src/opencv/build \
  && cd /usr/local/src/opencv/build \
  && time cmake \
    -DBUILD_DOCS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_opencv_java=OFF \
    -DBUILD_opencv_python2=OFF \
    -DBUILD_opencv_python3=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DCMAKE_INSTALL_TYPE=Release \
    -DFORCE_VTK=ON \
    -DINSTALL_C_EXAMPLES=OFF \
    -DINSTALL_PYTHON_EXAMPLES=OFF \
    -DOPENCV_ENABLE_NONFREE=OFF \
    -DOPENCV_EXTRA_MODULES_PATH=/usr/local/src/opencv_contrib/modules \
    -DOPENCV_GENERATE_PKGCONFIG=YES \
    -DWITH_CSTRIPES=ON \
    -DWITH_EIGEN=ON \
    -DWITH_GDAL=ON \
    -DWITH_GSTREAMER=ON \
    -DWITH_GSTREAMER_0_10=OFF \
    -DWITH_GTK=ON \
    -DWITH_IPP=OFF \
    -DWITH_OPENCL=OFF \
    -DWITH_OPENGL=ON \
    -DWITH_OPENMP=ON \
    -DWITH_TBB=ON \
	  -DBUILD_TIFF=ON \
    -DWITH_V4L=ON \
    -DWITH_LIBV4L=ON \
    -DWITH_WEBP=ON \
    -DWITH_XINE=ON \
    -DCUDNN_INCLUDE_DIR=/usr/include/aarch64-linux-gnu \
    -DEIGEN_INCLUDE_PATH=/usr/include/eigen3 \
    -DOPENCV_GENERATE_PKGCONFIG=ON \
    -DOPENCV_PC_FILE_NAME=opencv.pc \
    -DOPENCV_PYTHON3_INSTALL_PATH=/usr/lib/python3/dist-packages \
    ${CTO_CUDA_BUILD} \
    .. \
  && make -j${CTO_NUMPROC} install \
  && sh -c 'echo "/usr/local/lib" >> /etc/ld.so.conf.d/opencv.conf' \
  && ldconfig \
  && rm -rf /usr/local/src/opencv /usr/local/src/opencv_contrib
## FYI: We are removing the OpenCV directory (in /usr/local/src) to attempt to save additional disk space
# Comment the above line (and remove the \ in the line above) if you want to
#  rerun cmake with additional/modified options AFTER it was built; for example:
# cd /usr/local/src/opencv/build
# cmake -DOPENCV_ENABLE_NONFREE=ON -DBUILD_EXAMPLES=ON -DBUILD_DOCS=ON -DBUILD_TESTS=ON -DBUILD_PERF_TESTS=ON .. && make install

# Confirm OpenCV python is built
RUN python3 -c "import cv2; print(cv2.__version__);"

# Skipping torchaudio and torchvision in this build
# https://github.com/dusty-nv/jetson-containers/blob/master/Dockerfile.pytorch
#ARG TORCH_CUDA_ARCH_LIST="5.3;6.2;7.2"

#ARG TORCHVISION_VERSION="0.11.3"
#RUN pip3 install --no-cache-dir torchvision==${TORCHVISION_VERSION}
#RUN git clone -b v${TORCHVISION_VERSION} https://github.com/pytorch/vision torchvision && \
#    cd torchvision && \
#    CC=/usr/bin/clang CXX=/usr/bin/clang++ python3 setup.py install && \
#    cd ../ && \
#    rm -rf torchvision && rm -rf /root/.cache/pip

#ARG TORCHAUDIO_VERSION="0.10.2"
#RUN pip3 install --no-cache-dir torchaudio==${TORCHAUDIO_VERSION}
#RUN git clone --recursive -b v${TORCHAUDIO_VERSION} https://github.com/pytorch/audio torchaudio && \
#    cd torchaudio && \
#    BUILD_SOX=1 python3 setup.py install && \
#    cd ../ && \
#    rm -rf torchaudio && rm -rf /root/.cache/pip

# Installing a built-TF compatible keras
ARG CTO_TF_KERAS
RUN pip3 install ${CTO_TF_KERAS} \
  && rm -rf /root/.cache/pip

# Add dataframe display widget
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension

RUN touch /.within_container

# https://github.com/opencv/opencv/issues/14884
ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

# Setting up working directory 
RUN mkdir /dmc
WORKDIR /dmc

CMD /bin/bash

LABEL "Author"="Data Machines Corp <help@datamachines.io>"
