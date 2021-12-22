ARG CTO_FROM
FROM ${CTO_FROM}

##### Splitting installation into controlled parts:
## - base container setup
## - Tensorflow build and install
## - OpenCV build and install
## - additional setup

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

##### TensorFlow

# Install system packages
RUN apt-get install -y \
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
  && apt-get clean

# Setup pip
RUN wget -q -O /tmp/get-pip.py --no-check-certificate https://bootstrap.pypa.io/get-pip.py \
  && python3 /tmp/get-pip.py \
  && pip3 install -U pip \
  && rm /tmp/get-pip.py

# Some TF tools expect a "python" binary
RUN ln -s $(which python3) /usr/local/bin/python

# Additional specialized apt installs
ARG CTO_CUDA_APT
ARG CTO_CUDA11_APT_XTRA
RUN apt-get install -y --no-install-recommends ${CTO_CUDA11_APT_XTRA} \
      time ${CTO_CUDA_APT} \
  && apt-get clean

# CUPTI library needed by TensorFlow1 but sometimes not in default path, adding if at unconventional location
COPY tools/cupti_helper.sh /tmp/
RUN chmod +x /tmp/cupti_helper.sh \
  && /tmp/cupti_helper.sh \
  && rm /tmp/cupti_helper.sh

# Install TF needed Python tools (for buiding) 
ARG CTO_TF_NUMPY=numpy
RUN pip3 install -U \
  mock \
  ${CTO_TF_NUMPY} \
  setuptools \
  six \
  wheel \
  future \
  && pip3 install -U keras_applications --no-deps \
  && pip3 install -U keras_preprocessing --no-deps \
  && rm -rf /root/.cache/pip

## Download & Building TensorFlow from source in same RUN
ARG LATEST_BAZELISK=1.5.0
ARG LATEST_BAZEL=3.4.1
ARG CTO_TENSORFLOW_VERSION
ARG CTO_TF_CUDNN="no"
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
  && time /tmp/tf_build.sh ${CTO_TF_CUDNN} ${CTO_TF_OPT} \
  && time ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg \
  && time pip3 install /tmp/tensorflow_pkg/tensorflow-*.whl \
  && rm -rf /usr/local/src/tensorflow /tmp/tensorflow_pkg /tmp/bazel_check.pl /tmp/tf_build.sh /tmp/hsperfdata_root /root/.cache/bazel /root/.cache/pip /root/.cache/bazelisk

# Confirm TensorFlow python is built
RUN python3 -c "import tensorflow"

##### OpenCV

# Install system packages
RUN apt-get install -y \
    doxygen \
    file \
    gfortran \
    gnupg \
    gstreamer1.0-plugins-good \
    imagemagick \
    libatk-adaptor \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
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
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libtheora-dev \
    libtiff5-dev \
    libv4l-dev \
    libvorbis-dev \
    libx264-dev \
    libxi-dev \
    libxine2-dev \
    libxmu-dev \
    libxvidcore-dev \
    libzmq3-dev \
    v4l-utils \
    x11-apps \
    x264 \
    yasm \
  && apt-get clean

# Python
RUN pip3 install -U \
  Pillow \
  lxml \
  && rm -rf /root/.cache/pip

# Download & Build OpenCV in same RUN
ARG CTO_OPENCV_VERSION
ARG CTO_NUMPROC=1
ARG CTO_CUDA_BUILD
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
    .. \
  && time make -j${CTO_NUMPROC} install \
  && sh -c 'echo "/usr/local/lib" >> /etc/ld.so.conf.d/opencv.conf' \
  && ldconfig \
  && rm -rf /usr/local/src/opencv /usr/local/src/opencv_contrib
## FYI: We are removing the OpenCV source and build directory in /usr/local/src to attempt to save additional disk space
# Comment the above line (and remove the \ in the line above) if you want to rerun cmake with additional/modified options. For example:
# cd /usr/local/src/opencv/build
# cmake -DOPENCV_ENABLE_NONFREE=ON -DBUILD_EXAMPLES=ON -DBUILD_DOCS=ON -DBUILD_TESTS=ON -DBUILD_PERF_TESTS=ON .. && make install

# Confirm OpenCV python is built
RUN python3 -c "import cv2"

##### Additional tools

# Install additional Python tools 
RUN pip3 install -U \
  autovizwidget \
  ipython \
  jupyter \
  matplotlib \
  moviepy \
  notebook \
  pandas \
  scikit-image \
  scikit-learn \
  && rm -rf /root/.cache/pip
# Removed scipy: the version required by TF2 is already installed during the TF2 step (and it will be installed by Keras for TF1)

# Installing a built-TF compatible keras (for TF1) and PyTorch
ARG CTO_TF_KERAS
ARG CTO_PYTORCH
RUN pip3 install ${CTO_TF_KERAS} ${CTO_PYTORCH} \
  && rm -rf /root/.cache/pip

# Add dataframe display widget
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension

# Tool to dump some installation details
COPY tools/tf_info.sh /tmp/
RUN chmod +x /tmp/tf_info.sh \
  && touch /.within_container

# Setting up working directory 
RUN mkdir /dmc
WORKDIR /dmc

CMD bash

LABEL "Author"="Data Machines Corp <help@datamachines.io>"
