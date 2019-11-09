ARG CTO_TENSORFLOW_TAG
FROM tensorflow/tensorflow:$CTO_TENSORFLOW_TAG
#
# Note: tensorflow-gpu requires nvidia-docker v2 to run
# and is based of nvidia's CUDA Docker image running on Ubuntu 18.04 
##
# Recommended build: follow the options offered by the Makefile
#
# using: CTO_TAG=<valid tag, use docker images to list available tags> ./runDocker.sh

ARG CTO_OPENCV_VERSION
ARG CTO_NUMPROC=1
ARG CTO_CUDA_APT
ARG CTO_CUDA_BUILD

# Install system packages
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update \
  && apt-get install -y --no-install-recommends apt-utils \
  && apt-get install -y --no-install-recommends \ 
    wget unzip vim file \
    build-essential cmake git pkg-config software-properties-common \
    libatlas-base-dev libboost-all-dev \
    x11-apps libgtk2.0-dev libgtk2.0-dev libcanberra-gtk-module libgtk-3-dev qt4-default \
    libtbb2 libtbb-dev \ 
    libjpeg-dev libpng-dev libtiff-dev libpng-dev imagemagick \
    libv4l-dev libdc1394-22-dev libatk-adaptor \
    python3-dev libpython3-dev python-pil python-lxml python-tk \
    libfreetype6-dev libhdf5-serial-dev libzmq3-dev \
    libavcodec-dev libavformat-dev libswscale-dev libxvidcore-dev \
    libx264-dev ffmpeg ${CTO_CUDA_APT}

# For OpenCV 3 compilation
RUN add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main" \
  && apt-get update \
  && apt-get install -y libjasper1 libjasper-dev

# Install core python packages 
RUN wget -q -O /tmp/get-pip.py --no-check-certificate https://bootstrap.pypa.io/get-pip.py \
  && python3 /tmp/get-pip.py \
  && pip install -U pip \
  && pip install -U numpy matplotlib notebook pandas \
    moviepy keras autovizwidget jupyter

# Download & build OpenCV
RUN mkdir -p /usr/local/src \
  && cd /usr/local/src \
  && wget -q --no-check-certificate https://github.com/opencv/opencv/archive/${CTO_OPENCV_VERSION}.tar.gz \
  && tar xfz ${CTO_OPENCV_VERSION}.tar.gz \
  && mv opencv-${CTO_OPENCV_VERSION} opencv \
  && rm ${CTO_OPENCV_VERSION}.tar.gz \
  && wget -q --no-check-certificate https://github.com/opencv/opencv_contrib/archive/${CTO_OPENCV_VERSION}.tar.gz \
  && tar xfz ${CTO_OPENCV_VERSION}.tar.gz \
  && mv opencv_contrib-${CTO_OPENCV_VERSION} opencv_contrib \
  && rm ${CTO_OPENCV_VERSION}.tar.gz
RUN mkdir -p /usr/local/src/opencv/build \
  && cd /usr/local/src/opencv/build \
  && cmake \
    -DCMAKE_INSTALL_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local/ \
    -DOPENCV_GENERATE_PKGCONFIG=ON \
    -DWITH_WEBP=OFF \
    -DINSTALL_C_EXAMPLES=OFF -DINSTALL_PYTHON_EXAMPLES=OFF -DBUILD_EXAMPLES=OFF \
    -DOPENCV_EXTRA_MODULES_PATH=/usr/local/src/opencv_contrib/modules \
    -DBUILD_DOCS=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF \
    -DWITH_TBB=ON -D WITH_EIGEN=ON -D WITH_OPENMP=ON \
    -DWITH_IPP=ON -DWITH_CSTRIPES=ON -DWITH_OPENCL=ON \
    -DWITH_V4L=ON -DENABLE_FAST_MATH=1 -DFORCE_VTK=ON \
    -DWITH_GDAL=ON -DWITH_XINE=ON -DWITH_GTK=ON \
    -DWITH_OPENMP=ON ${CTO_CUDA_BUILD} \
    .. \
  && make -j${CTO_NUMPROC} install \
  && rm -rf /usr/local/src/opencv

# Add dataframe display widget
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension

# Minimize image size 
RUN (apt-get autoremove -y; apt-get autoclean -y)

# Setting up working directory 
RUN mkdir /dmc
WORKDIR /dmc

CMD bash

# Labels -- kept at the end to maximize layer reuse when possible
LABEL "Author"="Data Machines Corp <help@datamachines.io>"
