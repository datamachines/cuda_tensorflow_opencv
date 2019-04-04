FROM tensorflow/tensorflow:1.13.1-gpu-py3-jupyter
# tensorflow-gpu requires nvidia-docker v2 to run
# and is based of nvidia's CUDA 10.0 Docker image running on Ubuntu 16.04 
#
# Build using: docker build -t cuda_tensorflow_opencv:10.0_1.13.1_4.0.1-0.2 .

ENV PROTOBUF_VERSION 3.5.1
ENV OPENCV_VERSION 4.0.1

LABEL "Author"="Data Machines Corp <help@datamachines.io>"
LABEL description="Preconfigured Ubuntu 16.04 with Nvidia CUDA enabled version of TensorFlow and OpenCV"
LABEL version="0.2"

# Install system packages
RUN apt-get -qq update \
  && apt-get -qq install --no-install-recommends -y \ 
    wget \
    unzip \
    build-essential \
    cmake \ 
    git \ 
    pkg-config \
    libatlas-base-dev \
    libgtk2.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \ 
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libv4l-dev \
    libdc1394-22-dev \
    qt4-default \
    libatk-adaptor \
    libcanberra-gtk-module \
    imagemagick \
    python3-dev \
    python-pil \
    python-lxml \
    python-tk \
    libgtk2.0-dev \ 
    libjasper-dev \
    x11-apps \
    vim \
    ffmpeg \
    libfreetype6-dev \
    libhdf5-serial-dev \
    libpng12-dev \
    libzmq3-dev \
    software-properties-common \
    libgtk-3-dev \
    libboost-all-dev \
    cuda-npp-dev-10.0 \
    cuda-cublas-dev-10.0 \
    cuda-cufft-dev-10.0 \
    libxvidcore-dev \
    libx264-dev 

# Download & build protobuf-python
RUN wget -q -P /usr/local/src/ --no-check-certificate https://github.com/google/protobuf/releases/download/v3.5.1/protobuf-python-${PROTOBUF_VERSION}.tar.gz
RUN cd /usr/local/src/ \
  && tar xf protobuf-python-${PROTOBUF_VERSION}.tar.gz \
  && rm protobuf-python-${PROTOBUF_VERSION}.tar.gz \
  && cd /usr/local/src/protobuf-${PROTOBUF_VERSION}/ \
  && ./configure \
  && export NUMPROC=$(nproc --all) \
  && make -j$NUMPROC VERBOSE=1 install \
  && ldconfig \
  && rm -rf /usr/local/src/protobuf-${PROTOBUF_VERSION}/

# Add dataframe display widget
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension

# Download & build OpenCV
RUN mkdir -p /usr/local/src \
  && cd /usr/local/src \
  && git clone https://github.com/opencv/opencv.git \
  && cd opencv \
  && git checkout $OPENCV_VERSION \
  && cd .. \
  && git clone https://github.com/opencv/opencv_contrib \
  && cd opencv_contrib \
  && git checkout $OPENCV_VERSION
RUN mkdir -p /usr/local/src/opencv/build \
  && cd /usr/local/src/opencv/build \
  && cmake \
    -D CMAKE_INSTALL_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=/usr/local/ \
    -D INSTALL_C_EXAMPLES=OFF \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D BUILD_EXAMPLES=OFF \
    -D OPENCV_EXTRA_MODULES_PATH=/usr/local/src/opencv_contrib/modules \
    -D BUILD_DOCS=OFF \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D WITH_TBB=ON \
    -D WITH_OPENMP=ON \
    -D WITH_IPP=ON \
    -D WITH_CSTRIPES=ON \
    -D WITH_OPENCL=ON \
    -D WITH_V4L=ON \
    -D WITH_CUDA=ON \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D WITH_CUBLAS=1 \
    -D FORCE_VTK=ON \
    -D WITH_GDAL=ON \
    -D WITH_XINE=ON \
    -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.0 \
    -D CUDA_cublas_LIBRARY=cublas \
    -D CUDA_cufft_LIBRARY=cufft \
    -D CUDA_nppim_LIBRARY=nppim \
    -D CUDA_nppidei_LIBRARY=nppidei \
    -D CUDA_nppif_LIBRARY=nppif \
    -D CUDA_nppig_LIBRARY=nppig \
    -D CUDA_nppim_LIBRARY=nppim \
    -D CUDA_nppist_LIBRARY=nppist \
    -D CUDA_nppisu_LIBRARY=nppisu \
    -D CUDA_nppitc_LIBRARY=nppitc \
    -D CUDA_npps_LIBRARY=npps \
    -D CUDA_nppc_LIBRARY=nppc \
    -D CUDA_nppial_LIBRARY=nppial \
    -D CUDA_nppicc_LIBRARY=nppicc \
    -D CUDA_nppicom_LIBRARY=nppicom \
    -DWITH_WEBP=OFF \
    .. \
  && export NUMPROC=$(nproc --all) \
  && make -j$NUMPROC VERBOSE=1 install \
  && rm -rf /usr/local/src/opencv
 
# Install core python packages 
RUN wget -q -O /tmp/get-pip.py --no-check-certificate https://bootstrap.pypa.io/get-pip.py \
  && python3 /tmp/get-pip.py
RUN pip install -U pip \
  numpy \
  matplotlib \
  notebook \
  pandas \
  moviepy \
  keras \
  autovizwidget

# Minimize image size 
RUN (apt-get autoremove -y; apt-get autoclean -y)

# Setting up working directory 
RUN mkdir /dmc
WORKDIR /dmc

CMD bash