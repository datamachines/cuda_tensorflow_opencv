ARG CTO_FROM
FROM ${CTO_FROM}

# Install system packages
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -y \
  && apt-get install -y --no-install-recommends apt-utils \
  && apt-get install -y \ 
    autoconf \
    automake \
    build-essential \
    bzip2 \
    checkinstall \
    cmake \
    curl \
    doxygen \
    file \
    g++-6 \
    gcc-6 \
    gfortran \
    git \
    gstreamer1.0-plugins-good \
    hdf5-tools \
    imagemagick \
    libaec-dev \
    libatk-adaptor \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libblosc-dev \
    libboost-all-dev \
    libbrotli-dev \
    libbz2-dev \
    libcanberra-gtk-module \
    libcharls-dev \
    libdc1394-22-dev \
    libeigen3-dev \
    libfaac-dev \
    libfreetype6-dev \
    libgflags-dev \
    libgif-dev \
    libglew-dev \
    libglu1-mesa \
    libglu1-mesa-dev \
    libgoogle-glog-dev \
    libgphoto2-dev \
    libgstreamer-plugins-bad1.0-0 \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev \
    libgtk-3-dev \
    libgtk2.0-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libjpeg-dev \
    libjpeg-turbo8-dev \
    libjxr-dev \
    liblapack-dev \
    liblcms2-dev \
    liblz4-dev \
    liblzma-dev \
    libmp3lame-dev \
    libopenblas-dev \
    libopencore-amrnb-dev \
    libopencore-amrwb-dev \
    libopenjp2-7-dev \
    libopenjp2-tools \
    libpng-dev \
    libpostproc-dev \
    libprotobuf-dev \
    libpython3-dev \
    libsnappy-dev \
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
    locales \
    pkg-config \
    protobuf-compiler \
    python-imaging-tk \
    python-lxml \
    python-pil \
    python3-dev \
    python3-tk \
    software-properties-common \
    unzip \
    v4l-utils \
    wget \
    x11-apps \
    x264 \
    yasm \
    zip \
    zlib1g-dev

# Set locale to UTF-8; seems to be ANSI_X3.4-1968 by default (to allow scikit-image to compile)
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
    && dpkg-reconfigure locales \
    && update-locale LANG=en_US.UTF-8
ENV LANG en_US.UTF-8

# CFLAGS (for imagecodecs)
ENV CFLAGS="-I/usr/include/openjpeg-2.3 -I/usr/include/jxrlib"

# Building libtiff (to allow imagecodecs to compile)
RUN cd /tmp \
  && wget https://gitlab.com/libtiff/libtiff/-/archive/v4.1.0/libtiff-v4.1.0.tar.bz2 \
  && tar xvfj libtiff-v4.1.0.tar.bz2 \
  && cd libtiff-v4.1.0 \
  && ./autogen.sh \
  && ./configure \
  && make install

# brunsli (for imagecodecs)
RUN cd /tmp \
  && git clone --depth=1 https://github.com/google/brunsli.git \
  && cd brunsli \
  && git submodule update --init --recursive \
  && cmake -DCMAKE_BUILD_TYPE=Release \
  && make install

# Setup pip
RUN wget -q -O /tmp/get-pip.py --no-check-certificate https://bootstrap.pypa.io/get-pip.py \
  && python3 /tmp/get-pip.py \
  && pip3 install -U pip
RUN pip3 install -U testresources setuptools

## Nano refactorization: build takes a lot longer than download, so build common components first, including OpenCV

# Install Python tools (split install for Nano)
RUN pip3 install -U Cython
RUN pip3 install -U pillow
RUN pip3 install -U numpy
RUN pip3 install -U matplotlib
RUN pip3 install -U PyWavelets
RUN pip3 install -U kiwisolver
RUN pip3 install -U imagecodecs
RUN pip3 install -U scikit-image
RUN pip3 install -U scikit-learn
RUN pip3 install -U ipython
RUN pip3 install -U notebook
RUN pip3 install -U pandas
RUN pip3 install -U moviepy
RUN pip3 install -U keras
RUN pip3 install -U autovizwidget
RUN pip3 install -U jupyter

# Additional specialized apt installs
ARG CTO_CUDA_APT
RUN apt-get install -y --no-install-recommends \
      vim ${CTO_CUDA_APT}
      
# Download OpenCV
ARG CTO_OPENCV_VERSION
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

# Build OpenCV
ARG CTO_NUMPROC=1
ARG CTO_CUDA_BUILD
RUN mkdir -p /usr/local/src/opencv/build \
  && cd /usr/local/src/opencv/build \
  && cmake \
    -DBUILD_DOCS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_TESTS=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local/ \
    -DCMAKE_INSTALL_TYPE=Release \
    -DENABLE_FAST_MATH=1 \
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
    -DWITH_OPENCL=ON \
    -DWITH_OPENMP=ON \
    -DWITH_TBB=ON \
    -DWITH_V4L=ON \
    -DWITH_WEBP=ON \
    -DWITH_XINE=ON \
    ${CTO_CUDA_BUILD} \
    .. \
  && make -j${CTO_NUMPROC} install \
  && sh -c 'echo "/usr/local/lib" >> /etc/ld.so.conf.d/opencv.conf' \
  && ldconfig \
  && rm -rf /usr/local/src/opencv/build
## FYI: We are removing the OpenCV build directory (in /usr/local/src/opencv) 
#   to attempt to save additional disk space
# Comment the above line (and remove the \ in the line above) if you want to
#  rerun cmake with additional/modified options AFTER it was built; for example:
# cd /usr/local/src/opencv/build
# cmake -DOPENCV_ENABLE_NONFREE=ON -DBUILD_EXAMPLES=ON -DBUILD_DOCS=ON -DBUILD_TESTS=ON -DBUILD_PERF_TESTS=ON .. && make install

# Install NVidia's monthly version of Tensorflow
ARG CTO_TENSORFLOW_PYTHON
RUN pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v43 ${CTO_TENSORFLOW_PYTHON}

# Add dataframe display widget
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension

# Setting up working directory 
RUN mkdir /dmc
WORKDIR /dmc

CMD /bin/bash

LABEL "Author"="Data Machines Corp <help@datamachines.io>"
