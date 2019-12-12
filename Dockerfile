ARG CTO_FROM
FROM ${CTO_FROM}

# Install system packages
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update \
  && apt-get install -y --no-install-recommends apt-utils \
  && apt-get install -y --no-install-recommends \ 
    wget unzip file \
    build-essential cmake git pkg-config software-properties-common \
    libatlas-base-dev libboost-all-dev \
    x11-apps libgtk2.0-dev libgtk2.0-dev libcanberra-gtk-module libgtk-3-dev qt4-default \
    libtbb2 libtbb-dev \ 
    libjpeg-dev libpng-dev libtiff-dev imagemagick \
    libv4l-dev libdc1394-22-dev libatk-adaptor \
    python3-dev libpython3-dev python-pil python-lxml python-tk \
    libfreetype6-dev libhdf5-serial-dev libzmq3-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev \
    gcc-6 g++-6 libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev \
    libopenblas-dev libatlas-base-dev liblapack-dev gfortran \
    libhdf5-serial-dev python3-tk python-imaging-tk libgtk-3-dev

# Setup pip
# Install core python packages 
RUN wget -q -O /tmp/get-pip.py --no-check-certificate https://bootstrap.pypa.io/get-pip.py \
  && python3 /tmp/get-pip.py \
  && pip install -U pip

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

# Install Python tools and TensorFlow
ARG CTO_TENSORFLOW_PYTHON
RUN pip install -U numpy matplotlib notebook pandas moviepy keras autovizwidget jupyter \
    && pip install ${CTO_TENSORFLOW_PYTHON}

# Build OpenCV
ARG CTO_NUMPROC=1
ARG CTO_CUDA_BUILD
RUN mkdir -p /usr/local/src/opencv/build \
  && cd /usr/local/src/opencv/build \
  && cmake \
    -DCMAKE_INSTALL_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local/ \
    -DOPENCV_GENERATE_PKGCONFIG=ON \
    -DWITH_WEBP=OFF \
    -DINSTALL_C_EXAMPLES=OFF -DINSTALL_PYTHON_EXAMPLES=OFF -DBUILD_EXAMPLES=OFF \
    -DOPENCV_EXTRA_MODULES_PATH=/usr/local/src/opencv_contrib/modules \
    -DBUILD_DOCS=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF \
    -DWITH_TBB=ON -D WITH_EIGEN=ON \
    -DWITH_IPP=ON -DWITH_CSTRIPES=ON -DWITH_OPENCL=ON \
    -DWITH_V4L=ON -DENABLE_FAST_MATH=1 -DFORCE_VTK=ON \
    -DWITH_GDAL=ON -DWITH_XINE=ON -DWITH_GTK=ON \
    -DWITH_OPENMP=ON ${CTO_CUDA_BUILD} \
    .. \
  && make -j${CTO_NUMPROC} install

# Add dataframe display widget
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension

# Setting up working directory 
RUN mkdir /dmc
WORKDIR /dmc

CMD bash

LABEL "Author"="Data Machines Corp <help@datamachines.io>"

# Attempt to Minimize image size 
RUN (apt-get autoremove -y; apt-get autoclean -y)

## FYI: We are removing the OpenCV build directory (in /usr/local/src/opencv) to attempt to save additional disk space
RUN rm -rf /usr/local/src/opencv/build
# Comment the above line if you want to rerun cmake with additional/modified options; for example:
#  cd /usr/local/src/opencv/build
#  cmake -DBUILD_EXAMPLES=ON -DBUILD_DOCS=ON -DBUILD_TESTS=ON -DBUILD_PERF_TESTS=ON .
#  make
