ARG CTO_FROM
FROM ${CTO_FROM}

# Install system packages
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -y \
  && apt-get install -y --no-install-recommends apt-utils \
  && apt-get install -y \ 
    wget curl unzip file \
    build-essential cmake git pkg-config software-properties-common \
    checkinstall yasm \
    libatlas-base-dev libboost-all-dev \
    x11-apps libgtk2.0-dev libgtk2.0-dev libcanberra-gtk-module libgtk-3-dev \
    libtbb2 libtbb-dev \ 
    libjpeg-dev libpng-dev libtiff5-dev libopenjp2-7-dev libopenjp2-tools imagemagick \
    v4l-utils libv4l-dev libdc1394-22-dev libatk-adaptor \
    python3-dev libpython3-dev python-pil python-lxml python-tk \
    libfreetype6-dev libhdf5-serial-dev libzmq3-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libxvidcore-dev libxine2-dev x264 libx264-dev libavutil-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-good libgstreamer-plugins-bad1.0-0\
    libfaac-dev libmp3lame-dev libtheora-dev libvorbis-dev \
    libopencore-amrnb-dev libopencore-amrwb-dev \
    libprotobuf-dev protobuf-compiler \
    libgoogle-glog-dev libgflags-dev \ 
    libgphoto2-dev libeigen3-dev libhdf5-dev doxygen \
    gcc-6 g++-6 libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev \
    libopenblas-dev liblapack-dev gfortran \
    python3-tk python-imaging-tk libgtk-3-dev \
    libglew-dev libpostproc-dev zlib1g-dev

# Setup pip
RUN wget -q -O /tmp/get-pip.py --no-check-certificate https://bootstrap.pypa.io/get-pip.py \
  && python3 /tmp/get-pip.py \
  && pip3 install -U pip

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

# Install Python tools 
RUN pip3 install -U numpy scipy matplotlib scikit-image scikit-learn ipython notebook pandas moviepy keras autovizwidget jupyter

# Build OpenCV
ARG CTO_NUMPROC=1
ARG CTO_CUDA_BUILD
RUN mkdir -p /usr/local/src/opencv/build \
  && cd /usr/local/src/opencv/build \
  && cmake \
    -DOPENCV_ENABLE_NONFREE=OFF \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local/ \
    -DOPENCV_GENERATE_PKGCONFIG=YES \
    -DWITH_WEBP=ON \
    -DINSTALL_C_EXAMPLES=OFF -DINSTALL_PYTHON_EXAMPLES=OFF -DBUILD_EXAMPLES=OFF \
    -DOPENCV_EXTRA_MODULES_PATH=/usr/local/src/opencv_contrib/modules \
    -DBUILD_DOCS=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF \
    -DWITH_TBB=ON -DWITH_EIGEN=ON \
    -DBUILD_opencv_python3=ON -DBUILD_opencv_python2=OFF\
    -DWITH_IPP=ON -DWITH_CSTRIPES=ON -DWITH_OPENCL=ON \
    -DWITH_V4L=ON -DENABLE_FAST_MATH=1 -DFORCE_VTK=ON \
    -DWITH_GDAL=ON -DWITH_XINE=ON -DWITH_GTK=ON \
    -DWITH_GSTREAMER=ON -DWITH_GSTREAMER_0_10=OFF \
    -DWITH_OPENMP=ON ${CTO_CUDA_BUILD} \
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

# TensorFlow GPU 's pip seems to only work with CUDA 10.0
RUN apt-get install -y rsync
RUN cd /usr/local && ln -s cuda nvidia

ARG CTO_TF_CUDA="None"
COPY tf_cuda_adds.sh /tmp/
RUN chmod +x /tmp/tf_cuda_adds.sh && /tmp/tf_cuda_adds.sh ${CTO_TF_CUDA}

# Install TensorFlow
ARG CTO_TENSORFLOW_PYTHON
RUN pip3 install ${CTO_TENSORFLOW_PYTHON}

# Add dataframe display widget
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension

# Setting up working directory 
RUN mkdir /dmc
WORKDIR /dmc

CMD bash

LABEL "Author"="Data Machines Corp <help@datamachines.io>"

# Attempt to Minimize image size 
RUN (apt-get autoremove -y; apt-get autoclean -y)
