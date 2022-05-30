# Needed SHELL since I'm using zsh
SHELL := /bin/bash
.PHONY: all build_all actual_build build_prep

# Release to match data of Dockerfile and follow YYYYMMDD pattern
CTO_RELEASE=20220530

# The default is not to build OpenCV non-free or build FFmpeg with libnpp, as those would make the images unredistributable 
# Replace "" by "unredistributable" if you need to use those for a personal build
CTO_ENABLE_NONFREE=""

# Maximize build speed
CTO_NUMPROC := $(shell nproc --all)

# docker build extra parameters
DOCKER_BUILD_ARGS=
#DOCKER_BUILD_ARGS="--no-cache"

# Use "yes" below before a multi build to have docker pull the base images using "make build_all" 
DOCKERPULL="no"

# Use "yes" below to force a TF check post build (recommended)
# this will use docker run [...] --gpus all and extend the TF build log
MLTK_CHECK="yes"

# Table below shows driver/CUDA support; for example the 10.2 container needs at least driver 440.33
# https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility__table-toolkit-driver
#
# According to https://hub.docker.com/r/nvidia/cuda/
# Looking at the tags, Ubuntu 18.04 is still the primary for 9.x and 10.x
# 
# CUDA 11 came out in May 2020
# Nivida released their CUDA11 containers only with Ubuntu 20.04 support
# https://hub.docker.com/r/nvidia/cuda/tags?page=1&name=20.04
# 10.2 release now is available with cuddn8
#
# Note: CUDA11 minimum version has to match the one used by PyTorch
STABLE_CUDA11p=11.3.1
STABLE_CUDA11l=11.6.2
# For CUDA11 it might be possible to upgrade some of the pre-installed libraries to their latest version, this will add significant space to the container
# to do, uncomment the line below the empty string set
CUDA11_APT_XTRA=""
#CUDA11_APT_XTRA="--allow-change-held-packages"

# CUDNN needs 5.3 at minimum, extending list from https://en.wikipedia.org/wiki/CUDA#GPUs_supported 
# Skipping Tegra, Jetson, ... (ie not desktop/server GPUs) from this list
# Keeping from Pascal and above
# Also only installing cudnn7 for 18.04 based systems
DNN_ARCH_CUDA11=6.0,6.1,7.0,7.5,8.0,8.6

# According to https://opencv.org/releases/
STABLE_OPENCV3=3.4.16
STABLE_OPENCV4=4.5.5

# FFmpeg
# Release list: https://ffmpeg.org/download.html
# Note: FFmpeg < 5 because https://github.com/pytorch/vision/issues/5928
# Note: GPU extensions are added directly in the Dockerfile
CTO_FFMPEG_VERSION=4.4.2
# https://github.com/FFmpeg/nv-codec-headers/releases
CTO_FFMPEG_NVCODEC="11.1.5.1"

# TF2 CUDA11 minimum is 2.4.0
##
# According to https://github.com/tensorflow/tensorflow/tags
STABLE_TF2=2.9.1

## Information for build
# https://github.com/bazelbuild/bazelisk
LATEST_BAZELISK=1.11.0
# https://github.com/bazelbuild/bazel (4+ is out but keeping with 3.x for TF < 2.8)
LATEST_BAZEL=5.1.1
# https://github.com/keras-team/keras/releases
TF2_KERAS="keras"

TF2_NUMPY='numpy'

# Magma
# Release page: https://icl.utk.edu/magma/software/index.html
# Note: GPU targets (ie ARCH) are directly added in Dockerfile
CTO_MAGMA="2.6.2"

## PyTorch (with FFmpeg + OpenCV & Magma if available)
# Note: same as FFmpeg and Magma, GPU specific selection (including ARCH) are in the Dockerfile
# Use release branch https://github.com/pytorch/pytorch
CTO_TORCH="1.11"
# Use release branch https://github.com/pytorch/vision
CTO_TORCHVISION="0.12"
# Use release branch https://github.com/pytorch/audio
CTO_TORCHAUDIO="0.11"

##########

##### CuDNN _ Tensorflow _ OpenCV (aka CTO)
CTO_BUILDALL =cudnn_tensorflow_opencv-${STABLE_CUDA11p}_${STABLE_TF2}_${STABLE_OPENCV3}
CTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA11p}_${STABLE_TF2}_${STABLE_OPENCV4}
CTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA11l}_${STABLE_TF2}_${STABLE_OPENCV3}
CTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA11l}_${STABLE_TF2}_${STABLE_OPENCV4}

##### Tensorflow _ OpenCV (aka TO)
TO_BUILDALL =tensorflow_opencv-${STABLE_TF2}_${STABLE_OPENCV3}
TO_BUILDALL+=tensorflow_opencv-${STABLE_TF2}_${STABLE_OPENCV4}

##### Jupyter Notebook ready based on TO & CTO
TO_JUP=jupyter_to-${STABLE_TF2}_${STABLE_OPENCV4}
CTO_JUP=jupyter_cto-${STABLE_CUDA11p}_${STABLE_TF2}_${STABLE_OPENCV4}

## By default, provide the list of build targets
all:
	@echo "**** Docker Image tag ending: ${CTO_RELEASE}"
	@echo ""
	@echo "*** Available Docker images to be built (make targets):"
	@echo "  tensorflow_opencv (aka TO): "; echo -n "      "; echo ${TO_BUILDALL} | sed -e 's/ /\n      /g'
	@echo "  cudnn_tensorflow_opencv (aka CTO): "; echo -n "      "; echo ${CTO_BUILDALL} | sed -e 's/ /\n      /g'
	@echo ""
	@echo "** To build all TO & CTO, use: make build_all"
	@echo ""
	@echo "*** Jupyter Notebook ready containers (requires the base TO & CTO container to be built, will pull otherwise)"
	@echo "  jupyter_to: "; echo -n "      "; echo ${TO_JUP}
	@echo "  jupyter_cto: "; echo -n "      "; echo ${CTO_JUP}
	@echo "  jupyter_all: jupyter_to jupyter_cto"
	@echo ""
	@echo "Note: TensorFlow GPU support can only be compiled for CuDNN containers"

## special command to build all targets
build_all:
	@make ${TO_BUILDALL}
	@make ${CTO_BUILDALL}

tensorflow_opencv:
	@make ${TO_BUILDALL}

cudnn_tensorflow_opencv:
	@make ${CTO_BUILDALL}

${TO_BUILDALL}:
	@CUDX="" CUDX_COMP="" BTARG="$@" make build_prep

${CTO_BUILDALL}:
	@CUDX="cudnn" CUDX_COMP="-DWITH_CUDNN=ON -DOPENCV_DNN_CUDA=ON" BTARG="$@" make build_prep
# CUDA_ARCH_BIN and CUDX_FROM are set in build_prep now 

build_prep:
	@$(eval CTO_NAME=$(shell echo ${BTARG} | cut -d- -f 1))
	@$(eval TARGET_VALUE=$(shell echo ${BTARG} | cut -d- -f 2))
	@$(eval CTO_SC=$(shell echo ${TARGET_VALUE} | grep -o "_" | wc -l)) # where 2 means 3 components
	@$(eval CTO_BUILD=$(shell if [ ${CTO_SC} == 1 ]; then echo "CPU"; else echo "GPU"; fi))
	@$(eval CTO_V=$(shell if [ ${CTO_SC} == 1 ]; then echo "0_${TARGET_VALUE}"; else echo "${TARGET_VALUE}"; fi))
	@$(eval CTO_CUDA_VERSION=$(shell echo ${CTO_V} | cut -d_ -f 1))
	@$(eval CTO_CUDA_PRIMEVERSION=$(shell echo ${CTO_CUDA_VERSION} | perl -pe 's/^(\d+\.\d+)\.\d+$$/$$1/;s/\.\d+/.0/'))
	@$(eval CTO_CUDA_USEDVERSION=$(shell echo ${CTO_CUDA_VERSION} | perl -pe 's/^(\d+\.\d+)\.\d+$$/$$1/;s/\./\-/'))
	@$(eval CTO_TENSORFLOW_VERSION=$(shell echo ${CTO_V} | cut -d_ -f 2))
	@$(eval CTO_OPENCV_VERSION=$(shell echo ${CTO_V} | cut -d_ -f 3))

# Two CUDA11 possiblities, work with both
	@$(eval STABLE_CUDA11=$(shell if [ "A${CTO_CUDA_PRIMEVERSION}" == "A11.0" ]; then echo ${CTO_CUDA_VERSION}; else echo "_____"; fi))

# Nvidia's container requires Ubuntu 20.04 for CUDA11 + CPU only now use Ubuntu 20.04, only CUDA9 and CUDA10 uses 18.04
	@$(eval CTO_UBUNTU=$(shell if [ "A${CTO_CUDA_VERSION}" == "A${STABLE_CUDA11}" ]; then echo "ubuntu20.04"; else if [ ${CTO_SC} == 1 ]; then echo "ubuntu20.04"; else echo "ubuntu18.04"; fi; fi))

	@$(eval CTO_TMP=${CTO_TENSORFLOW_VERSION})
	@$(eval CTO_TF_OPT=$(shell if [ "A${CTO_TMP}" == "A${STABLE_TF1}" ]; then echo "v1"; else echo "v2"; fi))
	@$(eval CTO_TF_KERAS=$(shell if [ "A${CTO_TMP}" == "A${STABLE_TF1}" ]; then echo ${TF1_KERAS}; else echo ${TF2_KERAS}; fi))

	@$(eval CTO_TF_NUMPY=$(shell if [ "A${CTO_TMP}" == "A${STABLE_TF1}" ]; then echo ${TF1_NUMPY}; else echo ${TF2_NUMPY}; fi))

	@$(eval CTO_TMP=${CTO_TENSORFLOW_VERSION}_${CTO_OPENCV_VERSION}-${CTO_RELEASE})
	@$(eval CTO_TAG=$(shell if [ ${CTO_SC} == 1 ]; then echo ${CTO_TMP}; else echo ${CTO_CUDA_VERSION}_${CTO_TMP}; fi))

# 20.04: https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/
	@$(eval CTO_TMP20=$(shell if [ "A${CUDA11_APT_XTRA}" == "A" ]; then echo ""; else echo "cuda-libraries-${CTO_CUDA_USEDVERSION} cuda-libraries-dev-${CTO_CUDA_USEDVERSION} cuda-tools-${CTO_CUDA_USEDVERSION} cuda-toolkit-${CTO_CUDA_USEDVERSION} libcublas-${CTO_CUDA_USEDVERSION} libcublas-dev-${CTO_CUDA_USEDVERSION} libcufft-${CTO_CUDA_USEDVERSION} libcufft-dev-${CTO_CUDA_USEDVERSION} libnccl2 libnccl-dev libnpp-${CTO_CUDA_USEDVERSION} libnpp-dev-${CTO_CUDA_USEDVERSION}"; fi))
	@$(eval CTO_CUDA_APT=$(shell if [ ${CTO_SC} == 1 ]; then echo ""; else if [ "A${CTO_UBUNTU}" == "Aubuntu18.04" ]; then echo ${CTO_TMP18}; else echo ${CTO_TMP20}; fi; fi))

	@$(eval CTO_DNN_ARCH=$(shell if [ ${CTO_SC} == 1 ]; then echo ""; else echo "${DNN_ARCH_CUDA11}"; fi))
	@$(eval CUDX_COMP=$(shell if [ ${CTO_SC} == 1 ]; then echo ""; else echo "${CUDX_COMP} -DCUDA_ARCH_BIN=${CTO_DNN_ARCH}"; fi))

	@$(eval CUDX_FROM=$(shell if [ "A${CUDX}" == "Acudnn" ]; then echo "-cudnn8"; else echo ""; fi))

	@$(eval CTO_FROM=$(shell if [ ${CTO_SC} == 1 ]; then echo "ubuntu:20.04"; else echo "nvidia/cuda:${CTO_CUDA_VERSION}${CUDX_FROM}-devel-${CTO_UBUNTU}"; fi))

	@$(eval CTO_TMP="-D WITH_CUDA=ON -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda -D CMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs -D CUDA_FAST_MATH=1 -D WITH_CUBLAS=1 ${CUDX_COMP} -D WITH_NVCUVID=ON")
	@$(eval CTO_CUDA_BUILD=$(shell if [ ${CTO_SC} == 1 ]; then echo ""; else echo ${CTO_TMP}; fi))

# Enable Non-free?
	$(eval CTO_OPENCV_NONFREE=$(shell if [ "A${CTO_ENABLE_NONFREE}" == "Aunredistributable" ]; then echo "-DOPENCV_ENABLE_NONFREE=ON"; else echo ""; fi))
	$(eval CTO_FFMPEG_NONFREE=$(shell if [ "A${CTO_ENABLE_NONFREE}" == "Aunredistributable" ]; then echo "--enable-nonfree --enable-libnpp"; else echo ""; fi))

	@echo ""; echo "";
	@echo "[*****] Build: datamachines/${CTO_NAME}:${CTO_TAG}";\

	@if [ "A${DOCKERPULL}" == "Ayes" ]; then echo "** Base image: ${CTO_FROM}"; docker pull ${CTO_FROM}; echo ""; else if [ -f ./${CTO_NAME}-${CTO_TAG}.log ]; then echo "  !! Log file (${CTO_NAME}-${CTO_TAG}.log) exists, skipping rebuild (remove to force)"; echo ""; else CTO_NAME=${CTO_NAME} CTO_TAG=${CTO_TAG} CTO_UBUNTU=${CTO_UBUNTU} CTO_FROM=${CTO_FROM} CTO_TENSORFLOW_VERSION=${CTO_TENSORFLOW_VERSION} CTO_OPENCV_VERSION=${CTO_OPENCV_VERSION} CTO_NUMPROC=$(CTO_NUMPROC) CTO_CUDA_APT="${CTO_CUDA_APT}" CTO_CUDA_BUILD="${CTO_CUDA_BUILD}" CTO_TF_OPT="${CTO_TF_OPT}" CTO_TF_KERAS="${CTO_TF_KERAS}" CTO_TF_NUMPY="${CTO_TF_NUMPY}" CTO_CUDA11_APT_XTRA="${CUDA11_APT_XTRA}" CTO_DNN_ARCH="${CTO_DNN_ARCH}" CTO_BUILD="${CTO_BUILD}" CTO_OPENCV_NONFREE="${CTO_OPENCV_NONFREE}" CTO_FFMPEG_NONFREE="${CTO_FFMPEG_NONFREE}" make actual_build; fi; fi


actual_build:
# Build prep
	@$(eval VAR_NT="${CTO_NAME}-${CTO_TAG}")
	@$(eval VAR_DD="BuildInfo/${VAR_NT}")
	@$(eval VAR_CV="BuildInfo/${VAR_NT}/${VAR_NT}-OpenCV.txt")
	@$(eval VAR_TF="BuildInfo/${VAR_NT}/${VAR_NT}-TensorFlow.txt")
	@$(eval VAR_FF="BuildInfo/${VAR_NT}/${VAR_NT}-FFmpeg.txt")
	@$(eval VAR_PT="BuildInfo/${VAR_NT}/${VAR_NT}-PyTorch.txt")
	@mkdir -p ${VAR_DD}
	@echo ""
	@echo "  CTO_FROM               : ${CTO_FROM}" | tee ${VAR_CV} | tee ${VAR_TF} | tee ${VAR_FF} | tee ${VAR_PT}
	@echo ""
	@echo "-- Docker command to be run:"
#	@echo "docker buildx build --progress plain --allow network.host --platform linux/amd64 ${DOCKER_BUILD_ARGS} \\" > ${VAR_NT}.cmd
	@echo "docker build ${DOCKER_BUILD_ARGS} \\" > ${VAR_NT}.cmd
	@echo "  --build-arg CTO_FROM=\"${CTO_FROM}\" \\" >> ${VAR_NT}.cmd
	@echo "  --build-arg CTO_BUILD=\"${CTO_BUILD}\" \\" >> ${VAR_NT}.cmd
	@echo "  --build-arg CTO_TENSORFLOW_VERSION=\"${CTO_TENSORFLOW_VERSION}\" \\" >> ${VAR_NT}.cmd
	@echo "  --build-arg CTO_OPENCV_VERSION=\"${CTO_OPENCV_VERSION}\" \\" >> ${VAR_NT}.cmd
	@echo "  --build-arg CTO_NUMPROC=\"$(CTO_NUMPROC)\" \\" >> ${VAR_NT}.cmd
	@echo "  --build-arg CTO_CUDA_APT=\"${CTO_CUDA_APT}\" \\" >> ${VAR_NT}.cmd
	@echo "  --build-arg CTO_CUDA11_APT_XTRA=\"${CTO_CUDA11_APT_XTRA}\" \\" >> ${VAR_NT}.cmd
	@echo "  --build-arg CTO_CUDA_BUILD=\"${CTO_CUDA_BUILD}\" \\" >> ${VAR_NT}.cmd
	@echo "  --build-arg CTO_OPENCV_NONFREE=\"${CTO_OPENCV_NONFREE}\" \\" >> ${VAR_NT}.cmd
	@echo "  --build-arg LATEST_BAZELISK=\"${LATEST_BAZELISK}\" \\" >> ${VAR_NT}.cmd
	@echo "  --build-arg LATEST_BAZEL=\"${LATEST_BAZEL}\" \\" >> ${VAR_NT}.cmd
	@echo "  --build-arg CTO_TF_OPT=\"${CTO_TF_OPT}\" \\" >> ${VAR_NT}.cmd
	@echo "  --build-arg CTO_TF_KERAS=\"${CTO_TF_KERAS}\" \\" >> ${VAR_NT}.cmd
	@echo "  --build-arg CTO_TF_NUMPY=\"${CTO_TF_NUMPY}\" \\" >> ${VAR_NT}.cmd
	@echo "  --build-arg CTO_DNN_ARCH=\"${CTO_DNN_ARCH}\" \\" >> ${VAR_NT}.cmd
	@echo "  --build-arg CTO_FFMPEG_VERSION=\"${CTO_FFMPEG_VERSION}\" \\" >> ${VAR_NT}.cmd
	@echo "  --build-arg CTO_FFMPEG_NVCODEC=\"${CTO_FFMPEG_NVCODEC}\" \\" >> ${VAR_NT}.cmd
	@echo "  --build-arg CTO_FFMPEG_NONFREE=\"${CTO_FFMPEG_NONFREE}\" \\" >> ${VAR_NT}.cmd
	@echo "  --build-arg CTO_MAGMA=\"${CTO_MAGMA}\" \\" >> ${VAR_NT}.cmd
	@echo "  --build-arg CTO_TORCH=\"${CTO_TORCH}\" \\" >> ${VAR_NT}.cmd
	@echo "  --build-arg CTO_TORCHVISION=\"${CTO_TORCHVISION}\" \\" >> ${VAR_NT}.cmd
	@echo "  --build-arg CTO_TORCHAUDIO=\"${CTO_TORCHAUDIO}\" \\" >> ${VAR_NT}.cmd
	@echo "  --tag=\"datamachines/${CTO_NAME}:${CTO_TAG}\" \\" >> ${VAR_NT}.cmd
	@echo "  -f ${CTO_UBUNTU}/Dockerfile \\" >> ${VAR_NT}.cmd
	@echo "  ." >> ${VAR_NT}.cmd
	@cat ${VAR_NT}.cmd | tee ${VAR_NT}.log.temp | tee -a ${VAR_CV} | tee -a ${VAR_TF} | tee -a ${VAR_FF} | tee -a ${VAR_PT}
	@echo "" | tee -a ${VAR_NT}.log.temp
	@echo "Press Ctl+c within 5 seconds to cancel"
	@for i in 5 4 3 2 1; do echo -n "$$i "; sleep 1; done; echo ""
# Actual build
	@chmod +x ./${VAR_NT}.cmd
	@./${VAR_NT}.cmd | tee -a ${VAR_NT}.log.temp; exit "$${PIPESTATUS[0]}"
	@fgrep "CUDA NVCC" ${VAR_NT}.log.temp >> ${VAR_CV} || true
	@docker run --rm datamachines/${CTO_NAME}:${CTO_TAG} opencv_version -v >> ${VAR_CV}
	@docker run --rm datamachines/${CTO_NAME}:${CTO_TAG} /tmp/tf_info.sh >> ${VAR_TF}
	@printf "\n\n***** FFmpeg configuration:\n" >> ${VAR_FF}; docker run --rm datamachines/${CTO_NAME}:${CTO_TAG} cat /tmp/ffmpeg_config.txt >> ${VAR_FF}
	@printf "\n\n***** PyTorch configuration:\n" >> ${VAR_PT}; docker run --rm datamachines/${CTO_NAME}:${CTO_TAG} cat /tmp/torch_config.txt >> ${VAR_PT}
	@printf "\n\n***** TorchVision configuration:\n" >> ${VAR_PT}; docker run --rm datamachines/${CTO_NAME}:${CTO_TAG} cat /tmp/torchvision_config.txt >> ${VAR_PT}
	@printf "\n\n***** TorchAudio configuration:\n" >> ${VAR_PT}; docker run --rm datamachines/${CTO_NAME}:${CTO_TAG} cat /tmp/torchaudio_config.txt >> ${VAR_PT}
	@if [ "A${MLTK_CHECK}" == "Ayes" ]; then CTO_NAME=${CTO_NAME} CTO_TAG=${CTO_TAG} make force_mltk_check; fi
	@mv ${VAR_NT}.log.temp ${VAR_NT}.log
	@rm -f ./${VAR_NT}.cmd

##### Force ML Toolkit checks
force_mltk_check:
	@docker run --rm -v `pwd`:/dmc --gpus all datamachines/${CTO_NAME}:${CTO_TAG} python3 /dmc/test/tf_hw.py | tee -a BuildInfo/${CTO_NAME}-${CTO_TAG}/${CTO_NAME}-${CTO_TAG}-TensorFlow.txt; exit "$${PIPESTATUS[0]}"
	@docker run --rm -v `pwd`:/dmc --gpus all datamachines/${CTO_NAME}:${CTO_TAG} python3 /dmc/test/pt_hw.py | tee -a BuildInfo/${CTO_NAME}-${CTO_TAG}/${CTO_NAME}-${CTO_TAG}-PyTorch.txt; exit "$${PIPESTATUS[0]}"


##########
##### Jupyter Notebook
JN_MODE=""
JN_UID=$(shell id -u)
JN_GID=$(shell id -g)

jupyter_all:
	@make jupyter_to
	@make jupyter_cto

jupyter_to:
	@make ${TO_JUP}

jupyter_cto:
	@make ${CTO_JUP}

${TO_JUP}:
	@$(eval JN=$(shell echo ${TO_JUP} | sed 's/-/:/'))
	@$(eval JB=$(shell echo ${JN} | cut -d : -f 1))
	@$(eval JT=$(shell echo ${JN} | cut -d : -f 2))
	@$(eval JN="${JB}${JN_MODE}:${JT}")
	@cd Jupyter_build; docker build --build-arg JUPBC="datamachines/tensorflow_opencv:${JT}-${CTO_RELEASE}" --build-arg JUID=${JN_UID} --build-arg JGID=${JN_GID} -f Dockerfile${JN_MODE} --tag="datamachines/${JN}-${CTO_RELEASE}" .

${CTO_JUP}:
	@$(eval JN=$(shell echo ${CTO_JUP} | sed 's/-/:/'))
	@$(eval JB=$(shell echo ${JN} | cut -d : -f 1))
	@$(eval JT=$(shell echo ${JN} | cut -d : -f 2))
	@$(eval JN="${JB}${JN_MODE}:${JT}")
	@cd Jupyter_build; docker build --build-arg JUPBC="datamachines/cudnn_tensorflow_opencv:${JT}-${CTO_RELEASE}" --build-arg JUID=${JN_UID} --build-arg JGID=${JN_GID} -f Dockerfile${JN_MODE} --tag="datamachines/${JN}-${CTO_RELEASE}" .

##### Various cleanup
clean:
	rm -f *.log.temp

allclean:
	@make clean
	rm -f *.log