# Needed SHELL since I'm using zsh
SHELL := /bin/bash
.PHONY: all build_all actual_build build_prep

# Release to match data of Dockerfile and follow YYYYMMDD pattern
CTO_RELEASE=20200211

# Maximize build speed
CTO_NUMPROC := $(shell nproc --all)

# According to https://hub.docker.com/r/nvidia/cuda/
# Building .1 less for latest to help with systems which do not have the latest driver installed
# (example: latest is 10.2, will also build 10.1)
STABLE_CUDA9=9.2
STABLE_CUDA10p=10.1
STABLE_CUDA10=10.2
# CUDNN needs 5.3 at minimum, extending list from https://en.wikipedia.org/wiki/CUDA#GPUs_supported 
DNN_ARCH_CUDA9=5.3,6.0,6.1,6.2
DNN_ARCH_CUDA10=5.3,6.0,6.1,6.2,7.0,7.2,7.5

# According to https://opencv.org/releases/
STABLE_OPENCV3=3.4.9
STABLE_OPENCV4=4.2.0

# According to https://github.com/tensorflow/tensorflow/blob/master/RELEASE.md
STABLE_TF1=1.15.2
STABLE_TF2=2.1.0

##### CUDA _ Tensorflow _ OpenCV
CTO_BUILDALL =cuda_tensorflow_opencv-${STABLE_CUDA9}_${STABLE_TF1}_${STABLE_OPENCV3}
CTO_BUILDALL+=cuda_tensorflow_opencv-${STABLE_CUDA9}_${STABLE_TF1}_${STABLE_OPENCV4}
CTO_BUILDALL+=cuda_tensorflow_opencv-${STABLE_CUDA9}_${STABLE_TF2}_${STABLE_OPENCV3}
CTO_BUILDALL+=cuda_tensorflow_opencv-${STABLE_CUDA9}_${STABLE_TF2}_${STABLE_OPENCV4}
CTO_BUILDALL+=cuda_tensorflow_opencv-${STABLE_CUDA10p}_${STABLE_TF1}_${STABLE_OPENCV3}
CTO_BUILDALL+=cuda_tensorflow_opencv-${STABLE_CUDA10p}_${STABLE_TF1}_${STABLE_OPENCV4}
CTO_BUILDALL+=cuda_tensorflow_opencv-${STABLE_CUDA10p}_${STABLE_TF2}_${STABLE_OPENCV3}
CTO_BUILDALL+=cuda_tensorflow_opencv-${STABLE_CUDA10p}_${STABLE_TF2}_${STABLE_OPENCV4}
CTO_BUILDALL+=cuda_tensorflow_opencv-${STABLE_CUDA10}_${STABLE_TF1}_${STABLE_OPENCV3}
CTO_BUILDALL+=cuda_tensorflow_opencv-${STABLE_CUDA10}_${STABLE_TF1}_${STABLE_OPENCV4}
CTO_BUILDALL+=cuda_tensorflow_opencv-${STABLE_CUDA10}_${STABLE_TF2}_${STABLE_OPENCV3}
CTO_BUILDALL+=cuda_tensorflow_opencv-${STABLE_CUDA10}_${STABLE_TF2}_${STABLE_OPENCV4}

##### CuDNN _ Tensorflow _ OpenCV
DTO_BUILDALL =cudnn_tensorflow_opencv-${STABLE_CUDA9}_${STABLE_TF1}_${STABLE_OPENCV3}
DTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA9}_${STABLE_TF1}_${STABLE_OPENCV4}
DTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA9}_${STABLE_TF2}_${STABLE_OPENCV3}
DTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA9}_${STABLE_TF2}_${STABLE_OPENCV4}
DTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA10p}_${STABLE_TF1}_${STABLE_OPENCV3}
DTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA10p}_${STABLE_TF1}_${STABLE_OPENCV4}
DTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA10p}_${STABLE_TF2}_${STABLE_OPENCV3}
DTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA10p}_${STABLE_TF2}_${STABLE_OPENCV4}
DTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA10}_${STABLE_TF1}_${STABLE_OPENCV3}
DTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA10}_${STABLE_TF1}_${STABLE_OPENCV4}
DTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA10}_${STABLE_TF2}_${STABLE_OPENCV3}
DTO_BUILDALL+=cudnn_tensorflow_opencv-${STABLE_CUDA10}_${STABLE_TF2}_${STABLE_OPENCV4}

##### Tensorflow _ OpenCV
TO_BUILDALL =tensorflow_opencv-${STABLE_TF1}_${STABLE_OPENCV3}
TO_BUILDALL+=tensorflow_opencv-${STABLE_TF1}_${STABLE_OPENCV4}
TO_BUILDALL+=tensorflow_opencv-${STABLE_TF2}_${STABLE_OPENCV3}
TO_BUILDALL+=tensorflow_opencv-${STABLE_TF2}_${STABLE_OPENCV4}

## By default, provide the list of build targets
all:
	@echo "** Docker Image tag ending: ${CTO_RELEASE}"
	@echo ""
	@echo "** Available Docker images to be built (make targets):"
	@echo "  tensorflow_opencv: "; echo -n "      "; echo ${TO_BUILDALL} | sed -e 's/ /\n      /g'
	@echo "  cuda_tensorflow_opencv: "; echo -n "      "; echo ${CTO_BUILDALL} | sed -e 's/ /\n      /g'
	@echo "  cudnn_tensorflow_opencv: "; echo -n "      "; echo ${DTO_BUILDALL} | sed -e 's/ /\n      /g'
	@echo ""
	@echo "** To build all, use: make build_all"

## special command to build all targets
build_all:
	@make ${TO_BUILDALL}
	@make ${CTO_BUILDALL}
	@make ${DTO_BUILDALL}

tensorflow_opencv:
	@make ${TO_BUILDALL}

cuda_tensorflow_opencv:
	@make ${CTO_BUILDALL}

cudnn_tensorflow_opencv:
	@make ${DTO_BUILDALL}

${TO_BUILDALL}:
	@CUDX="" CUDX_FROM="" CUDX_COMP="" BTARG="$@" make build_prep

${CTO_BUILDALL}:
	@CUDX="cuda" CUDX_FROM="" CUDX_COMP="" BTARG="$@" make build_prep

${DTO_BUILDALL}:
	@CUDX="cudnn" CUDX_FROM="-cudnn7" CUDX_COMP="-DWITH_CUDNN=ON -DOPENCV_DNN_CUDA=ON" BTARG="$@" make build_prep
# CUDA_ARCH_BIN is set in build_prep now 

build_prep:
	@$(eval CTO_NAME=$(shell echo ${BTARG} | cut -d- -f 1))
	@$(eval TARGET_VALUE=$(shell echo ${BTARG} | cut -d- -f 2))
	@$(eval CTO_SC=$(shell echo ${TARGET_VALUE} | grep -o "_" | wc -l)) # where 2 means 3 components
	@$(eval CTO_V=$(shell if [ ${CTO_SC} == 1 ]; then echo "0_${TARGET_VALUE}"; else echo "${TARGET_VALUE}"; fi))
	@$(eval CTO_CUDA_VERSION=$(shell echo ${CTO_V} | cut -d_ -f 1))
	@$(eval CTO_CUDA_PRIMEVERSION=$(shell echo ${CTO_CUDA_VERSION} | perl -pe 's/\.\d+/.0/'))
	@$(eval CTO_TENSORFLOW_VERSION=$(shell echo ${CTO_V} | cut -d_ -f 2))
	@$(eval CTO_OPENCV_VERSION=$(shell echo ${CTO_V} | cut -d_ -f 3))

	@$(eval CTO_TMP=${CTO_TENSORFLOW_VERSION})
	@$(eval CTO_TENSORFLOW_PYTHON=$(shell if [ "A${CTO_TMP}" == "A${STABLE_TF1}" ]; then echo "tensorflow==${STABLE_TF1}"; else if [ ${CTO_SC} == 1 ]; then echo "tensorflow==${STABLE_TF2}"; else echo "tensorflow-gpu==${STABLE_TF2}"; fi; fi))

	@$(eval CTO_TMP=${CTO_TENSORFLOW_VERSION}_${CTO_OPENCV_VERSION}-${CTO_RELEASE})
	@$(eval CTO_TAG=$(shell if [ ${CTO_SC} == 1 ]; then echo ${CTO_TMP}; else echo ${CTO_CUDA_VERSION}_${CTO_TMP}; fi))

	@$(eval CTO_TMP="cuda-npp-${CTO_CUDA_VERSION} cuda-cublas-${CTO_CUDA_PRIMEVERSION} cuda-cufft-${CTO_CUDA_VERSION} cuda-libraries-${CTO_CUDA_VERSION} cuda-npp-dev-${CTO_CUDA_VERSION} cuda-cublas-dev-${CTO_CUDA_PRIMEVERSION} cuda-cufft-dev-${CTO_CUDA_VERSION} cuda-libraries-dev-${CTO_CUDA_VERSION}")
	@$(eval CTO_CUDA_APT=$(shell if [ ${CTO_SC} == 1 ]; then echo ""; else echo ${CTO_TMP}; fi))

	@$(eval DNN_ARCH=$(shell if [ "A${CTO_CUDA_VERSION}" == "A${STABLE_CUDA9}" ]; then echo "${DNN_ARCH_CUDA9}"; else echo "${DNN_ARCH_CUDA10}"; fi))
	@$(eval CUDX_COMP=$(shell if [ "A${CUDX}" == "Acudnn" ]; then echo "${CUDX_COMP} -DCUDA_ARCH_BIN=${DNN_ARCH}"; else echo "${CUDX_COMP}"; fi))

	@$(eval CTO_FROM=$(shell if [ ${CTO_SC} == 1 ]; then echo "ubuntu:18.04"; else echo "nvidia/cuda:${CTO_CUDA_VERSION}${CUDX_FROM}-devel-ubuntu18.04"; fi))

	@$(eval CTO_TMP="-DWITH_CUDA=ON -DCUDA_FAST_MATH=1 -DWITH_CUBLAS=1 ${CUDX_COMP} -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-${CTO_CUDA_VERSION} -DCUDA_cublas_LIBRARY=cublas -DCUDA_cufft_LIBRARY=cufft -DCUDA_nppim_LIBRARY=nppim -DCUDA_nppidei_LIBRARY=nppidei -DCUDA_nppif_LIBRARY=nppif -DCUDA_nppig_LIBRARY=nppig -DCUDA_nppim_LIBRARY=nppim -DCUDA_nppist_LIBRARY=nppist -DCUDA_nppisu_LIBRARY=nppisu -DCUDA_nppitc_LIBRARY=nppitc -DCUDA_npps_LIBRARY=npps -DCUDA_nppc_LIBRARY=nppc -DCUDA_nppial_LIBRARY=nppial -DCUDA_nppicc_LIBRARY=nppicc -D CUDA_nppicom_LIBRARY=nppicom")
	@$(eval CTO_CUDA_BUILD=$(shell if [ ${CTO_SC} == 1 ]; then echo ""; else echo ${CTO_TMP}; fi))

	@echo ""; echo ""
	@echo "[*****] About to build datamachines/${CTO_NAME}:${CTO_TAG}"

	@if [ -f ./${CTO_NAME}-${CTO_TAG}.log ]; then echo "  !! Log file (${CTO_NAME}-${CTO_TAG}.log) exists, skipping rebuild (remove to force)"; echo ""; else CTO_NAME=${CTO_NAME} CTO_TAG=${CTO_TAG} CTO_FROM=${CTO_FROM} CTO_TENSORFLOW_PYTHON=${CTO_TENSORFLOW_PYTHON} CTO_OPENCV_VERSION=${CTO_OPENCV_VERSION} CTO_NUMPROC=$(CTO_NUMPROC) CTO_CUDA_APT="${CTO_CUDA_APT}" CTO_CUDA_BUILD="${CTO_CUDA_BUILD}" make actual_build; fi


actual_build:
	@echo "Press Ctl+c within 5 seconds to cancel"
	@echo "  CTO_FROM               : ${CTO_FROM}"
	@echo "  CTO_TENSORFLOW_PYTHON  : ${CTO_TENSORFLOW_PYTHON}"
	@for i in 5 4 3 2 1; do echo -n "$$i "; sleep 1; done; echo ""
	docker build \
	  --build-arg CTO_FROM=${CTO_FROM} \
	  --build-arg CTO_TENSORFLOW_PYTHON=${CTO_TENSORFLOW_PYTHON} \
	  --build-arg CTO_OPENCV_VERSION=${CTO_OPENCV_VERSION} \
	  --build-arg CTO_NUMPROC=$(CTO_NUMPROC) \
	  --build-arg CTO_CUDA_APT="${CTO_CUDA_APT}" \
	  --build-arg CTO_CUDA_BUILD="${CTO_CUDA_BUILD}" \
	  --tag="datamachines/${CTO_NAME}:${CTO_TAG}" \
	  . | tee ${CTO_NAME}-${CTO_TAG}.log.temp; exit "$${PIPESTATUS[0]}"
	@mv ${CTO_NAME}-${CTO_TAG}.log.temp ${CTO_NAME}-${CTO_TAG}.log
	@mkdir -p OpenCV_BuildConf
	@perl -0777 -ne 'print $$1 if m%(\-\-\sGeneral\sconfiguration\sfor.+?)\-\-\s*\-\-\sConfiguring\sdone%s' ${CTO_NAME}-${CTO_TAG}.log > OpenCV_BuildConf/${CTO_NAME}-${CTO_TAG}.txt
