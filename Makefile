# Needed SHELL since I'm using zsh
SHELL := /bin/bash
.PHONY: all build_all

# Release to match data of Dockerfile and follow YYYYMMDD pattern
CTO_RELEASE=20191107

# Maximize build speed
CTO_NUMPROC := $(shell nproc --all)

STABLE_OPENCV3=3.4.8
STABLE_OPENCV4=4.1.2

STABLE_TF12=1.12.3

##### CUDA _ Tensorflow _ OpenCV
# the Tensorflow 1.12.3 uses CUDA 9.0 on Ubuntu 16.04 
CTO_BUILDALL =9.0_${STABLE_TF12}_${STABLE_OPENCV3}
CTO_BUILDALL+=9.0_${STABLE_TF12}_${STABLE_OPENCV4}

##### Tensorflow _ OpenCV
TO_BUILDALL =${STABLE_TF12}_${STABLE_OPENCV3}
TO_BUILDALL+=${STABLE_TF12}_${STABLE_OPENCV4}

## By default, provide the list of build targets
all:
	@echo "** Docker Image tag ending: ${CTO_RELEASE}"
	@echo ""
	@echo "** Available Docker images to be built (make targets):"
	@echo "  cuda_tensorflow_opencv: ${CTO_BUILDALL}"
	@echo "  tensorflow_opencv: ${TO_BUILDALL}"
	@echo ""
	@echo "** To build all, use: make build_all"

## special command to build all targets
build_all:
	@make ${CTO_BUILDALL}
	@make ${TO_BUILDALL}

cuda_tensorflow_opencv:
	@make ${CTO_BUILDALL}

tensorflow_opencv:
	@make ${TO_BUILDALL}

${CTO_BUILDALL} ${TO_BUILDALL}:
	@$(eval CTO_SC=$(shell echo $@ | grep -o "_" | wc -l)) # where 2 means 3 components
	@$(eval CTO_V=$(shell if [ ${CTO_SC} == 1 ]; then echo "0_$@"; else echo "$@"; fi))
	@$(eval CTO_CUDA_VERSION=$(shell echo ${CTO_V} | cut -d_ -f 1))
	@$(eval CTO_TENSORFLOW_VERSION=$(shell echo ${CTO_V} | cut -d_ -f 2))
	@$(eval CTO_OPENCV_VERSION=$(shell echo ${CTO_V} | cut -d_ -f 3))

	@$(eval CTO_TMP=${CTO_TENSORFLOW_VERSION})
	@$(eval CTO_TENSORFLOW_TAG=$(shell if [ ${CTO_SC} == 1 ]; then echo ${CTO_TMP}-py3; else echo ${CTO_TMP}-gpu-py3; fi))

	@$(eval CTO_TMP=${CTO_TENSORFLOW_VERSION}_${CTO_OPENCV_VERSION}-${CTO_RELEASE})
	@$(eval CTO_TAG=$(shell if [ ${CTO_SC} == 1 ]; then echo ${CTO_TMP}; else echo ${CTO_CUDA_VERSION}_${CTO_TMP}; fi))

	@$(eval CTO_TMP="tensorflow_opencv")
	@$(eval CTO_NAME=$(shell if [ ${CTO_SC} == 1 ]; then echo ${CTO_TMP}; else echo cuda_${CTO_TMP}; fi))

	@$(eval CTO_TMP="cuda-npp-dev-${CTO_CUDA_VERSION} cuda-cublas-dev-${CTO_CUDA_VERSION} cuda-cufft-dev-${CTO_CUDA_VERSION} cuda-libraries-dev-${CTO_CUDA_VERSION}")
	@$(eval CTO_CUDA_APT=$(shell if [ ${CTO_SC} == 1 ]; then echo ""; else echo ${CTO_TMP}; fi))

	@$(eval CTO_TMP="-DWITH_CUDA=ON -DCUDA_FAST_MATH=1 -DWITH_CUBLAS=1 -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-${CTO_CUDA_VERSION} -DCUDA_cublas_LIBRARY=cublas -DCUDA_cufft_LIBRARY=cufft -DCUDA_nppim_LIBRARY=nppim -DCUDA_nppidei_LIBRARY=nppidei -DCUDA_nppif_LIBRARY=nppif -DCUDA_nppig_LIBRARY=nppig -DCUDA_nppim_LIBRARY=nppim -DCUDA_nppist_LIBRARY=nppist -DCUDA_nppisu_LIBRARY=nppisu -DCUDA_nppitc_LIBRARY=nppitc -DCUDA_npps_LIBRARY=npps -DCUDA_nppc_LIBRARY=nppc -DCUDA_nppial_LIBRARY=nppial -DCUDA_nppicc_LIBRARY=nppicc -D CUDA_nppicom_LIBRARY=nppicom")
	@$(eval CTO_CUDA_BUILD=$(shell if [ ${CTO_SC} == 1 ]; then echo ""; else echo ${CTO_TMP}; fi))

	@echo ""; echo ""
	@echo "[*****] About to build datamachines/${CTO_NAME}:${CTO_TAG}"
	@echo "Press Ctl+c within 5 seconds to cancel"
	@for i in 5 4 3 2 1; do echo -n "$$i "; sleep 1; done; echo ""
	docker build \
	  --build-arg CTO_TENSORFLOW_TAG=$(CTO_TENSORFLOW_TAG) \
	  --build-arg CTO_OPENCV_VERSION=${CTO_OPENCV_VERSION} \
	  --build-arg CTO_NUMPROC=$(CTO_NUMPROC) \
	  --build-arg CTO_CUDA_APT="${CTO_CUDA_APT}" \
	  --build-arg CTO_CUDA_BUILD="${CTO_CUDA_BUILD}" \
	  --tag="datamachines/${CTO_NAME}:${CTO_TAG}" \
	  . | tee ${CTO_TAG}.log.temp; exit "$${PIPESTATUS[0]}"
	@mv ${CTO_TAG}.log.temp ${CTO_TAG}.log
