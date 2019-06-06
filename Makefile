# Needed SHELL since I'm using zsh
SHELL := /bin/bash
.PHONY: all build_all

# Release to match data of Dockerfile and follow YYYYMMDD pattern
CTO_RELEASE=20190605

# Maximize build speed
CTO_NUMPROC := $(shell nproc --all)

## List of Targets to build: CUDA _ Tensorflow _ OpenCV
CTO_BUILDALL=9.0_1.12.0_3.4.6
CTO_BUILDALL+=9.0_1.12.0_4.1.0
CTO_BUILDALL+=10.0_1.13.1_3.4.6
CTO_BUILDALL+=10.0_1.13.1_4.1.0


## By default, provide the list of build targets
all:
	@echo "** Docker Image tag ending: ${CTO_RELEASE}"
	@echo ""
	@echo "** Available Docker images to be built (make targets):"
	@echo ${CTO_BUILDALL} | sed -e 's/ /\n/g' 
	@echo ""
	@echo "** To build all, use: make build_all"

## special command to build all targets
build_all:
	@make ${CTO_BUILDALL}

${CTO_BUILDALL}:
	$(eval CTO_CUDA_VERSION=$(shell echo $@ | cut -d_ -f 1))
	$(eval CTO_TENSORFLOW_VERSION=$(shell echo $@ | cut -d_ -f 2))
	$(eval CTO_TENSORFLOW_TAG=${CTO_TENSORFLOW_VERSION}-gpu-py3)
	$(eval CTO_OPENCV_VERSION=$(shell echo $@ | cut -d_ -f 3))
	$(eval CTO_TAG=${CTO_CUDA_VERSION}_${CTO_TENSORFLOW_VERSION}_${CTO_OPENCV_VERSION}-${CTO_RELEASE})
	@echo ""; echo ""
	@echo "[*****] About to build cuda_tensorflow_opencv:${CTO_TAG}"
	@echo "Press Ctl+c within 5 seconds to cancel"
	@for i in 5 4 3 2 1; do echo -n "$$i "; sleep 1; done; echo ""
	docker build \
	  --build-arg CTO_TENSORFLOW_TAG=$(CTO_TENSORFLOW_TAG) \
	  --build-arg CTO_CUDA_VERSION=$(CTO_CUDA_VERSION) \
	  --build-arg CTO_OPENCV_VERSION=${CTO_OPENCV_VERSION} \
	  --build-arg CTO_NUMPROC=$(CTO_NUMPROC) \
	  --tag="cuda_tensorflow_opencv:${CTO_TAG}" \
	  . | tee ${CTO_TAG}.log
