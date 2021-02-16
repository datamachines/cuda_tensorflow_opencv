#!/bin/bash
set -e

tmp="/usr/local/cuda/extras/CUPTI/lib64"

if [[ -d $tmp ]] ; then 
  echo $tmp >> /etc/ld.so.conf.d/nvidia-cupti.conf
  ldconfig
  echo "***** CUPTI added to LD path"
fi
