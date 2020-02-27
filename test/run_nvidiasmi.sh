#!/bin/bash

tag=`make -f ../Makefile | grep tag | cut -d ':' -f 2`
list=`make -f ../Makefile| grep '-' | grep cud | tr -s ' ' | cut -d ' ' -f 2`

for i in $list;
do
  v=`echo $i | sed 's/-/:/'`
  t=`echo $tag | sed 's/\s+//'`
  CONTAINER_ID="datamachines/$v-$t"

  echo ""
  echo "############################################################"
  echo "#################### Nvidia-smi test for ${CONTAINER_ID}"
  
  CONTAINER_ID="${CONTAINER_ID}" ../runDocker.sh -N -X -c /usr/bin/nvidia-smi

done

echo "############################################################"
echo "Done"
