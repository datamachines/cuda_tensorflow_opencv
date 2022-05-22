#!/bin/bash

tag=`make -f ../Makefile | grep tag | cut -d ':' -f 2`
list=`make -f ../Makefile| grep -v jupyter | grep '-' | tr -s ' ' | cut -d ' ' -f 2`

for i in $list;
do
  v=`echo $i | sed 's/-/:/'`
  t=`echo $tag | sed 's/\s+//'`
  CONTAINER_ID="datamachines/$v-$t"

  echo ""
  echo "############################################################"
  echo "#################### ${CONTAINER_ID}"
  
  echo "########## Test 1: Confirm ML toolkits are available, display version and available hardware"
  CONTAINER_ID="${CONTAINER_ID}" ../runDocker.sh -N -X -c python3 /dmc/tf_hw.py
  sleep 1

  echo "########## Test 2: TensorFlow requesting CPU Matrix Multiplication"
  CONTAINER_ID="${CONTAINER_ID}" time ../runDocker.sh -N -X -c python3 /dmc/tf_cputest.py
  sleep 1

  echo "########## Test 3: TensorFlow requesting GPU Matrix Multiplication"
  CONTAINER_ID="${CONTAINER_ID}" time ../runDocker.sh -N -X -c python3 /dmc/tf_gputest.py
  sleep 1

  sleep 2
done

echo "############################################################"
echo "Done"
