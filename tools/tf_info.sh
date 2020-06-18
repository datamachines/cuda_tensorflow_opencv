#!/bin/bash

# Expected to be run in container
if [ ! -f /.within_container ]; then
  echo "Tool expected to be ran within the container, aborting"
  exit 1
fi

echo "[pip list]"
pip3 list

echo ""
echo "-------------------------------------------------------"
echo ""

echo "[TensorFlow build information]"
cat /tmp/tf_env.dump

