#!/bin/bash

tag=`make -f Makefile | grep tag | cut -d ':' -f 2`
list=`make -f Makefile| grep '-' | tr -s ' ' | cut -d ' ' -f 2`

# Link to the Github release (do not forget to do the release :) )
gh="https://github.com/datamachines/cuda_tensorflow_opencv/releases/tag/"
# Base link for OpenCV build dumps
lgb="https://github.com/datamachines/cuda_tensorflow_opencv/blob/master/OpenCV_BuildConf/"
# CuDNN version (consistent among entries)
dnn="7.6.5"
# Ubuntu version (consistent among entries)
ub="18.04"

for i in $list;
do
  v=`echo $i | sed 's/-/:/'`
  t=`echo $tag | sed 's/\s+//'`

  g="$v $t"
# t_o:      | Docker Tag | TensorFlow | OpenCV | Ubuntu | Github Link | OpenCV Build Conf | 
# c_t_o :   | Docker Tag | CUDA | TensorFlow | OpenCV | Ubuntu | Github Link | OpenCV Build Conf | 
# n_t_o :   | Docker Tag | CUDA | CUDNN | TensorFlow | OpenCV | Ubuntu | Github Link | OpenCV Build Conf |
  echo "$g" | perl -ne '@it = ($_ =~ m%^(.+)\:([\d\.]+\_)?([\d\.]+)\_([\d\.]+)\s(\d+)$%); $n = shift @it; $x = pop @it; ($c, $t, $o) = @it; $c=~s%\_$%%; $f= (($c ne "") ? "$c\_": "") . "$t\_$o"; print "| $f-$x "; print "| $c " if ($c != 0); print "| '$dnn' " if ($n =~ m%^cudnn_%); print "| $t | $o | '$ub' | [link]('$gh'$x) | [link]('$lgb'$n-$f-$x.txt) |\n\n";'

  
done
