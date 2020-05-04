#!/bin/bash

tag=`make -f Makefile | grep tag | cut -d ':' -f 2`
list=`make -f Makefile| grep -E '\-[[:digit:]]' | tr -s ' ' | cut -d ' ' -f 2`

# Link to the Github release tag (do not forget to tag, push the tag and do the release)
gh="https://github.com/datamachines/cuda_tensorflow_opencv/tree/"
# Base link for OpenCV build dumps
lgb="https://github.com/datamachines/cuda_tensorflow_opencv/blob/master/JetsonNano/OpenCV_BuildConf/"
# CuDNN version (consistent among entries)
dnn="7.6.5"
# Ubuntu version (consistent among entries)
ub="18.04"

for i in $list;
do
  v=`echo $i | sed 's/-/:/2'`
  t=`echo $tag | sed 's/\s+//'`

  g="$v $t"
# j-c_t_o :   | Docker Tag | CUDA | TensorFlow | OpenCV | Ubuntu | Github Link | OpenCV Build Conf | 
# j-n_t_o :   | Docker Tag | CUDA | CUDNN | TensorFlow | OpenCV | Ubuntu | Github Link | OpenCV Build Conf |
  echo "$g" | perl -ne '@it = ($_ =~ m%^(.+)\:([\d\.]+\_)?([\d\.]+)\_([\d\.]+)\s(\d+)$%); $n = shift @it; $x = pop @it; ($c, $t, $o) = @it; $c=~s%\_$%%; $f= (($c ne "") ? "$c\_": "") . "$t\_$o"; print "| $f-$x "; print "| $c " if ($c != 0); print "| '$dnn' " if ($n =~ m%^cudnn_%); print "| $t | $o | '$ub' | [link]('$gh'$x) | [link]('$lgb'$n-$f-$x.txt) |\n\n";'

  
done
