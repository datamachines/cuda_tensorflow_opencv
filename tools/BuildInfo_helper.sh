#!/bin/bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

tag=`make -f ${SCRIPTPATH}/../Makefile | grep tag | cut -d ':' -f 2`
list=`make -f ${SCRIPTPATH}/../Makefile| grep -v jupyter | grep '-' | tr -s ' ' | cut -d ' ' -f 2`

# Link to the Github released tag (do not forget to tag, push the tag and do the release)
gh="https://github.com/datamachines/cuda_tensorflow_opencv/tree/"
# Branch (for testing, keep to master)
bra="master"
# Base link for all build info
bla="https://github.com/datamachines/cuda_tensorflow_opencv/blob/${bra}/BuildInfo/"

skipped=0
missing=0
for i in $list;
do
  v=`echo $i | sed 's/-/:/'`
  t=`echo $tag | sed 's/\s+//'`

  g="$v $t"

  cont=1
  # Confirm we have a matching file (here for possible future extractions)
  l=`echo $g | perl -pe 's%\:%-%;s%\s%-%'`
  of="${SCRIPTPATH}/../BuildInfo/$l/$l-OpenCV.txt"; if [ ! -f $of ]; then echo "***** CV: No $of file, skipping"; cont=0; fi
  tf="${SCRIPTPATH}/../BuildInfo/$l/$l-TensorFlow.txt"; if [ ! -f $tf ]; then echo "***** TF: No $tf file, skipping"; cont=0; fi
  ff="${SCRIPTPATH}/../BuildInfo/$l/$l-FFmpeg.txt"; if [ ! -f $of ]; then echo "***** FF: No $ff file, skipping"; cont=0; fi
  pf="${SCRIPTPATH}/../BuildInfo/$l/$l-PyTorch.txt"; if [ ! -f $of ]; then echo "***** PT: No $pf file, skipping"; cont=0; fi

  if [ $cont == 1 ]; then
    tmp=`fgrep FOUND_UBUNTU $tf | cut -d " " -f 2`
    ub=`if [ "A$tmp" == "A" ]; then echo "**MISSING**";  else echo $tmp; fi`

    tmp=`fgrep FOUND_CUDNN $tf | cut -d " " -f 2`
    dnn=`if [ "A$tmp" == "A" ]; then echo "**MISSING**"; else echo $tmp; fi`

    # t_o:      | Docker Tag | TensorFlow | OpenCV | Ubuntu | Github Link | OpenCV Build Info | TensorFlow Build Info |
    # n_t_o :   | Docker Tag | CUDA | CUDNN | TensorFlow | OpenCV | Ubuntu | Github Link | OpenCV Build Info | TensorFlow Build Info |
    #echo "[$v-$t]"
    line=`echo "$g" | perl -ne '@it = ($_ =~ m%^(.+)\:([\d\.]+\_)?([\d\.]+)\_([\d\.]+)\s(\d+)$%); $n = shift @it; $x = pop @it; ($c, $t, $o) = @it; $c=~s%\_$%%; $f= (($c ne "") ? "$c\_": "") . "$t\_$o"; print "| $f-$x "; print "| $c " if ($c != 0); print "| '$dnn' " if ($n =~ m%^cudnn_%); print "| $t | $o | '$ub' | [link]('$gh'$x) | [link]('$bla'$n-$f-$x/$n-$f-$x-OpenCV.txt) | [link]('$bla'$n-$f-$x/$n-$f-$x-TensorFlow.txt) | [link]('$bla'$n-$f-$x/$n-$f-$x-FFmpeg.txt) | [link]('$bla'$n-$f-$x/$n-$f-$x-PyTorch.txt) |\n";'`
    echo $line
    check=`echo $line | grep '**MISSING**' | wc -l`
    if [ $check == 1 ]; then
      missing=$((missing+1))
    fi
  else
    skipped=$((skipped+1))
  fi
done

if [ $skipped -gt 0 ]; then
  echo ""
  echo "!!!!! Warning: Skipped $skipped potential entries"
fi
if [ $missing -gt 0 ]; then
  echo ""
  echo "!!!!! Warning: $missing entry is missing a valid CV and/or TF value"
fi
