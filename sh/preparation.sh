#!/bin/bash

os_system=`uname`
if [ $os_system == "Darwin" ]; then
    realpath() {
        path=`eval echo "$1"`
        folder=$(dirname "$path")
        echo $(cd "$folder"; pwd)/$(basename "$path");
    }
fi

absolute_path=`pwd`/`dirname $0`
models_path=`realpath $absolute_path/../models`
mkdir -p $models_path
wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5 -P $models_path
wget https://www.dropbox.com/s/i2iljvtzca5yzoa/mask_rcnn_fcscsandwich_0030.h5 -P $models_path

logs_path=`realpath $absolute_path/../logs`
mkdir -p $logs_path
