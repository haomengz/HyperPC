#!/bin/bash

if [ -z $1 ] ; then
    GPU=all
else
    GPU=$1
fi

PROJECT= #Path to the project
COMPLETION3D= #Path to the Completion3D dataset
MODLENET40= #Path to the ModelNet40 dataset
SHAPENET= #Path to the ShapeNet dataset
MVP= #Path to the MVP dataset

docker run -it --rm \
  --gpus '"device='$GPU'"' \
  -u $(id -u):$(id -g) \
  -v $COMPLETION3D:$PROJECT/data_root/Completion3D \
  -v $MODLENET40:$PROJECT/data_root/ModelNet40 \
  -v $SHAPENET:$PROJECT/data_root/ShapeNet \
  -v $MVP:$PROJECT/data_root/MVP \
  -v $(pwd):$PROJECT \
  -w $PROJECT \
  hyper_point_clouds