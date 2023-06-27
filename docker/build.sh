#!/bin/bash
docker build \
    --network=host \
    --build-arg UNAME=haomeng \
    --build-arg GID=$(id -g) \
    --build-arg UID=$(id -u) \
    -t hyper_point_clouds \
    .