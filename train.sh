#!/bin/bash

CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python main.py \
--config_path ./cfgs/config.yaml \