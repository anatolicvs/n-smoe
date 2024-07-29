#!/bin/bash

# nvidia/cuda:11.4.0-base-ubuntu20.04 nvidia-smi
# docker run -it --gpus all aytacozkan/cuda:latest 

docker run -v /home/ozkan/:/home/ozkan/ --gpus all aytacozkan/cuda:latest


