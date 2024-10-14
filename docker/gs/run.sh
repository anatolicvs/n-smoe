#!/bin/bash

docker run -v /home/ozkan/:/home/ozkan/ --gpus 1 -itd --name dev_env aytacozkan/gs:1.0.1

docker exec -it dev_env /bin/bash