#!/bin/bash
MEM=40960
N_GPUS=2

echo "running deeplab resnet  model"
bsub -n 1 -W 6:30 -R "rusage[mem=$MEM,ngpus_excl_p=$N_GPUS]" ./run_train.sh
