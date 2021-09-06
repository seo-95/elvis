#!/usr/bin/bash

#check if alive process "ps -elf | grep python"
#attach gdb to process "gdp -p <pid>"

#export QT_DEBUG_PLUGINS=1
#in order to attach the display you need to launch a background session with `ssh -Y`
export DISPLAY=localhost:10.0

export CONFIG_FOLDER=configs
export CHECKPOINTS_DIR=~/checkpoints/vqa
export OMP_NUM_THREADS=4

# 1 node with N gpu (total: N processes, one gpu each)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3,4

source ~/anaconda3/etc/profile.d/conda.sh
conda activate lvlab

python -m torch.distributed.launch --nproc_per_node=2 \
                                   --nnodes=1 \
                                   --node_rank=0 \
                                   --master_addr="127.0.0.1" \
                                   --master_port=6000 \
                                    train_dist.py \
                                    --config_file $CONFIG_FOLDER/cnnbert_vqa.yaml \
                                    --fp16 \
                                    --walltime 23:59:00 \
                                    --log_dir cnnbert_vqa

conda deactivate
