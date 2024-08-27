GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR= # your master address
MASTER_PORT= # your master port
NNODES=3
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH= # the directory of query_train.jsonl
SAVE_PATH= # your saving path 
IMAGE_PATH= # the training image directory
EPOCH=50
RESUME_PATH= # pre-trained visualized bge weights
SAVE_STEPS=100
GROUP_SIZE=5 # = one (positive sample) + number (of hard negative samples)
BSZ_PERGPU=30
LR=2e-5

Training_Dir= #your training dir
cd $Training_Dir
# Data and model


mkdir $SAVE_PATH
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

export LAUNCHER="torchrun \
    $DISTRIBUTED_ARGS \
    "

full_options="
  --output_dir $SAVE_PATH \
  --bge_model_name_or_path  BAAI/bge-base-en-v1.5 \
  --visual_model_name_or_path  EVA02-CLIP-B-16 \
  --dataloader_num_workers 1  \
  --train_data $DATA_PATH \
  --train_data_image $IMAGE_PATH \
  --train_group_size $GROUP_SIZE
  --learning_rate $LR \
  --fp16 \
  --per_device_train_batch_size $BSZ_PERGPU \
  --dataloader_drop_last True \
  --normlized True \
  --temperature 0.02 \
  --logging_steps 10 \
  --num_train_epochs $EPOCH \
  --negatives_cross_device \
  --train_text_tower True  \
  --train_vision_tower True \
  --resume_path $RESUME_PATH \
  --save_steps $SAVE_STEPS \
  --deepspeed ./EVA-CLIP/rei/training/deepspeed_config.json \
  --gradient_checkpointing \
  "

run_cmd="$LAUNCHER -m downstream.run_ds_cirr ${full_options}"
echo ${run_cmd}
eval ${run_cmd} 2>&1 | tee $SAVE_PATH/output_$NODE_RANK.log



set +x

