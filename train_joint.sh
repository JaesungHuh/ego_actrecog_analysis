#!/bin/bash
#SBATCH --job-name=Time-Prompt                 # Job name
#SBATCH --mail-type=ALL                        # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jaesung@robots.ox.ac.uk  # Where to send mail
#SBATCH --nodes=1                              # Node count
#SBATCH --cpus-per-task=8                      # Number of CPU cores per task
#SBATCH --mem=70GB                             # Job memory request
#SBATCH --time=48:00:00                        # Time limit hrs:min:sec
#SBATCH --gres=gpu:1                           # Requesting 1 GPUs       
#SBATCH --output=/jmain02/home/J2AD001/wwp01/jjh26-wwp01/logs/%j.log

module purge

# Run the application
nvidia-smi

jaesung_id=jjh26-wwp01

# Visual data
VIDEO_DATA_PATH=/raid/local_scratch/${jaesung_id}/Time_prompting/EPIC_100/npy_visual/omnivore
VIDEO_TRAIN_ACTION_PICKLE=/raid/local_scratch/${jaesung_id}/Time_prompting/EPIC_100/EPIC_100_train.pkl
VIDEO_VAL_ACTION_PICKLE=/raid/local_scratch/${jaesung_id}/Time_prompting/EPIC_100/EPIC_100_validation.pkl
VIDEO_TRAIN_CONTEXT_PICKLE=/raid/local_scratch/${jaesung_id}/Time_prompting/EPIC_100/epic_frame/epic_frame_train_1sec.pkl
VIDEO_VAL_CONTEXT_PICKLE=/raid/local_scratch/${jaesung_id}/Time_prompting/EPIC_100/epic_frame/epic_frame_val_1sec.pkl

# Audio data
AUDIO_DATASET='epic-sounds'
AUDIO_DATA_PATH=/raid/local_scratch/${jaesung_id}/Time_prompting/EPIC_Sounds/npyfiles/auditory_slowfast
#AUDIO_TRAIN_ACTION_PICKLE=/raid/local_scratch/${jaesung_id}/Time_prompting/EPIC_Sounds/EPIC_Sounds_train.pkl
#AUDIO_VAL_ACTION_PICKLE=/raid/local_scratch/${jaesung_id}/Time_prompting/EPIC_Sounds/EPIC_Sounds_validation.pkl
AUDIO_TRAIN_ACTION_PICKLE=/raid/local_scratch/${jaesung_id}/Time_prompting/EPIC_Sounds/after_cleaning/EPIC_SOUNDS_train_clean.pkl
AUDIO_VAL_ACTION_PICKLE=/raid/local_scratch/${jaesung_id}/Time_prompting/EPIC_Sounds/after_cleaning/EPIC_SOUNDS_validation_clean.pkl
AUDIO_TRAIN_CONTEXT_PICKLE=/raid/local_scratch/${jaesung_id}/Time_prompting/EPIC_100/epic_frame/epic_frame_train_1sec.pkl
AUDIO_VAL_CONTEXT_PICKLE=/raid/local_scratch/${jaesung_id}/Time_prompting/EPIC_100/epic_frame/epic_frame_val_1sec.pkl

# Loss
MIXUP_ALPHA=0.2
AUDIO_ALPHA=1.0
LAMBDA_DRLOC=0.3
M_DRLOC=32

# Optimizer
LR=0.0001
OPTIMIZER=adamw
WEIGHT_DECAY=0.0005
FINETUNE_LR=1e-4

# Training
BATCH_SIZE=64
VISUAL_EPOCHS=0
AUDIO_EPOCHS=0
FINETUNE_EPOCHS=150
DROP_AUDIO_PROB=0.0
TOK_SIZE=2

# Model
DROPOUT=0.1
NUM_LAYERS=4
TIME_NORMALIZE=True
SHARE_LAYER=False
CLASSIFICATION_MODE='all'
CLASSIFICATION_WEIGHTS='constant'

# Data
INPUT_LENGTH=50
MODALITY='audio_visual'
CUT_TIME=5
CONTEXT_HOP=2

# Misc
PRINT_FREQ=200
OUTPUT_DIR=output/featurelayernorm_mixup_crossmodaldrloc
NUM_WORKERS=8
PIN_MEMORY=False
NUM_GPUS=1

echo "Copying Data"
cp -ru /jmain02/home/J2AD001/wwp01/jjh26-wwp01/dataset/Time_prompting /raid/local_scratch/${jaesung_id}/

echo "RAID Directory"
ls /raid/local_scratch/${jaesung_id}/

source ~/.bashrc
conda activate huh
# export PYTHONPATH=/jmain02/home/J2AD001/wwp01/jxc31-wwp01/libs:/jmain02/home/J2AD001/wwp01/jxc31-wwp01/Time-Prompting-Embed:$PYTHONPATH

WANDB_MODE=offline python run_net.py \
        --num-gpus $NUM_GPUS \
        --video_data_path $VIDEO_DATA_PATH \
        --video_train_action_pickle $VIDEO_TRAIN_ACTION_PICKLE \
        --video_val_action_pickle $VIDEO_VAL_ACTION_PICKLE \
        --video_train_context_pickle $VIDEO_TRAIN_CONTEXT_PICKLE \
        --video_val_context_pickle $VIDEO_VAL_CONTEXT_PICKLE \
        --audio_data_path $AUDIO_DATA_PATH \
        --audio_train_action_pickle $AUDIO_TRAIN_ACTION_PICKLE \
        --audio_val_action_pickle $AUDIO_VAL_ACTION_PICKLE \
        --audio_train_context_pickle $AUDIO_TRAIN_CONTEXT_PICKLE \
        --audio_val_context_pickle $AUDIO_VAL_CONTEXT_PICKLE \
        --modality $MODALITY \
        --classification_mode $CLASSIFICATION_MODE \
        --classification_weights $CLASSIFICATION_WEIGHTS \
        --context_hop $CONTEXT_HOP \
        --input_length $INPUT_LENGTH \
        --tok-size $TOK_SIZE \
        -p $PRINT_FREQ \
        --output_dir $OUTPUT_DIR \
        --num_layers $NUM_LAYERS \
        --lambda_drloc $LAMBDA_DRLOC \
        --mixup_alpha $MIXUP_ALPHA \
        --audio_alpha $AUDIO_ALPHA \
        --m_drloc $M_DRLOC \
        --drop_audio_prob $DROP_AUDIO_PROB \
        -j $NUM_WORKERS \
        --pin-memory $PIN_MEMORY \
        --share_layer $SHARE_LAYER \
        --audio_dataset $AUDIO_DATASET \
        --lr $LR \
        --lr_steps 50 75 \
        -b $BATCH_SIZE \
        --cut_time $CUT_TIME \
        --dropout $DROPOUT \
        --optimizer $OPTIMIZER \
        --visual_epochs $VISUAL_EPOCHS \
        --audio_epochs $AUDIO_EPOCHS \
        --finetune_epochs $FINETUNE_EPOCHS \
        --weight_decay $WEIGHT_DECAY \
        --time_normalize $TIME_NORMALIZE \
        --finetune_lr $FINETUNE_LR
