#!/bin/bash
#SBATCH --job-name=train2      # Job name
#SBATCH --mail-type=BEGIN,END,FAIL                   # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jaesung@robots.ox.ac.uk    # Where to send mail
#SBATCH --nodes=1                              # Node count
#SBATCH --ntasks=1                             # Total number of tasks across all nodes
#SBATCH --cpus-per-task=16                      # Number of CPU cores per task
#SBATCH --mem=250gb                             # Job memory request
#SBATCH --time=160:00:00                        # Time limit hrs:min:sec
#SBATCH --partition=gpu                        # Partition (compute (default) / gpu)
#SBATCH --gres=gpu:4                           # Requesting 1 GPUs       
#SBATCH --output=/work/jaesung/logs/%j.log
#SBATCH --constraint=p40
#SBATCH --exclude=gnodec1,gnodeb2

#CONFIG_FILE=configs/EPIC-KITCHENS/OMNIVORE_slowfast_frame_samp.yaml
CONFIG_FILE=configs/EPIC-KITCHENS/OMNIVORE_feature.yaml
DATA_PATH=/scratch/shared/beegfs/jaesung/dataset/epic-kitchens/dataset
ANNOTATIONS_DIR=/scratch/shared/beegfs/jaesung/dataset/epic-kitchens/epic-kitchens-100-annotations

TEST_LIST=frame_features/1sec/epic_frame_train_1sec_split2.pkl
TEST_BATCH_SIZE=8
OUTPUT_DIR=/work/jaesung/dataset/omnivore1sec/npy_visual_newsampling/trainset2
NUM_GPUS=4
NUM_ENSEMBLE_VIEWS=3

source ~/.bashrc
conda activate py39
export PYTHONPATH=/users/jaesung/Motionformer/slowfast:$PYTHONPATH

python tools/run_net.py \
  --cfg $CONFIG_FILE \
  NUM_GPUS $NUM_GPUS \
  OUTPUT_DIR $OUTPUT_DIR \
  TEST.BATCH_SIZE $TEST_BATCH_SIZE \
  TEST.NUM_ENSEMBLE_VIEWS $NUM_ENSEMBLE_VIEWS \
  EPICKITCHENS.VISUAL_DATA_DIR $DATA_PATH \
  EPICKITCHENS.ANNOTATIONS_DIR $ANNOTATIONS_DIR \
  EPICKITCHENS.TEST_LIST $TEST_LIST
