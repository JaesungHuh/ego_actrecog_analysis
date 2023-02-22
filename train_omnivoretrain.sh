#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=END,FAIL                   # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jaesung@robots.ox.ac.uk    # Where to send mail
#SBATCH --gres=gpu:4
#SBATCH --job-name=mfHR16x2
#SBATCH --mem=200GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/work/jaesung/logs/%j.log
#SBATCH --partition=gpu
#SBATCH --constraint=gmem48G
#SBATCH --time=160:00:00

#CONFIG_FILE=configs/EK/motionformer_336_16x1.yaml
CONFIG_FILE=configs/EK/motionformer_336_16x2.yaml
#CONFIG_FILE=configs/EK/motionformer_224_16x2.yaml
#CHECKPOINT=/work/jaesung/pretrained/motionformer/k600_motionformer_224_16x4.pyth
#CHECKPOINT=/work/jaesung/pretrained/motionformer/k600_motionformer_336_16x4.pyth
#CHECKPOINT=/work/jaesung/checkpoints/epic-kitchens/8372_mfHR16x1/checkpoints/checkpoint_epoch_00012.pyth
CHECKPOINT=/work/jaesung/checkpoints/epic-kitchens/8373_mfHR16x2/checkpoints/checkpoint_epoch_00016.pyth
DATA_PATH=/scratch/shared/beegfs/jaesung/dataset/epic-kitchens/dataset
# TRAIN_LIST=frame_features/2sec/epic_frame_train_multilabel.pkl
# VAL_LIST=frame_features/2sec/epic_frame_val_multilabel.pkl
TRAIN_LIST=EPIC_100_train.pkl
VAL_LIST=EPIC_100_validation.pkl
ROOT_FOLDER=/work/jaesung/checkpoints/epic-kitchens
VISUAL_DATA_DIR=/scratch/shared/beegfs/jaesung/dataset/epic-kitchens/dataset
ANNOTATIONS_DIR=/scratch/shared/beegfs/jaesung/dataset/epic-kitchens/epic-kitchens-100-annotations/
LOG_PERIOD=2000
NUM_WORKERS=8
NUM_GPUS=4
BATCH_SIZE=8
WANDB_ENABLE=True

export PYTHONPATH=/users/jaesung/Motionformer/slowfast:$PYTHONPATH
source /users/jaesung/.bashrc
conda activate py39

export MASTER_ADDR=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:4}
export MASTER_PORT=19600

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# set the network interface
export NCCL_SOCKET_IFNAME=^docker0,lo
# echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES
# master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:4}
# dist_url="tcp://"
# dist_url+=$master_node
# dist_url+=:40000
dist_url="env://"
echo $dist_url

SAV_FOLDER="${ROOT_FOLDER}/${SLURM_JOB_ID}_mfHR16x2_resume"
mkdir -p ${SAV_FOLDER}

python tools/run_net.py \
  --cfg $CONFIG_FILE \
  --init_method $dist_url --num_shards 1 \
  WANDB.ENABLE ${WANDB_ENABLE} \
  DATA.PATH_TO_DATA_DIR $DATA_PATH \
  DATA_LOADER.NUM_WORKERS $NUM_WORKERS \
  TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
  TRAIN.BATCH_SIZE ${BATCH_SIZE} \
  NUM_GPUS ${NUM_GPUS} \
  EPICKITCHENS.VISUAL_DATA_DIR $VISUAL_DATA_DIR \
  EPICKITCHENS.ANNOTATIONS_DIR $ANNOTATIONS_DIR \
  EPICKITCHENS.TRAIN_LIST $TRAIN_LIST \
  EPICKITCHENS.VAL_LIST $VAL_LIST \
  OUTPUT_DIR ${SAV_FOLDER} \
  LOG_PERIOD $LOG_PERIOD
