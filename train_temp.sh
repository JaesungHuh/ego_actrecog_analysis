#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --error=checkpoint/%j.err
#SBATCH --gres=gpu:2
#SBATCH --job-name=omni32x1
#SBATCH --mem=100GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/work/jaesung/logs/%j.log
#SBATCH --partition=gpu
#SBATCH --constraint=gmem48G
#SBATCH --time=160:00:00

#CONFIG_FILE=configs/EK/omnivore_32x1_halfsec.yaml
CONFIG_FILE=configs/EK/omnivore_32x1.yaml
#CHECKPOINT=/work/jaesung/checkpoints/epic-kitchens/9355_omnivore_32x1_resume/checkpoints/checkpoint_epoch_00020.pyth
#CHECKPOINT=/work/jaesung/checkpoints/epic-kitchens/9356_omnivore_32x0.5_resume/checkpoints/checkpoint_epoch_00020.pyth
DATA_PATH=/work/jaesung/dataset/epic_video
TRAIN_LIST=EPIC_100_train_omnivore.pkl
VAL_LIST=EPIC_100_validation_omnivore.pkl
ROOT_FOLDER=/work/jaesung/checkpoints/epic-kitchens
WANDB_ENABLE=True

source /users/jaesung/.bashrc
conda activate py39
export PYTHONPATH=/users/jaesung/Omnivore_train/slowfast:$PYTHONPATH

export MASTER_ADDR=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:4}
export MASTER_PORT=19500

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# set the network interface
export NCCL_SOCKET_IFNAME=^docker0,lo
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES
master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:4}
dist_url="tcp://"
dist_url+=$master_node
dist_url+=:40000
echo $dist_url

SAV_FOLDER="${ROOT_FOLDER}/${SLURM_JOB_ID}_omnivore_32x1_lowlr2"
mkdir -p ${SAV_FOLDER}

LOG_PERIOD=1000

python tools/run_net.py \
  --cfg $CONFIG_FILE \
  --init_method $dist_url --num_shards 1 \
  DATA.PATH_TO_DATA_DIR $DATA_PATH \
  WANDB.ENABLE ${WANDB_ENABLE} \
  TRAIN.BATCH_SIZE 8 \
  NUM_GPUS 2 \
  LOG_PERIOD $LOG_PERIOD \
  EPICKITCHENS.TRAIN_LIST $TRAIN_LIST \
  EPICKITCHENS.VAL_LIST $VAL_LIST \
  OUTPUT_DIR ${SAV_FOLDER}
