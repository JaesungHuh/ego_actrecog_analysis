#!/bin/bash
#SBATCH --job-name=288                 # Job name
#SBATCH --mail-user=jaesung@robots.ox.ac.uk  # Where to send mail
#SBATCH --nodes=1                              # Node count
#SBATCH --cpus-per-task=16                    # Number of CPU cores per task
#SBATCH --mem=100GB                             # Job memory request
#SBATCH --partition=ddp-2way
#SBATCH --time=72:00:00                        # Time limit hrs:min:sec
#SBATCH --gres=gpu:2                           # Requesting 1 GPUs       
#SBATCH --output=/work/jaesung/logs/%j.log
#SBATCH --constraint=["a6000"|"a40"|"rtx8k"]
module purge

# Run the application
nvidia-smi

# For debug
#OUTPUT_DIR=/scratch/shared/beegfs/jaesung/dataset/epic-kitchens/dataset

# For triton
OUTPUT_DIR=/tmp/jaesung
CONFIG_FILE=configs/EPIC-KITCHENS/OMNIVORE_288.yaml
ANNOTATIONS_DIR=/scratch/shared/beegfs/jaesung/dataset/epic-kitchens/epic-kitchens-100-annotations
ROOT_FOLDER=/scratch/shared/beegfs/jaesung/checkpoints/omnivore

echo "Extracting RGB Frames..."
cd /datasets/EpicKitchens-100

for tar in P*/rgb_frames/P??_*??.tar
do
    person=$(dirname $tar | awk -F "/" '{print $1}')
    tar_file=$(basename $tar)
    video=$(cut -d'.' -f1<<<$(basename $tar))
    folder="$OUTPUT_DIR/$person/rgb_frames/$video"
    mkdir -p $folder
    cp -ru --verbose $tar $folder
    tar -xf $folder/$tar_file -C $folder
    rm -f $folder/$tar_file
done

DATA_PATH=${OUTPUT_DIR}
TRAIN_LIST=EPIC_100_train_omnivore.pkl
VAL_LIST=EPIC_100_validation_omnivore.pkl
SAV_FOLDER="${ROOT_FOLDER}/omnivore_imagenet21k_288"
TRAIN_BATCH_SIZE=2
NUM_GPUS=2
NUM_ENSEMBLE_VIEWS=1
LOG_PERIOD=1000
NUM_WORKERS=8
WANDB_ENABLE=True

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

# Back to the original
cd /users/jaesung/ego_temp
source ~/.bashrc
conda activate py39
export PYTHONPATH=/users/jaesung/ego_temp/slowfast:$PYTHONPATH

python tools/run_net.py \
  --cfg $CONFIG_FILE \
  --init_method $dist_url \
  WANDB.ENABLE $WANDB_ENABLE \
  DATA.PATH_TO_DATA_DIR $DATA_PATH \
  DATA_LOADER.NUM_WORKERS $NUM_WORKERS \
  TRAIN.BATCH_SIZE $TRAIN_BATCH_SIZE \
  NUM_GPUS $NUM_GPUS \
  EPICKITCHENS.VISUAL_DATA_DIR $DATA_PATH \
  EPICKITCHENS.ANNOTATIONS_DIR $ANNOTATIONS_DIR \
  EPICKITCHENS.TRAIN_LIST $TRAIN_LIST \
  EPICKITCHENS.VAL_LIST $VAL_LIST \
  OUTPUT_DIR $SAV_FOLDER \
  LOG_PERIOD $LOG_PERIOD \
  TEST.ENABLE False \
  TRAIN.ENABLE True 
