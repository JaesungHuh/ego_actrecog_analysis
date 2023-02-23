#!/bin/bash
#SBATCH --job-name=288                 # Job name
#SBATCH --mail-type=ALL                        # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jaesung@robots.ox.ac.uk  # Where to send mail
#SBATCH --nodes=1                              # Node count
#SBATCH --cpus-per-task=8                     # Number of CPU cores per task
#SBATCH --mem=150GB                             # Job memory request
#SBATCH --time=24:00:00                        # Time limit hrs:min:sec
#SBATCH --gres=gpu:8                          # Requesting 1 GPUs       
#SBATCH --output=/jmain02/home/J2AD001/wwp01/jjh26-wwp01/logs/%j.log

module purge

# Run the application
nvidia-smi

CONFIG_FILE=configs/EPIC-KITCHENS/OMNIVORE_288.yaml
ANNOTATIONS_DIR=/jmain02/home/J2AD001/wwp01/jjh26-wwp01/dataset/epic-kitchens-100-annotations
OUTPUT_DIR=/raid/local_scratch/jjh26-wwp01/jaesung
ROOT_FOLDER=/jmain02/home/J2AD001/wwp01/jjh26-wwp01/checkpoints

echo "Extracting RGB Frames..."
cd /jmain02/home/J2AD001/wwp01/shared/data/epic-100/frames
for tar in P*/rgb_frames/P??_*??.tar
do
    person=$(dirname $tar)
    tar_file=$(basename $tar)
    video=$(cut -d'.' -f1<<<$(basename $tar))
    folder="$OUTPUT_DIR/$person/rgb_frames/$video"
    mkdir -p $folder
    cp -ru $tar $folder
    tar -xf $folder/$tar_file -C $folder
    rm -f $folder/$tar_file
done

DATA_PATH=${OUTPUT_DIR}
TRAIN_LIST=EPIC_100_train_omnivore.pkl
VAL_LIST=EPIC_100_validation_omnivore.pkl
SAV_FOLDER="${ROOT_FOLDER}/omnivore_288"
TRAIN_BATCH_SIZE=8
NUM_GPUS=8
NUM_ENSEMBLE_VIEWS=1
LOG_PERIOD=1000
NUM_WORKERS=8
BASE_LR=1e-3
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
cd /jmain02/home/J2AD001/wwp01/jjh26-wwp01/ego_actrecog_analysis
source ~/.bashrc
conda activate huh
export PYTHONPATH=/jmain02/home/J2AD001/wwp01/jjh26-wwp01/ego_actrecog_analysis/slowfast:$PYTHONPATH

WANDB_MODE=offline python tools/run_net.py \
  --cfg $CONFIG_FILE \
  --init_method $dist_url \
  WANDB.ENABLE $WANDB_ENABLE \
  DATA.PATH_TO_DATA_DIR $DATA_PATH \
  DATA_LOADER.NUM_WORKERS $NUM_WORKERS \
  TRAIN.BATCH_SIZE $TRAIN_BATCH_SIZE \
  NUM_GPUS $NUM_GPUS \
  SOLVER.BASE_LR $BASE_LR \
  EPICKITCHENS.VISUAL_DATA_DIR $DATA_PATH \
  EPICKITCHENS.ANNOTATIONS_DIR $ANNOTATIONS_DIR \
  EPICKITCHENS.TRAIN_LIST $TRAIN_LIST \
  EPICKITCHENS.VAL_LIST $VAL_LIST \
  OUTPUT_DIR $SAV_FOLDER \
  LOG_PERIOD $LOG_PERIOD \
  TEST.ENABLE False \
  TRAIN.ENABLE True 
