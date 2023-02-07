#!/bin/bash
#SBATCH --job-name=Train4                 # Job name
#SBATCH --mail-type=ALL                        # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jaesung@robots.ox.ac.uk  # Where to send mail
#SBATCH --nodes=1                              # Node count
#SBATCH --cpus-per-task=16                     # Number of CPU cores per task
#SBATCH --mem=140GB                             # Job memory request
#SBATCH --time=100:00:00                        # Time limit hrs:min:sec
#SBATCH --gres=gpu:4                           # Requesting 1 GPUs       
#SBATCH --output=/jmain02/home/J2AD001/wwp01/jjh26-wwp01/logs/%j.log

module purge

# Run the application
nvidia-smi

#CONFIG_FILE=configs/EPIC-KITCHENS/OMNIVORE_slowfast_frame_samp.yaml
CONFIG_FILE=configs/EPIC-KITCHENS/OMNIVORE_feature.yaml
#DATA_PATH=/scratch/shared/beegfs/jaesung/dataset/epic-kitchens/dataset
ANNOTATIONS_DIR=/jmain02/home/J2AD001/wwp01/jjh26-wwp01/dataset/epic-kitchens-100-annotations
OUTPUT_DIR=/tmp/jaesung

echo "Extracting RGB Frames..."
cd /jmain02/home/J2AD001/wwp01/shared/data/epic-100/frames

for tar in P*/P??_*??.tar
do
    person=$(dirname $tar)
    tar_file=$(basename $tar)
    video=$(cut -d'.' -f1<<<$(basename $tar))
    folder="$OUTPUT_DIR/$person/rgb_frames/$video"
    mkdir -p $folder
    cp $tar $folder
    tar -xf $folder/$tar_file -C $folder
    rm -f $folder/$tar_file
done

DATA_PATH=$OUTPUT_DIR
TEST_LIST=frame_features/1sec/epic_frame_train_1sec_split4.pkl
TEST_BATCH_SIZE=16
OUTPUT_DIR=/jmain02/home/J2AD001/wwp01/jjh26-wwp01/dataset/feature_extraction/trainset4
NUM_GPUS=4
NUM_ENSEMBLE_VIEWS=3



# Back to the original
cd /jmain02/home/J2AD001/wwp01/jjh26-wwp01/ego_actrecog_analysis
source ~/.bashrc
conda activate huh
export PYTHONPATH=/jmain02/home/J2AD001/wwp01/jjh26-wwp01/ego_actrecog_analysis/slowfast:$PYTHONPATH

python tools/run_net.py \
  --cfg $CONFIG_FILE \
  NUM_GPUS $NUM_GPUS \
  OUTPUT_DIR $OUTPUT_DIR \
  TEST.BATCH_SIZE $TEST_BATCH_SIZE \
  TEST.NUM_ENSEMBLE_VIEWS $NUM_ENSEMBLE_VIEWS \
  EPICKITCHENS.VISUAL_DATA_DIR $DATA_PATH \
  EPICKITCHENS.ANNOTATIONS_DIR $ANNOTATIONS_DIR \
  EPICKITCHENS.TEST_LIST $TEST_LIST
