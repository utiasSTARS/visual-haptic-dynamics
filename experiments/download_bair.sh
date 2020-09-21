# Script taken from: https://github.com/edenton/svg
# Usage: sh experiments/data/download_bair.sh /path/to/data/
# Convert to pytorch format: python data/convert_bair.py --data_dir /path/to/data/

TARGET_DIR=$1
if [ -z $TARGET_DIR ]
then
  echo "Must specify target directory"
else
  mkdir $TARGET_DIR/
  URL=http://rail.eecs.berkeley.edu/datasets/bair_robot_pushing_dataset_v0.tar
  wget $URL -P $TARGET_DIR
  tar -xvf $TARGET_DIR/bair_robot_pushing_dataset_v0.tar -C $TARGET_DIR
fi