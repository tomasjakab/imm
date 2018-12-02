N_KEYPOINTS=$1
python scripts/train.py --configs configs/paths/default.yaml configs/experiments/celeba-"$1"pts.yaml