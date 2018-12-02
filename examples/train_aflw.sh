N_KEYPOINTS=$1
CELEBA_CHECKPOINT_PATH=$2 # path to the model checkpoint that was trained on celeba
python scripts/train.py --configs configs/paths/default.yaml configs/experiments/aflw-"$N_KEYPOINTS"pts-finetune.yaml --checkpoint "$CELEBA_CHECKPOINT_PATH" --restore-optim