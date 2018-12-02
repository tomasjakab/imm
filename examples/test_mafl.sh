N_KEYPOINTS=$1
python scripts/test.py --experiment-name celeba-"$1"pts --train-dataset mafl --test-dataset mafl